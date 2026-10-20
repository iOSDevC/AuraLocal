import Foundation
import Jinja
import LocalLLMClient

// MARK: - LlamaChatMessage

/// One message of a conversation as the GGUF backends render it.
struct LlamaChatMessage: Sendable, Equatable {
    enum Speaker: String, Sendable {
        case system, user, assistant, tool
    }

    var role: Speaker
    var content: String
    /// Calls this assistant turn made; rendered natively by templates with tool support, as <tool_call> text otherwise.
    var toolCalls: [LLMToolCall] = []

    static func system(_ content: String) -> LlamaChatMessage { .init(role: .system, content: content) }
    static func user(_ content: String) -> LlamaChatMessage { .init(role: .user, content: content) }
    static func assistant(_ content: String) -> LlamaChatMessage { .init(role: .assistant, content: content) }
    static func tool(_ content: String) -> LlamaChatMessage { .init(role: .tool, content: content) }

    static func assistant(_ content: String, toolCalls: [LLMToolCall]) -> LlamaChatMessage {
        .init(role: .assistant, content: content, toolCalls: toolCalls)
    }

    /// `[["role": …, "content": …]]` as `InferenceBackend.generate(messages:)` receives it; an unknown or missing
    /// role is user text.
    static func list(from raw: [[String: String]]) -> [LlamaChatMessage] {
        raw.map { entry in
            LlamaChatMessage(role: Speaker(rawValue: entry["role"] ?? "") ?? .user, content: entry["content"] ?? "")
        }
    }

    /// The single-prompt entry points: an optional system prompt and one user turn.
    static func list(prompt: String, systemPrompt: String?) -> [LlamaChatMessage] {
        (systemPrompt.map { [.system($0)] } ?? []) + [.user(prompt)]
    }
}

// MARK: - LlamaChatNormalizer

enum LlamaChatNormalizer {
    /// Reshapes a history into what every chat template accepts: at most one leading system message, then
    /// alternating user/assistant turns (tool results stay where they are).
    ///
    /// - Every system message (a stored summary can sit mid-history) merges into the leading one, in order.
    /// - Assistant turns before the first user turn join the system text; alternation templates raise on them.
    /// - Empty assistant turns (a failed reply) are dropped.
    /// - Consecutive user or assistant turns merge.
    ///
    /// An assistant turn that made tool calls is kept as it is: never dropped, joined or merged.
    static func normalize(_ messages: [LlamaChatMessage]) -> [LlamaChatMessage] {
        var systemParts: [String] = []
        var turns: [LlamaChatMessage] = []
        for message in messages {
            switch message.role {
            case .system:
                if !isBlank(message.content) { systemParts.append(message.content) }
            case .assistant where message.toolCalls.isEmpty && isBlank(message.content):
                continue
            case .assistant where message.toolCalls.isEmpty && !turns.contains(where: { $0.role == .user }):
                systemParts.append(message.content)
            case .user, .assistant:
                if let last = turns.last, last.role == message.role, last.toolCalls.isEmpty,
                   message.toolCalls.isEmpty {
                    turns[turns.count - 1].content = last.content + "\n\n" + message.content
                } else {
                    turns.append(message)
                }
            case .tool:
                turns.append(message)
            }
        }
        let system = systemParts.isEmpty ? [] : [LlamaChatMessage.system(systemParts.joined(separator: "\n\n"))]
        return system + turns
    }

    /// Drops the oldest exchange: its user turn up to the next one. `nil` when only the last user turn is left,
    /// which must reach the model.
    static func droppingOldestExchange(_ messages: [LlamaChatMessage]) -> [LlamaChatMessage]? {
        let userIndices = messages.indices.filter { messages[$0].role == .user }
        guard userIndices.count >= 2 else { return nil }
        var trimmed = messages
        trimmed.removeSubrange(userIndices[0]..<userIndices[1])
        return trimmed
    }

    static func isBlank(_ text: String) -> Bool {
        text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

// MARK: - LlamaPromptRenderer

/// Renders a conversation with the GGUF's own chat template the way LocalLLMClient's `.chat` input does (same
/// Jinja engine, context variables, message shapes and tool instructions), so the text a turn decodes is what
/// `LLMSession` would have decoded for the same roles.
struct LlamaPromptRenderer: Sendable {
    /// `nil` renders a role-labelled transcript.
    let template: String?
    let specialTokens: [String: String]
    let tools: [AnyLLMTool]
    private let parsed: Result<Template, any Error>?

    enum RenderError: LocalizedError {
        case templateFailed(String)

        var errorDescription: String? {
            switch self {
            case .templateFailed(let reason): "The model's chat template could not render the conversation: \(reason)"
            }
        }
    }

    init(template: String?, specialTokens: [String: String] = [:], tools: [AnyLLMTool] = []) {
        let source = template.map(Self.patched)
        self.template = source
        self.specialTokens = specialTokens
        self.tools = tools
        parsed = source.map { text in Result<Template, any Error> { try Template(text) } }
    }

    init(metadata: GGUFChatMetadata, tools: [any LLMTool]) {
        self.init(template: metadata.template, specialTokens: metadata.specialTokens,
                  tools: tools.map { AnyLLMTool($0) })
    }

    /// Same template and tokens, no tool instructions: what a constrained call sends.
    var withoutTools: LlamaPromptRenderer {
        tools.isEmpty ? self : LlamaPromptRenderer(template: template, specialTokens: specialTokens)
    }

    /// How message content reaches the template. LocalLLMClient picks one per template family.
    enum ContentShape: Equatable {
        /// `content` is a string (ChatML and the fallback processor).
        case text
        /// `content` is `[{"type": "text", "text": …}]` (Gemma, SmolVLM, Qwen2-VL).
        case parts
        /// User content is a string, every other role uses parts (Llama 3.2 Vision).
        case partsExceptUser
    }

    /// Mirrors `MessageProcessorFactory.createAutoProcessor`.
    static func contentShape(for template: String) -> ContentShape {
        if template.contains("<|turn>") { return .parts }
        if template.contains("<|im_start|>") && template.contains("<end_of_utterance>") { return .parts }
        if template.contains("content[i].type == 'image'") || template.contains("<vision>") { return .parts }
        if template.contains("<|begin_of_text|>") && template.contains("<|start_header_id|>") { return .partsExceptUser }
        if template.contains("<start_of_image>") { return .parts }
        return .text
    }

    /// How a tool round reaches the template.
    private enum ToolTurnStyle {
        /// Assistant `tool_calls` and `tool` turns, which templates with tool support render in their own format.
        case native
        /// Each round as one assistant turn: its calls as `<tool_call>` text, its results as `<tool_response>` (the
        /// wrapping LocalLLMClient gives the last result).
        case text
    }

    /// Tries LocalLLMClient's message shape first, then plain strings, then the system prompt folded into the
    /// first user turn. A render only counts when every message's text reaches the prompt: some templates drop the
    /// system prompt silently, and Llama 3 text templates blank assistant turns sent as parts.
    ///
    /// A tool round renders natively when the template mentions `tools`; if every native attempt throws, the
    /// attempts run again with the text form.
    func render(_ messages: [LlamaChatMessage]) throws -> String {
        try render(messages, toolRoundsRemain: false)
    }

    /// As ``render(_:)``. After a tool result the tool instructions are sent again only while `toolRoundsRemain`
    /// (another round may still run); on a template without tool support they are the only listing of the tools.
    func render(_ messages: [LlamaChatMessage], toolRoundsRemain: Bool) throws -> String {
        let normalized = LlamaChatNormalizer.normalize(messages)
        guard let template, parsed != nil else { return Self.transcript(normalized) }
        let compiled = try compiledTemplate()
        let attempts = Self.attempts(for: normalized, template: template)
        let nativeTools = template.contains("tools")
        if nativeTools && Self.hasToolRounds(normalized),
           let rendered = try? render(attempts, with: compiled, nativeTools: nativeTools, style: .native,
                                      keepsInstructions: toolRoundsRemain) {
            return rendered
        }
        return try render(attempts, with: compiled, nativeTools: nativeTools, style: .text,
                          keepsInstructions: toolRoundsRemain)
    }

    // MARK: Rendering

    private func compiledTemplate() throws -> Template {
        guard let parsed else { throw RenderError.templateFailed("the model has no chat template") }
        do {
            return try parsed.get()
        } catch {
            throw RenderError.templateFailed(String(describing: error))
        }
    }

    private static func attempts(for normalized: [LlamaChatMessage],
                                 template: String) -> [(messages: [LlamaChatMessage], shape: ContentShape)] {
        let preferred = contentShape(for: template)
        var attempts: [(messages: [LlamaChatMessage], shape: ContentShape)] = [(normalized, preferred)]
        if preferred != .text { attempts.append((normalized, .text)) }
        if normalized.first?.role == .system {
            attempts.append((foldingSystemIntoFirstUser(normalized), .text))
        }
        return attempts
    }

    private func render(_ attempts: [(messages: [LlamaChatMessage], shape: ContentShape)], with compiled: Template,
                        nativeTools: Bool, style: ToolTurnStyle, keepsInstructions: Bool) throws -> String {
        var lastFailure = "no attempt rendered"
        for (index, attempt) in attempts.enumerated() {
            do {
                let rendered = try render(attempt.messages, with: compiled, nativeTools: nativeTools,
                                          shape: attempt.shape, style: style, keepsInstructions: keepsInstructions)
                let isLast = index == attempts.count - 1
                if isLast || Self.keepsEveryMessage(attempt.messages, in: rendered) {
                    return rendered
                }
                lastFailure = "the template dropped part of the conversation"
            } catch {
                lastFailure = "\(error)"
            }
        }
        throw RenderError.templateFailed(lastFailure)
    }

    private static func hasToolRounds(_ messages: [LlamaChatMessage]) -> Bool {
        messages.contains { $0.role == .tool || !$0.toolCalls.isEmpty }
    }

    private func render(_ messages: [LlamaChatMessage], with compiled: Template, nativeTools: Bool,
                        shape: ContentShape, style: ToolTurnStyle, keepsInstructions: Bool) throws -> String {
        let payload = try toolProcessed(try messages.map { try Self.payload(for: $0, shape: shape, style: style) },
                                        nativeSupport: nativeTools, shape: shape, style: style,
                                        keepsInstructions: keepsInstructions)
        var context: [String: Value] = [
            "add_generation_prompt": .boolean(true),
            "messages": try Value(any: payload),
        ]
        try context.merge(specialTokens.mapValues { try Value(any: $0) }) { _, new in new }
        if !tools.isEmpty && nativeTools {
            context["tools"] = try Value(any: tools.map { $0.toOAICompatJSON() })
        }

        // LocalLLMClient sets these on the environment, which this Jinja ignores; setting them the same way keeps
        // the two renderings identical if that ever changes.
        let environment = Environment()
        environment.lstripBlocks = true
        environment.trimBlocks = true
        return try compiled.render(context, environment: environment)
    }

    private static func payload(for message: LlamaChatMessage, shape: ContentShape,
                                style: ToolTurnStyle) throws -> [String: any Sendable] {
        guard !message.toolCalls.isEmpty else {
            return ["role": message.role.rawValue, "content": content(message.content, role: message.role, shape: shape)]
        }
        switch style {
        case .native:
            return ["role": message.role.rawValue, "content": content(message.content, role: message.role, shape: shape),
                    "tool_calls": try nativeCalls(message.toolCalls)]
        case .text:
            let text = callTranscript(text: message.content, calls: message.toolCalls)
            return ["role": message.role.rawValue, "content": content(text, role: message.role, shape: shape)]
        }
    }

    /// The OpenAI shape chat templates read, with `arguments` as an object so `tojson` writes it back verbatim.
    private static func nativeCalls(_ calls: [LLMToolCall]) throws -> Value {
        let entries: [Any?] = try calls.map { call in
            ["type": "function", "id": call.id,
             "function": ["name": call.name, "arguments": try nativeArguments(call.arguments)] as [String: Any?]]
                as [String: Any?]
        }
        return try Value(any: entries)
    }

    /// Native values, not `JSONSerialization`'s: its `NSNumber` booleans would reach the template as integers.
    private static func nativeArguments(_ arguments: String) throws -> [String: Any?] {
        guard case .object(let members)? = JSONValue.parse(arguments) else {
            throw RenderError.templateFailed("tool call arguments are not a JSON object")
        }
        return members.mapValues(nativeValue)
    }

    private static func nativeValue(_ value: JSONValue) -> Any? {
        switch value {
        case .null: return nil
        case .bool(let flag): return flag
        case .number(let number):
            if let integer = number.intValue { return integer }
            return Double(number.description)
        case .string(let text): return text
        case .array(let elements): return elements.map(nativeValue)
        case .object(let members): return members.mapValues(nativeValue)
        }
    }

    /// The format the injected tool instructions ask the model to write.
    static func callTranscript(text: String, calls: [LLMToolCall]) -> String {
        let blocks = calls.map { call in
            let arguments = LlamaChatNormalizer.isBlank(call.arguments) ? "{}" : call.arguments
            return "<tool_call>\n{\"name\": \(JSONValue.quote(call.name)), \"arguments\": \(arguments)}\n</tool_call>"
        }
        return text + (text.isEmpty ? "" : "\n") + blocks.joined(separator: "\n")
    }

    private static func content(_ text: String, role: LlamaChatMessage.Speaker, shape: ContentShape) -> any Sendable {
        let usesParts = switch shape {
        case .text: false
        case .parts: true
        case .partsExceptUser: role != .user
        }
        return usesParts ? [["type": "text", "text": text] as [String: String]] : text
    }

    /// `StandardToolInstructionProcessor.processMessages`, which LocalLLMClient applies whenever tools exist. A system
    /// message it has to add gets the template's content shape; LocalLLMClient always uses parts there. Unlike it,
    /// `keepsInstructions` keeps the instructions after a tool result too.
    private func toolProcessed(_ messages: [[String: any Sendable]], nativeSupport: Bool, shape: ContentShape,
                               style: ToolTurnStyle, keepsInstructions: Bool) throws -> [[String: any Sendable]] {
        guard !tools.isEmpty else { return messages }
        var processed = messages
        // Sorted: the instructions are rendered again every round, and the same tools must give the same text.
        let toolsJSON = try tools.toOAICompatJSONString(options: [.sortedKeys])
        let lastIsTool = processed.last?["role"] as? String == LlamaChatMessage.Speaker.tool.rawValue

        if nativeSupport {
            if let index = Self.systemIndex(in: processed) {
                processed[index]["tools"] = toolsJSON
            } else {
                processed.insert(["role": "system", "content": Self.content("", role: .system, shape: shape),
                                  "tools": toolsJSON], at: 0)
            }
        }
        if !lastIsTool || keepsInstructions {
            let instructions = Self.toolInstructions(toolsJSON)
            if let index = Self.systemIndex(in: processed) {
                processed[index]["content"] = Self.appending(instructions, to: processed[index]["content"])
            } else {
                processed.insert(["role": "system", "content": Self.content(instructions, role: .system, shape: shape)],
                                 at: 0)
            }
        }
        return style == .text ? Self.foldingToolRounds(processed) : processed
    }

    /// Each round (its calls and results) becomes one assistant turn: alternation templates such as Gemma 3's raise
    /// on a `tool` role and on two assistant turns in a row.
    private static func foldingToolRounds(_ messages: [[String: any Sendable]]) -> [[String: any Sendable]] {
        let assistant = LlamaChatMessage.Speaker.assistant.rawValue
        var folded: [[String: any Sendable]] = []
        for (index, message) in messages.enumerated() {
            var entry = message
            if entry["role"] as? String == LlamaChatMessage.Speaker.tool.rawValue {
                entry["role"] = assistant
                entry["content"] = index == messages.count - 1
                    ? wrappingToolResponse(entry["content"]) : taggingToolResponse(entry["content"])
            }
            if entry["role"] as? String == assistant, let previous = folded.last,
               previous["role"] as? String == assistant {
                folded[folded.count - 1]["content"] = joining(previous["content"], entry["content"])
            } else {
                folded.append(entry)
            }
        }
        return folded
    }

    private static func joining(_ first: (any Sendable)?, _ second: (any Sendable)?) -> any Sendable {
        if var parts = first as? [[String: String]], !parts.isEmpty {
            parts[0]["text"] = joinedLines(text(of: first), text(of: second))
            return parts
        }
        return joinedLines(text(of: first), text(of: second))
    }

    private static func text(of content: (any Sendable)?) -> String {
        (content as? String) ?? (content as? [[String: String]])?.first?["text"] ?? ""
    }

    private static func joinedLines(_ first: String, _ second: String) -> String {
        first.isEmpty ? second : second.isEmpty ? first : first + "\n" + second
    }

    private static func taggingToolResponse(_ content: (any Sendable)?) -> any Sendable {
        if var parts = content as? [[String: String]], let text = parts.first?["text"] {
            parts[0]["text"] = "<tool_response>\n\(text)\n</tool_response>"
            return parts
        }
        return "<tool_response>\n\((content as? String) ?? "")\n</tool_response>"
    }

    private static func systemIndex(in messages: [[String: any Sendable]]) -> Int? {
        messages.firstIndex { $0["role"] as? String == LlamaChatMessage.Speaker.system.rawValue }
    }

    private static func appending(_ instructions: String, to content: (any Sendable)?) -> any Sendable {
        if let text = content as? String {
            return text.isEmpty ? instructions : text + "\n\n" + instructions
        }
        if var parts = content as? [[String: String]], let first = parts.first?["text"] {
            parts[0]["text"] = first.isEmpty ? instructions : first + "\n\n" + instructions
            return parts
        }
        return instructions
    }

    private static func wrappingToolResponse(_ content: (any Sendable)?) -> any Sendable {
        if var parts = content as? [[String: String]], let text = parts.first?["text"] {
            parts[0]["text"] = "You reply to the user using the following results of the invoked function:\n"
                + "<tool_response>\n\(text)\n</tool_response>"
            return parts
        }
        return "<tool_response>\n\((content as? String) ?? "")\n</tool_response>"
    }

    private static func toolInstructions(_ toolsJSON: String) -> String {
        """
        If you decide to invoke any of the function(s), you MUST put it in the format of
        <tool_call>
        {"name": function name, "arguments": dictionary of argument name and its value}
        </tool_call>\n
        You SHOULD NOT include any other text in the response if you call a function
        \(toolsJSON)\n
        """
    }

    // MARK: Fallbacks

    /// Looks for the end of each message only: templates trim content, and reasoning templates drop everything up
    /// to `</think>` from assistant turns.
    static func keepsEveryMessage(_ messages: [LlamaChatMessage], in rendered: String) -> Bool {
        messages.allSatisfy { message in
            let content = message.content
            let answer = content.range(of: "</think>", options: .backwards).map { content[$0.upperBound...] } ?? content[...]
            let tail = answer.trimmingCharacters(in: .whitespacesAndNewlines).suffix(64)
            return tail.isEmpty || rendered.contains(tail)
        }
    }

    /// For templates without a system role (Gemma 2, some Mistral builds).
    static func foldingSystemIntoFirstUser(_ messages: [LlamaChatMessage]) -> [LlamaChatMessage] {
        guard let system = messages.first, system.role == .system else { return messages }
        var rest = Array(messages.dropFirst())
        if let index = rest.firstIndex(where: { $0.role == .user }) {
            rest[index].content = system.content + "\n\n" + rest[index].content
        } else {
            rest.insert(.user(system.content), at: 0)
        }
        return rest
    }

    /// LocalLLMClient falls back to a Gemma-style template without role markers; a labelled transcript keeps the
    /// speakers apart.
    static func transcript(_ messages: [LlamaChatMessage]) -> String {
        let lines = messages.map { message in
            let content = message.toolCalls.isEmpty ? message.content
                : callTranscript(text: message.content, calls: message.toolCalls)
            return "\(message.role.rawValue.capitalized): \(content)"
        }
        return (lines + ["Assistant:"]).joined(separator: "\n\n")
    }

    /// Qwen2.5 GGUFs escape the tool-call example as `{{"name": …}}` inside a string literal, which renders the
    /// braces doubled; small models copy them and no tool call parses.
    /// One trailing newline goes too, as in Python Jinja (swift-jinja keeps it): Qwen's generation prompt ended
    /// `assistant\n\n`, which no later prompt extends.
    static func patched(_ template: String) -> String {
        let repaired = template.replacingOccurrences(
            of: #"{{\"name\": <function-name>, \"arguments\": <args-json-object>}}"#,
            with: #"{\"name\": <function-name>, \"arguments\": <args-json-object>}"#
        )
        guard let last = repaired.last, last == "\n" || last == "\r\n" || last == "\r" else { return repaired }
        return String(repaired.dropLast())
    }
}
