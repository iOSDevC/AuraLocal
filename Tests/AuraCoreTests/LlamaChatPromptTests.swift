import XCTest
@testable import AuraCore

/// Always-on: how the GGUF backends turn a conversation into the prompt they decode, read a GGUF's template and
/// special tokens, and keep the prompt inside the KV window. No model needed.
final class LlamaChatPromptTests: XCTestCase {

    // MARK: - Roles

    func testRawMessagesKeepTheirRolesAndUnknownRolesBecomeUserText() {
        let messages = LlamaChatMessage.list(from: [
            ["role": "system", "content": "s"],
            ["role": "assistant", "content": "a"],
            ["role": "narrator", "content": "n"],
            ["content": "no role"],
        ])
        XCTAssertEqual(messages, [.system("s"), .assistant("a"), .user("n"), .user("no role")])
    }

    func testSinglePromptEntryPointsSendTheSystemPromptAsItsOwnRole() {
        XCTAssertEqual(LlamaChatMessage.list(prompt: "hi", systemPrompt: "be brief"), [.system("be brief"), .user("hi")])
        XCTAssertEqual(LlamaChatMessage.list(prompt: "hi", systemPrompt: nil), [.user("hi")])
    }

    // MARK: - Normalisation

    func testEverySystemMessageMergesIntoOneLeadingSystemMessage() {
        let normalized = LlamaChatNormalizer.normalize([
            .user("u1"), .system("first"), .assistant("a1"), .system("summary"), .user("u2"),
        ])
        XCTAssertEqual(normalized, [.system("first\n\nsummary"), .user("u1"), .assistant("a1"), .user("u2")])
    }

    func testAssistantTurnsBeforeTheFirstUserTurnJoinTheSystemText() {
        let normalized = LlamaChatNormalizer.normalize([.system("rules"), .assistant("Hi! Ask me."), .user("u1")])
        XCTAssertEqual(normalized, [.system("rules\n\nHi! Ask me."), .user("u1")])
    }

    func testConsecutiveSameRoleTurnsMergeAndEmptyRepliesAreDropped() {
        let normalized = LlamaChatNormalizer.normalize([
            .user("u1"), .assistant("  "), .user("u2"), .assistant("a1"), .assistant("a2"), .user("u3"),
        ])
        XCTAssertEqual(normalized, [.user("u1\n\nu2"), .assistant("a1\n\na2"), .user("u3")])
    }

    func testABlankAssistantWithToolCallsIsKeptAndNotMerged() {
        let first = LLMToolCall(id: "1", name: "echo", arguments: "{}")
        let second = LLMToolCall(id: "2", name: "echo", arguments: "{}")
        let normalized = LlamaChatNormalizer.normalize([
            .user("u1"), .assistant("", toolCalls: [first]), .tool("r1"), .assistant("so far"),
            .assistant(" ", toolCalls: [second]), .assistant("after"),
        ])
        XCTAssertEqual(normalized, [
            .user("u1"), .assistant("", toolCalls: [first]), .tool("r1"), .assistant("so far"),
            .assistant(" ", toolCalls: [second]), .assistant("after"),
        ])
        let leading = LlamaChatNormalizer.normalize([.system("s"), .assistant("", toolCalls: [first]), .user("u")])
        XCTAssertEqual(leading, [.system("s"), .assistant("", toolCalls: [first]), .user("u")],
                       "a call before the first user turn does not join the system text")
    }

    func testToolResultsStayInPlace() {
        let normalized = LlamaChatNormalizer.normalize([.user("u1"), .tool("r1"), .tool("r2")])
        XCTAssertEqual(normalized, [.user("u1"), .tool("r1"), .tool("r2")])
    }

    func testDroppingTheOldestExchangeKeepsTheSystemPromptAndTheLastUserTurn() {
        let history: [LlamaChatMessage] = [.system("s"), .user("u1"), .assistant("a1"), .user("u2"), .assistant("a2"),
                                           .user("u3")]
        XCTAssertEqual(LlamaChatNormalizer.droppingOldestExchange(history),
                       [.system("s"), .user("u2"), .assistant("a2"), .user("u3")])
        XCTAssertNil(LlamaChatNormalizer.droppingOldestExchange([.system("s"), .user("only")]))
    }

    // MARK: - Rendering

    private static let chatML = """
    {%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}\
    {%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}
    """

    func testChatTemplateReceivesRealRolesInOrder() throws {
        let rendered = try LlamaPromptRenderer(template: Self.chatML)
            .render([.system("Be terse."), .user("Hi"), .assistant("Hello."), .user("Again")])
        XCTAssertEqual(rendered, """
        <|im_start|>system\nBe terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n\
        <|im_start|>assistant\nHello.<|im_end|>\n<|im_start|>user\nAgain<|im_end|>\n<|im_start|>assistant\n
        """)
    }

    func testBOSComesFromTheSpecialTokensExactlyOnce() throws {
        let renderer = LlamaPromptRenderer(template: "{{ bos_token }}" + Self.chatML, specialTokens: ["bos_token": "<s>"])
        let rendered = try renderer.render([.user("Hi")])
        XCTAssertTrue(rendered.hasPrefix("<s><|im_start|>user"), rendered)
        XCTAssertEqual(rendered.components(separatedBy: "<s>").count - 1, 1)
    }

    /// Gemma 2's template: raises on a system role and on broken alternation, and calls the assistant `model`.
    private static let gemma2 = """
    {{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}\
    {% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\
    {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}\
    {% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}\
    {% endif %}{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}{% endfor %}\
    {% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}
    """

    func testGemmaStyleTemplatesWithoutASystemRoleGetItInTheFirstUserTurn() throws {
        let renderer = LlamaPromptRenderer(template: Self.gemma2, specialTokens: ["bos_token": "<bos>"])
        let rendered = try renderer.render([
            .system("Be terse."), .assistant("Welcome."), .user("Hi"), .user("there"), .assistant("Hello."), .user("Again"),
        ])
        XCTAssertEqual(rendered, """
        <bos><start_of_turn>user\nBe terse.\n\nWelcome.\n\nHi\n\nthere<end_of_turn>\n\
        <start_of_turn>model\nHello.<end_of_turn>\n<start_of_turn>user\nAgain<end_of_turn>\n<start_of_turn>model\n
        """)
    }

    func testATemplateThatSilentlyDropsTheSystemPromptStillReceivesIt() throws {
        let template = "{%- for message in messages %}{%- if message.role != 'system' %}"
            + "{{- message.role + ':' + message.content + '|' }}{%- endif %}{%- endfor %}"
        let rendered = try LlamaPromptRenderer(template: template).render([.system("Speak Spanish."), .user("Hi")])
        XCTAssertEqual(rendered, "user:Speak Spanish.\n\nHi|")
    }

    func testPartsShapedTemplatesGetContentPartsAndKeepTheSystemPrompt() throws {
        // `<|turn>` is what LocalLLMClient keys its Gemma 4 processor on.
        let template = """
        {%- for message in messages %}{{- '<|turn>' + message.role + ':' }}\
        {%- for part in message.content %}{{- part.text }}{%- endfor %}{%- endfor %}
        """
        XCTAssertEqual(LlamaPromptRenderer.contentShape(for: template), .parts)
        let rendered = try LlamaPromptRenderer(template: template).render([.system("S"), .user("U")])
        XCTAssertEqual(rendered, "<|turn>system:S<|turn>user:U")
    }

    /// Llama 3 text templates match LocalLLMClient's Llama 3.2 Vision processor, which sends system and assistant
    /// turns as parts; `parts | trim` renders them empty.
    func testAStringTemplateThatBlanksPartsGetsStrings() throws {
        let template = "<|begin_of_text|>{%- for message in messages %}{{- '<|start_header_id|>' + message.role"
            + " + '<|end_header_id|>' + message.content | trim + '<|eot_id|>' }}{%- endfor %}"
        XCTAssertEqual(LlamaPromptRenderer.contentShape(for: template), .partsExceptUser)
        let rendered = try LlamaPromptRenderer(template: template).render([.system("S"), .user("U"), .assistant("A"),
                                                                          .user("V")])
        XCTAssertEqual(rendered, "<|begin_of_text|><|start_header_id|>system<|end_header_id|>S<|eot_id|>"
                       + "<|start_header_id|>user<|end_header_id|>U<|eot_id|>"
                       + "<|start_header_id|>assistant<|end_header_id|>A<|eot_id|>"
                       + "<|start_header_id|>user<|end_header_id|>V<|eot_id|>")
    }

    func testEveryMessageCheckReadsOnlyTheTrimmedEndOfEachMessage() {
        let reasoning = LlamaChatMessage.assistant("<think>long reasoning</think>\n\nThe answer is 4.  ")
        XCTAssertTrue(LlamaPromptRenderer.keepsEveryMessage([.user("2+2?"), reasoning, .user(" ")],
                                                            in: "user: 2+2? assistant: The answer is 4."))
        XCTAssertFalse(LlamaPromptRenderer.keepsEveryMessage([.system("S"), .user("U")], in: "user: U"))
    }

    func testContentShapeFollowsLocalLLMClientsProcessorChoice() {
        XCTAssertEqual(LlamaPromptRenderer.contentShape(for: Self.chatML), .text)
        XCTAssertEqual(LlamaPromptRenderer.contentShape(for: "<start_of_image>"), .parts)
        XCTAssertEqual(LlamaPromptRenderer.contentShape(for: "<|im_start|> <end_of_utterance>"), .parts)
        XCTAssertEqual(LlamaPromptRenderer.contentShape(for: "<|begin_of_text|><|start_header_id|>"), .partsExceptUser)
    }

    func testAGGUFWithoutATemplateGetsARoleLabelledTranscript() throws {
        let rendered = try LlamaPromptRenderer(template: nil).render([.system("S"), .user("U"), .assistant("A"), .user("V")])
        XCTAssertEqual(rendered, "System: S\n\nUser: U\n\nAssistant: A\n\nUser: V\n\nAssistant:")
    }

    func testAnUnrenderableTemplateThrowsInsteadOfSendingSomethingElse() {
        XCTAssertThrowsError(try LlamaPromptRenderer(template: "{{ raise_exception('nope') }}").render([.user("U")]))
        XCTAssertThrowsError(try LlamaPromptRenderer(template: "{% for %}").render([.user("U")])) { error in
            XCTAssertTrue(error is LlamaPromptRenderer.RenderError, "\(error)")
        }
    }

    // MARK: - Tools

    private struct EchoTool: LLMTool {
        let name = "echo"
        let description = "Echo text."
        struct Arguments: Decodable, ToolSchemaGeneratable {
            let text: String
            static var argumentsSchema: LLMToolArgumentsSchema { ["text": .string(description: "Text")] }
        }
        func call(arguments: Arguments) async throws -> ToolOutput { ToolOutput(data: ["text": arguments.text]) }
    }

    func testToolsReachNativeTemplatesAndTheSystemInstructions() throws {
        let template = "{%- if tools %}{{- 'TOOLS:' + tools[0].function.name + '|' }}{%- endif %}" + Self.chatML
        let renderer = LlamaPromptRenderer(template: template, tools: [AnyLLMTool(EchoTool())])
        let rendered = try renderer.render([.system("Be terse."), .user("Echo hi")])
        XCTAssertTrue(rendered.hasPrefix("TOOLS:echo|<|im_start|>system\nBe terse.\n\nIf you decide to invoke"), rendered)
        XCTAssertTrue(rendered.contains(#""name":"echo""#), rendered)
    }

    func testWithoutASystemTurnToolInstructionsBecomeOne() throws {
        let renderer = LlamaPromptRenderer(template: Self.chatML, tools: [AnyLLMTool(EchoTool())])
        let rendered = try renderer.render([.user("Echo hi")])
        XCTAssertTrue(rendered.hasPrefix("<|im_start|>system\nIf you decide to invoke"), rendered)
    }

    func testTheLastToolResultIsSentAsAnAssistantToolResponse() throws {
        let renderer = LlamaPromptRenderer(template: Self.chatML, tools: [AnyLLMTool(EchoTool())])
        let rendered = try renderer.render([.user("Echo hi"), .tool("text: hi")])
        XCTAssertTrue(rendered.contains("<|im_start|>assistant\n<tool_response>\ntext: hi\n</tool_response><|im_end|>"),
                      rendered)
        XCTAssertFalse(rendered.contains("If you decide to invoke"), "instructions only precede a fresh request")
    }

    func testWhileRoundsRemainTheToolInstructionsFollowAResultToo() throws {
        let renderer = LlamaPromptRenderer(template: Self.chatML, tools: [AnyLLMTool(EchoTool())])
        let round: [LlamaChatMessage] = [.user("Echo hi, then echo bye"),
                                         .assistant("<tool_call>\n{\"name\": \"echo\", \"arguments\": {}}\n</tool_call>"),
                                         .tool(#"{"text":"hi"}"#)]
        let next = try renderer.render(round, toolRoundsRemain: true)
        XCTAssertTrue(next.hasPrefix("<|im_start|>system\nIf you decide to invoke"), next)
        XCTAssertTrue(next.contains(#""name":"echo""#), "a template without tool support lists the tools only there")
        XCTAssertEqual(try renderer.render([.user("Echo hi")]).prefix(while: { $0 != "[" }),
                       next.prefix(while: { $0 != "[" }), "the same instructions as the first reply's")
        XCTAssertFalse(try renderer.render(round, toolRoundsRemain: false).contains("If you decide to invoke"),
                       "the reply after the last round is steered to answer")
    }

    func testTheToolListIsTheSameEveryRender() throws {
        let renderer = LlamaPromptRenderer(template: Self.chatML, tools: [AnyLLMTool(EchoTool())])
        let request: [LlamaChatMessage] = [.user("Echo hi")]
        let first = try renderer.render(request)
        for _ in 0..<5 { XCTAssertEqual(try renderer.render(request), first) }
        XCTAssertTrue(first.contains(#"[{"function":{"description":"Echo text.","name":"echo","parameters":"#), first)
    }

    func testATemplatesTrailingNewlineIsDroppedAsJinjaDoes() throws {
        let rendered = try LlamaPromptRenderer(template: Self.chatML + "\n").render([.user("Hi")])
        XCTAssertTrue(rendered.hasSuffix("<|im_start|>assistant\n"), rendered.debugDescription)
        XCTAssertFalse(rendered.hasSuffix("\n\n"), "a reply's turn renders as `assistant\\n` + text, so the prompt must too")
        XCTAssertEqual(LlamaPromptRenderer.patched("A\r\n"), "A")
        XCTAssertEqual(LlamaPromptRenderer.patched("A\n\n"), "A\n", "only one, like Jinja")
    }

    /// ChatML that renders `tool_calls` and keeps the `tool` role, the way Qwen2.5's template does.
    private static let nativeChatML = """
    {%- if tools %}{{- 'TOOLS|' }}{%- endif %}{%- for message in messages %}\
    {%- if message.role == 'assistant' and message.tool_calls %}{{- '<|im_start|>assistant\\n' + message.content }}\
    {%- for tool_call in message.tool_calls %}{{- '<call id=' + tool_call.id + '>' + tool_call.function.name + ':' \
    + (tool_call.function.arguments | tojson) + '</call>' }}{%- endfor %}{{- '<|im_end|>\\n' }}\
    {%- else %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}{%- endif %}\
    {%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}
    """

    private static let echoCall = LLMToolCall(
        id: "c1", name: "echo", arguments: #"{"text": "hi", "loud": true, "n": 2, "ratio": 0.5, "none": null}"#
    )

    func testNativeTemplatesGetToolCallsAndToolTurns() throws {
        let renderer = LlamaPromptRenderer(template: Self.nativeChatML, tools: [AnyLLMTool(EchoTool())])
        let rendered = try renderer.render([.user("Echo hi"), .assistant("", toolCalls: [Self.echoCall]),
                                            .tool(#"{"text":"hi"}"#)])
        XCTAssertTrue(rendered.contains("<|im_start|>assistant\n<call id=c1>echo:"
                                        + #"{"loud":true,"n":2,"none":null,"ratio":0.5,"text":"hi"}</call>"#), rendered)
        XCTAssertTrue(rendered.contains("<|im_start|>tool\n{\"text\":\"hi\"}<|im_end|>"), rendered)
        XCTAssertFalse(rendered.contains("<tool_response>"), rendered)
        XCTAssertFalse(rendered.contains("<tool_call>"), rendered)
        XCTAssertFalse(rendered.contains("If you decide to invoke"), "instructions only precede a fresh request")
    }

    func testNativeRenderingThatThrowsFallsBackToText() throws {
        let template = "{%- if tools %}{%- endif %}{%- for message in messages %}{%- if message.role == 'tool' %}"
            + "{{ raise_exception('no tool role') }}{%- endif %}{{- '<|im_start|>' + message.role + '\\n' "
            + "+ message.content + '<|im_end|>\\n' }}{%- endfor %}"
        let renderer = LlamaPromptRenderer(template: template, tools: [AnyLLMTool(EchoTool())])
        let rendered = try renderer.render([.user("Echo hi"), .assistant("", toolCalls: [Self.echoCall]),
                                            .tool("r1")])
        XCTAssertTrue(rendered.contains("<|im_start|>assistant\n<tool_call>\n{\"name\": \"echo\", \"arguments\": "
                                        + Self.echoCall.arguments + "}\n</tool_call>\n<tool_response>\nr1\n"
                                        + "</tool_response><|im_end|>"), rendered)
        XCTAssertFalse(rendered.contains("<|im_start|>tool"), rendered)
    }

    func testNonObjectArgumentsFallBackToText() throws {
        let renderer = LlamaPromptRenderer(template: Self.nativeChatML, tools: [AnyLLMTool(EchoTool())])
        let odd = LLMToolCall(id: "c2", name: "echo", arguments: "[1, 2]")
        let rendered = try renderer.render([.user("Echo hi"), .assistant("", toolCalls: [odd]), .tool("r1")])
        XCTAssertTrue(rendered.contains("<tool_call>\n{\"name\": \"echo\", \"arguments\": [1, 2]}\n</tool_call>"),
                      rendered)
        XCTAssertFalse(rendered.contains("<call id="), rendered)
    }

    func testTextTemplatesGetToolCallTranscripts() throws {
        let renderer = LlamaPromptRenderer(template: Self.chatML, tools: [AnyLLMTool(EchoTool())])
        let call = LLMToolCall(id: "c1", name: "echo", arguments: #"{"text": "hi"}"#)
        let blank = LLMToolCall(id: "c2", name: "echo", arguments: "")
        let rendered = try renderer.render([.user("Echo hi"), .assistant("Sure.", toolCalls: [call, blank]),
                                            .tool(#"{"text":"hi"}"#), .tool("{}")])
        XCTAssertTrue(rendered.contains("""
        <|im_start|>assistant\nSure.\n<tool_call>\n{"name": "echo", "arguments": {"text": "hi"}}\n</tool_call>\n\
        <tool_call>\n{"name": "echo", "arguments": {}}\n</tool_call>\n<tool_response>\n{"text":"hi"}\n\
        </tool_response>\n<tool_response>\n{}\n</tool_response><|im_end|>\n<|im_start|>assistant\n
        """), rendered)
        XCTAssertFalse(rendered.contains("<|im_start|>tool"), rendered)
    }

    /// Gemma 3's template: folds the system prompt into the first user turn, takes parts, raises on broken
    /// alternation.
    private static let gemma3 = """
    {{ bos_token }}{%- if messages[0]['role'] == 'system' -%}{%- if messages[0]['content'] is string -%}\
    {%- set first_user_prefix = messages[0]['content'] + '\\n\\n' -%}{%- else -%}\
    {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\\n\\n' -%}{%- endif -%}\
    {%- set loop_messages = messages[1:] -%}{%- else -%}{%- set first_user_prefix = "" -%}\
    {%- set loop_messages = messages -%}{%- endif -%}{%- for message in loop_messages -%}\
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}\
    {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}{%- endif -%}\
    {%- if (message['role'] == 'assistant') -%}{%- set role = "model" -%}{%- else -%}\
    {%- set role = message['role'] -%}{%- endif -%}\
    {{ '<start_of_turn>' + role + '\\n' + (first_user_prefix if loop.first else "") }}\
    {%- if message['content'] is string -%}{{ message['content'] | trim }}\
    {%- elif message['content'] is iterable -%}{%- for item in message['content'] -%}\
    {%- if item['type'] == 'image' -%}{{ '<start_of_image>' }}{%- elif item['type'] == 'text' -%}\
    {{ item['text'] | trim }}{%- endif -%}{%- endfor -%}{%- else -%}\
    {{ raise_exception("Invalid content type") }}{%- endif -%}{{ '<end_of_turn>\\n' }}{%- endfor -%}\
    {%- if add_generation_prompt -%}{{'<start_of_turn>model\\n'}}{%- endif -%}
    """

    private static let hiCall = LLMToolCall(id: "c1", name: "echo", arguments: #"{"text": "hi"}"#)
    private static let byeCall = LLMToolCall(id: "c2", name: "echo", arguments: #"{"text": "bye"}"#)

    private func assertAlternationTemplateRenders(_ messages: [LlamaChatMessage], rounds: [(LLMToolCall, String)],
                                                  line: UInt = #line) throws {
        let renderer = LlamaPromptRenderer(template: Self.gemma3, specialTokens: ["bos_token": "<bos>"],
                                           tools: [AnyLLMTool(EchoTool())])
        let rendered = try renderer.render(messages)
        XCTAssertEqual(rendered.components(separatedBy: "<start_of_turn>model\n").count - 1, 2,
                       "one model turn for the rounds, then the generation prompt: \(rendered)", line: line)
        XCTAssertFalse(rendered.contains("<start_of_turn>tool"), rendered, line: line)
        for (call, result) in rounds {
            XCTAssertTrue(rendered.contains("{\"name\": \"echo\", \"arguments\": \(call.arguments)}"), rendered,
                          line: line)
            XCTAssertTrue(rendered.contains("<tool_response>\n\(result)\n</tool_response>"), rendered, line: line)
        }
    }

    func testAlternationTemplatesGetEachTextRoundAsOneAssistantTurn() throws {
        let one: [LlamaChatMessage] = [.user("Echo hi"), .assistant("", toolCalls: [Self.hiCall]),
                                       .tool(#"{"text":"hi"}"#)]
        try assertAlternationTemplateRenders(one, rounds: [(Self.hiCall, #"{"text":"hi"}"#)])
        try assertAlternationTemplateRenders([.system("Be terse.")] + one, rounds: [(Self.hiCall, #"{"text":"hi"}"#)])
        let renderer = LlamaPromptRenderer(template: Self.gemma3, specialTokens: ["bos_token": "<bos>"],
                                           tools: [AnyLLMTool(EchoTool())])
        XCTAssertEqual(try renderer.render([.system("Be terse.")] + one), """
        <bos><start_of_turn>user\nBe terse.\n\nEcho hi<end_of_turn>\n<start_of_turn>model\n<tool_call>\n\
        {"name": "echo", "arguments": {"text": "hi"}}\n</tool_call>\n\
        You reply to the user using the following results of the invoked function:\n\
        <tool_response>\n{"text":"hi"}\n</tool_response><end_of_turn>\n<start_of_turn>model\n
        """)
    }

    func testAlternationTemplatesGetSeveralTextRoundsAsOneAssistantTurn() throws {
        let two: [LlamaChatMessage] = [
            .user("Echo hi, then bye"), .assistant("", toolCalls: [Self.hiCall]), .tool(#"{"text":"hi"}"#),
            .assistant("", toolCalls: [Self.byeCall]), .tool(#"{"text":"bye"}"#),
        ]
        let rounds = [(Self.hiCall, #"{"text":"hi"}"#), (Self.byeCall, #"{"text":"bye"}"#)]
        try assertAlternationTemplateRenders(two, rounds: rounds)
        try assertAlternationTemplateRenders([.system("Be terse.")] + two, rounds: rounds)
    }

    func testATemplatelessGGUFWritesCallsIntoTheTranscript() throws {
        let call = LLMToolCall(id: "c1", name: "echo", arguments: #"{"text": "hi"}"#)
        let rendered = try LlamaPromptRenderer(template: nil).render([.user("U"), .assistant("", toolCalls: [call]),
                                                                      .tool("R")])
        XCTAssertEqual(rendered, """
        User: U\n\nAssistant: <tool_call>\n{"name": "echo", "arguments": {"text": "hi"}}\n</tool_call>\n\n\
        Tool: R\n\nAssistant:
        """)
    }

    func testAConstrainedCallRendersWithoutToolInstructions() throws {
        let renderer = LlamaPromptRenderer(template: Self.chatML, tools: [AnyLLMTool(EchoTool())])
        let rendered = try renderer.withoutTools.render(LlamaChatMessage.list(prompt: "hi", systemPrompt: "be brief"))
        XCTAssertEqual(rendered, "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"
                       + "<|im_start|>assistant\n")
    }

    func testQwensDoubledToolCallBracesAreRepaired() {
        let qwen = #"{{- "<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>" }}"#
        XCTAssertEqual(LlamaPromptRenderer.patched(qwen),
                       #"{{- "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>" }}"#)
    }

    // MARK: - GGUF metadata

    func testMetadataReadsTheTemplateAndSpecialTokenTexts() throws {
        let data = GGUFFixture.make(strings: [
            ("tokenizer.ggml.model", "gpt2"),
            ("tokenizer.chat_template", "{{ bos_token }}"),
        ], ids: [
            ("tokenizer.ggml.bos_token_id", 3),
            ("tokenizer.ggml.eos_token_id", 4),
        ], tokens: ["<unk>", "a", "b", "<bos>", "<eos>"])
        let metadata = try GGUFChatMetadata.parse(data)
        XCTAssertEqual(metadata.template, "{{ bos_token }}")
        XCTAssertEqual(metadata.specialTokens["bos_token"], "<bos>")
        XCTAssertEqual(metadata.specialTokens["eos_token"], "<eos>")
        XCTAssertEqual(metadata.specialTokens["cls_token"], "<bos>")
        XCTAssertEqual(metadata.specialTokens["unk_token"], "<unk>")
        XCTAssertEqual(metadata.specialTokens["sep_token"], "<unk>", "an unset id reads token 0, as LocalLLMClient does")
        XCTAssertEqual(metadata.specialTokens["mask_token"], "")
    }

    func testSpecialTokensDeepInALargeVocabularyAreFound() throws {
        let tokens = (0..<5000).map { "t\($0)" }
        let data = GGUFFixture.make(strings: [("tokenizer.ggml.model", "gpt2")],
                                    ids: [("tokenizer.ggml.eos_token_id", 4999), ("tokenizer.ggml.padding_token_id", 4321)],
                                    tokens: tokens)
        let metadata = try GGUFChatMetadata.parse(data)
        XCTAssertEqual(metadata.specialTokens["eos_token"], "t4999")
        XCTAssertEqual(metadata.specialTokens["pad_token"], "t4321")
    }

    func testMissingOrOutOfRangeIdsFallBackToLlamaCppsDefaults() throws {
        let tokens = (0..<12).map { "t\($0)" }
        let data = GGUFFixture.make(strings: [("tokenizer.ggml.model", "gpt2")],
                                    ids: [("tokenizer.ggml.eos_token_id", 99)], tokens: tokens)
        let metadata = try GGUFChatMetadata.parse(data)
        XCTAssertNil(metadata.template)
        XCTAssertEqual(metadata.specialTokens["bos_token"], "t11")
        XCTAssertEqual(metadata.specialTokens["eos_token"], "t11")
    }

    func testAnEmptyTemplateCountsAsNone() throws {
        let data = GGUFFixture.make(strings: [("tokenizer.ggml.model", "llama"), ("tokenizer.chat_template", "")],
                                    ids: [], tokens: ["<unk>", "<s>", "</s>"])
        let metadata = try GGUFChatMetadata.parse(data)
        XCTAssertNil(metadata.template)
        XCTAssertEqual(metadata.specialTokens["bos_token"], "<s>")
        XCTAssertEqual(metadata.specialTokens["eos_token"], "</s>")
    }

    func testOtherKeysOfEveryTypeAreSkipped() throws {
        let data = GGUFFixture.make(strings: [("tokenizer.chat_template", "T")], ids: [], tokens: ["<unk>"],
                                    extras: GGUFFixture.everyOtherType)
        XCTAssertEqual(try GGUFChatMetadata.parse(data).template, "T")
    }

    func testSentencePieceDefaultsAndGLMsMissingBOS() {
        XCTAssertEqual(GGUFChatMetadata.TokenIDs.defaults(tokenizerModel: "llama", preTokenizer: nil),
                       .init(bos: 1, eos: 2, sep: -1, pad: -1))
        XCTAssertEqual(GGUFChatMetadata.TokenIDs.defaults(tokenizerModel: "gpt2", preTokenizer: "glm4").bos, -1)
    }

    func testNonGGUFAndTruncatedBytesAreRejected() {
        XCTAssertThrowsError(try GGUFChatMetadata.parse(Data("nope".utf8))) { error in
            XCTAssertEqual(error as? GGUFChatMetadata.ReadError, .notGGUF)
        }
        let whole = GGUFFixture.make(strings: [("tokenizer.chat_template", "T")], ids: [], tokens: ["<unk>"])
        XCTAssertThrowsError(try GGUFChatMetadata.parse(whole.prefix(whole.count - 3))) { error in
            XCTAssertEqual(error as? GGUFChatMetadata.ReadError, .truncated)
        }
    }

    // MARK: - Fitting the window

    func testOnlyContextSizeErrorsCountAsOverflowsAndTheirSizeIsRead() {
        let overflow = LLMError.failedToDecode(reason: "context size exceeded[1024 < 1082]")
        XCTAssertEqual(GGUFPromptFitter.overflowedTokens(in: overflow, windowSize: 1024), 1082)
        let wrapped = LLMError.failedToDecode(reason: "Failed to decode response: context size exceeded[1024 < 1500]")
        XCTAssertEqual(GGUFPromptFitter.overflowedTokens(in: wrapped, windowSize: 1024), 1500)
        let unsized = LLMError.failedToDecode(reason: "context size exceeded")
        XCTAssertEqual(GGUFPromptFitter.overflowedTokens(in: unsized, windowSize: 1024), 1025)
        XCTAssertNil(GGUFPromptFitter.overflowedTokens(in: LLMError.failedToDecode(reason: "batch decode failed"),
                                                       windowSize: 1024))
        XCTAssertNil(GGUFPromptFitter.overflowedTokens(in: CancellationError(), windowSize: 1024))
    }

    private static let history: [LlamaChatMessage] = [
        .system("S"), .user(String(repeating: "a", count: 400)), .assistant(String(repeating: "b", count: 400)),
        .user(String(repeating: "c", count: 400)), .assistant(String(repeating: "d", count: 400)), .user("last"),
    ]

    private static func byteCount(_ messages: [LlamaChatMessage]) -> String {
        messages.map(\.content).joined()
    }

    func testNothingIsTrimmedBeforeTheFirstOverflow() throws {
        let fitter = GGUFPromptFitter(windowSize: 100)
        let fitted = try fitter.fitted(Self.history, replyBudget: 64, render: Self.byteCount)
        XCTAssertEqual(fitted.history, Self.history)
    }

    func testAfterAnOverflowTheOldestExchangesGoUntilTheEstimateFits() throws {
        var fitter = GGUFPromptFitter(windowSize: 512)
        let prompt = Self.byteCount(Self.history)   // 1605 bytes
        XCTAssertTrue(fitter.learn(from: LLMError.failedToDecode(reason: "context size exceeded[512 < 802]"),
                                   promptBytes: prompt.utf8.count))
        XCTAssertEqual(try XCTUnwrap(fitter.tokensPerByte), 802.0 / 1605.0, accuracy: 1e-9)
        // 805 bytes ≈ 403 tokens + a 64-token reply fits in 512; 1605 bytes did not.
        let fitted = try fitter.fitted(Self.history, replyBudget: 64, render: Self.byteCount)
        XCTAssertEqual(fitted.history, [.system("S"), .user(String(repeating: "c", count: 400)),
                                        .assistant(String(repeating: "d", count: 400)), .user("last")])
        XCTAssertEqual(fitted.prompt, Self.byteCount(fitted.history))
    }

    func testTheReplyReserveIsTheBudgetCappedAtAQuarterOfTheWindow() {
        let fitter = GGUFPromptFitter(windowSize: 1024)
        XCTAssertEqual(fitter.replyReserve(forBudget: 48), 48)
        XCTAssertEqual(fitter.replyReserve(forBudget: 4096), 256)
        XCTAssertEqual(fitter.replyReserve(forBudget: -1), 0)
    }

    func testShortenedDropsAtLeastOneExchangeAndStopsAtTheLastUserTurn() throws {
        var fitter = GGUFPromptFitter(windowSize: 4096)
        XCTAssertTrue(fitter.learn(from: LLMError.failedToDecode(reason: "context size exceeded[4096 < 4097]"),
                                   promptBytes: 1_000_000))
        let once = try XCTUnwrap(try fitter.shortened(Self.history, replyBudget: 0, render: Self.byteCount))
        XCTAssertEqual(once.history.count, Self.history.count - 2, "the estimate fits, but llama.cpp said it did not")
        let twice = try XCTUnwrap(try fitter.shortened(once.history, replyBudget: 0, render: Self.byteCount))
        XCTAssertEqual(twice.history, [.system("S"), .user("last")])
        XCTAssertNil(try fitter.shortened(twice.history, replyBudget: 0, render: Self.byteCount))
    }

    func testOtherErrorsTeachNothing() {
        var fitter = GGUFPromptFitter(windowSize: 1024)
        XCTAssertFalse(fitter.learn(from: LLMError.failedToDecode(reason: "batch decode failed"), promptBytes: 10))
        XCTAssertNil(fitter.tokensPerByte)
    }
}

/// Builds a minimal GGUF v3 metadata section.
enum GGUFFixture {
    enum Extra {
        case uint8(UInt8), float32(Float), bool(Bool), int64(Int64), float64(Double), strings([String]), int32s([Int32])
    }

    static let everyOtherType: [(String, Extra)] = [
        ("general.alignment", .uint8(32)), ("general.score", .float32(0.5)), ("general.flag", .bool(true)),
        ("general.size", .int64(-7)), ("general.ratio", .float64(1.5)), ("tokenizer.ggml.merges", .strings(["a b"])),
        ("llama.head_counts", .int32s([4, 8])),
    ]

    static func make(strings: [(String, String)], ids: [(String, UInt32)], tokens: [String],
                     extras: [(String, Extra)] = []) -> Data {
        var data = Data("GGUF".utf8)
        append(UInt32(3), to: &data)
        append(UInt64(0), to: &data)
        append(UInt64(strings.count + ids.count + extras.count + 1), to: &data)
        for (key, value) in extras {
            appendString(key, to: &data)
            append(value, to: &data)
        }
        for (key, value) in strings {
            appendString(key, to: &data)
            append(GGUFValueType.string.rawValue, to: &data)
            appendString(value, to: &data)
        }
        appendString("tokenizer.ggml.tokens", to: &data)
        append(GGUFValueType.array.rawValue, to: &data)
        append(GGUFValueType.string.rawValue, to: &data)
        append(UInt64(tokens.count), to: &data)
        tokens.forEach { appendString($0, to: &data) }
        for (key, value) in ids {
            appendString(key, to: &data)
            append(GGUFValueType.uint32.rawValue, to: &data)
            append(value, to: &data)
        }
        return data
    }

    private static func append(_ extra: Extra, to data: inout Data) {
        switch extra {
        case .uint8(let value): typed(.uint8, value, to: &data)
        case .float32(let value): typed(.float32, value.bitPattern, to: &data)
        case .bool(let value): typed(.bool, UInt8(value ? 1 : 0), to: &data)
        case .int64(let value): typed(.int64, value, to: &data)
        case .float64(let value): typed(.float64, value.bitPattern, to: &data)
        case .strings(let values):
            append(GGUFValueType.array.rawValue, to: &data)
            append(GGUFValueType.string.rawValue, to: &data)
            append(UInt64(values.count), to: &data)
            values.forEach { appendString($0, to: &data) }
        case .int32s(let values):
            append(GGUFValueType.array.rawValue, to: &data)
            append(GGUFValueType.int32.rawValue, to: &data)
            append(UInt64(values.count), to: &data)
            values.forEach { append($0, to: &data) }
        }
    }

    private static func typed<T: FixedWidthInteger>(_ type: GGUFValueType, _ value: T, to data: inout Data) {
        append(type.rawValue, to: &data)
        append(value, to: &data)
    }

    private static func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    private static func appendString(_ text: String, to data: inout Data) {
        append(UInt64(text.utf8.count), to: &data)
        data.append(contentsOf: Array(text.utf8))
    }
}
