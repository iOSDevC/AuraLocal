import SwiftUI
import TonamoxCore

// MARK: - Agent pane

/// Demos of the agent foundations: tool calls over several rounds, and replies that must match a JSON Schema.
struct AgentDemoListView: View {
    var body: some View {
        List {
            Section {
                NavigationLink {
                    ToolCallingDemoView()
                } label: {
                    Label("Tool calling", systemImage: "wrench.and.screwdriver")
                }
                NavigationLink {
                    StructuredOutputDemoView()
                } label: {
                    Label("Structured output (JSON)", systemImage: "curlybraces")
                }
            } footer: {
                Text("Both run on a downloaded model on this device. Tool calling needs a GGUF model or an MLX text model; the Simulator runs GGUF only.")
            }
        }
    }
}

// MARK: - Demo tools

/// Deterministic tools, so a run shows what the model did rather than what a service returned.
enum DemoTools {
    static let all: [any LLMTool] = [ClockTool(), CalculatorTool(), CityLookupTool(), WeatherTool(), InventoryTool()]

    static let prompts: [(title: String, text: String)] = [
        ("Time", "What time is it right now?"),
        ("Math", "Use the calculator: what is 37 multiplied by 41?"),
        ("Two tools", "What is the weather in Lima? Look up the city id first, then get the weather for that id."),
        ("Failing tool", "Check the inventory for oat milk."),
    ]

    fileprivate static let cityIDs: [String: Int] = ["lima": 7, "mexico city": 3, "madrid": 5, "tokyo": 11]
    fileprivate static let weatherByID: [Int: (celsius: Double, conditions: String)] = [
        7: (19.5, "overcast"), 3: (23, "sunny"), 5: (14, "rain"), 11: (17.5, "clear"),
    ]
}

struct ClockTool: LLMTool {
    let name = "get_time"
    let description = "Get the current local date and time on this device."
    struct Arguments: Decodable, ToolSchemaGeneratable {
        static var argumentsSchema: LLMToolArgumentsSchema { [:] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput {
        ToolOutput(data: ["time": Date.now.formatted(date: .abbreviated, time: .standard),
                          "timezone": TimeZone.current.identifier])
    }
}

struct CalculatorTool: LLMTool {
    let name = "calculator"
    let description = "Apply an arithmetic operation to two numbers."
    struct Arguments: Decodable, ToolSchemaGeneratable {
        let a: Double
        let b: Double
        let operation: String
        static var argumentsSchema: LLMToolArgumentsSchema {
            ["a": .number(description: "First number"),
             "b": .number(description: "Second number"),
             "operation": .enum(values: ["add", "subtract", "multiply", "divide"], description: "Operation")]
        }
    }
    func call(arguments: Arguments) async throws -> ToolOutput {
        let result: Double
        switch arguments.operation {
        case "add": result = arguments.a + arguments.b
        case "subtract": result = arguments.a - arguments.b
        case "multiply": result = arguments.a * arguments.b
        case "divide":
            guard arguments.b != 0 else { throw DemoToolError("Division by zero.") }
            result = arguments.a / arguments.b
        default: throw DemoToolError("Unknown operation \(arguments.operation).")
        }
        return ToolOutput(data: ["result": result])
    }
}

struct CityLookupTool: LLMTool {
    let name = "find_city"
    let description = "Find the numeric id of a city, needed by get_weather."
    struct Arguments: Decodable, ToolSchemaGeneratable {
        let city: String
        static var argumentsSchema: LLMToolArgumentsSchema { ["city": .string(description: "City name")] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput {
        guard let id = DemoTools.cityIDs[arguments.city.lowercased()] else {
            throw DemoToolError("No city named \(arguments.city). Known: Lima, Mexico City, Madrid, Tokyo.")
        }
        return ToolOutput(data: ["city": arguments.city, "city_id": id])
    }
}

struct WeatherTool: LLMTool {
    let name = "get_weather"
    let description = "Get the current weather for a city id returned by find_city."
    struct Arguments: Decodable, ToolSchemaGeneratable {
        let cityID: Int
        enum CodingKeys: String, CodingKey { case cityID = "city_id" }
        static var argumentsSchema: LLMToolArgumentsSchema { ["city_id": .integer(description: "Id from find_city")] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput {
        guard let reading = DemoTools.weatherByID[arguments.cityID] else {
            throw DemoToolError("Unknown city id \(arguments.cityID). Call find_city first.")
        }
        return ToolOutput(data: ["city_id": arguments.cityID, "temperature_c": reading.celsius,
                                 "conditions": reading.conditions])
    }
}

/// Always fails, to show a tool error going back to the model as text.
struct InventoryTool: LLMTool {
    let name = "check_inventory"
    let description = "Check how many units of a product are in stock."
    struct Arguments: Decodable, ToolSchemaGeneratable {
        let product: String
        static var argumentsSchema: LLMToolArgumentsSchema { ["product": .string(description: "Product name")] }
    }
    func call(arguments: Arguments) async throws -> ToolOutput {
        throw DemoToolError("The inventory service is offline.")
    }
}

struct DemoToolError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

// MARK: - Tool calling

struct ToolCallingDemoView: View {
    @State private var choices: [Model] = []
    @State private var selected: Model?
    @State private var maxRounds = 2
    @State private var reportErrors = true
    @State private var prompt = DemoTools.prompts[2].text
    @State private var reply = ""
    @State private var events: [ToolEvent] = []
    @State private var summary: ToolTurnSummary?
    @State private var errorText: String?
    @State private var running: Task<Void, Never>?
    @State private var llm: Tonamox?

    var body: some View {
        Group {
            if choices.isEmpty {
                ModelDownloadPlaceholder(
                    "A model for the tool demo",
                    message: "The demo needs a downloaded model that can call tools.",
                    models: [RequiredModel(.qwen3_1_7b, purpose: "Calls the demo tools")],
                    onReady: refreshChoices)
            } else {
                form
            }
        }
        .navigationTitle("Tool calling")
        .onAppear(perform: refreshChoices)
        .onDisappear(perform: stop)
    }

    private var form: some View {
        Form {
            Section("Model") {
                Picker("Model", selection: $selected) {
                    ForEach(choices, id: \.id) { Text($0.displayName).tag(Optional($0)) }
                }
                Stepper("Rounds per turn: \(maxRounds)", value: $maxRounds, in: 1...4)
                Toggle("Report tool errors to the model", isOn: $reportErrors)
            }
            Section("Prompt") {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack {
                        ForEach(DemoTools.prompts, id: \.title) { preset in
                            Button(preset.title) { prompt = preset.text }
                                .buttonStyle(.bordered)
                        }
                    }
                }
                TextField("Ask something that needs a tool", text: $prompt, axis: .vertical)
                if running == nil {
                    Button("Run", action: run).disabled(selected == nil || prompt.isEmpty)
                } else {
                    Button("Stop", role: .destructive, action: stop)
                }
            }
            if !events.isEmpty || summary != nil {
                Section("Tool calls") {
                    ForEach(Array(events.enumerated()), id: \.offset) { _, event in
                        ToolEventRow(event: event)
                    }
                    if let summary {
                        Text("\(summary.rounds) round(s), \(summary.results.count) result(s), \(summary.skipped.count) skipped")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            if !reply.isEmpty || errorText != nil {
                Section("Reply") {
                    let parts = ReasoningSplit(reply)
                    if !parts.answer.isEmpty { Text(parts.answer).textSelection(.enabled) }
                    if !parts.reasoning.isEmpty {
                        DisclosureGroup("Reasoning") {
                            Text(parts.reasoning).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                        }
                    }
                    if let errorText { Text(errorText).foregroundStyle(.red) }
                }
            }
        }
        .formStyle(.grouped)
    }

    private func refreshChoices() {
        choices = ChatModels.downloaded(requiring: [.text, .tools])
        if selected.map({ choices.contains($0) }) != true { selected = choices.first }
    }

    private func run() {
        guard let model = selected else { return }
        reply = ""
        events = []
        summary = nil
        errorText = nil
        let options = ToolLoopOptions(maxToolRounds: maxRounds, toolErrors: reportErrors ? .reportToModel : .failTurn)
        running = Task {
            defer { running = nil }
            do {
                let instance = try await loadedInstance(for: model)
                instance.toolLoop = options
                instance.onToolEvent = { events.append($0) }
                for try await piece in instance.stream(prompt) {
                    reply += piece
                }
                summary = instance.lastToolTurn
            } catch is CancellationError {
                errorText = "Stopped."
            } catch {
                errorText = error.localizedDescription
            }
        }
    }

    /// Tool instances are not cached by ModelManager, so the demo keeps one per picked model.
    private func loadedInstance(for model: Model) async throws -> Tonamox {
        if let llm, llm.model == model { return llm }
        if let llm { ModelManager.shared.release(llm) }
        let loaded = try await ModelManager.shared.load(model, tools: DemoTools.all, temperature: 0.2)
        llm = loaded
        return loaded
    }

    private func stop() {
        running?.cancel()
        running = nil
    }
}

private struct ToolEventRow: View {
    let event: ToolEvent

    var body: some View {
        switch event {
        case .started(let call):
            Label("Round \(call.round): \(call.name)(\(call.arguments))", systemImage: "arrow.right.circle")
                .font(.callout.monospaced())
        case .finished(let result):
            Label(result.output, systemImage: result.isError ? "exclamationmark.triangle" : "arrow.left.circle")
                .font(.callout.monospaced())
                .foregroundStyle(result.isError ? Color.red : Color.primary)
                .accessibilityLabel("\(result.call.name) \(result.isError ? "failed" : "returned"): \(result.output)")
        case .roundLimitReached(let skipped):
            Label("Round limit reached; skipped \(skipped.map(\.name).joined(separator: ", "))",
                  systemImage: "hand.raised")
                .font(.callout)
                .foregroundStyle(.orange)
        }
    }
}

/// Splits Qwen3-style `<think>` blocks from the answer; an unclosed block while streaming counts as reasoning.
struct ReasoningSplit {
    let reasoning: String
    let answer: String

    init(_ text: String) {
        var reasoning = ""
        var answer = ""
        var rest = text[...]
        while let open = rest.range(of: "<think>") {
            answer.append(contentsOf: rest[..<open.lowerBound])
            let after = rest[open.upperBound...]
            if let close = after.range(of: "</think>") {
                reasoning.append(contentsOf: after[..<close.lowerBound])
                rest = after[close.upperBound...]
            } else {
                reasoning.append(contentsOf: after)
                rest = rest[rest.endIndex...]
            }
        }
        answer.append(contentsOf: rest)
        self.reasoning = reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
        self.answer = answer.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Structured output

struct StructuredOutputDemoView: View {
    private static let receiptSchema = """
        {
          "type": "object",
          "properties": {
            "merchant": { "type": "string" },
            "total": { "type": "number", "minimum": 0 },
            "currency": { "type": "string", "enum": ["USD", "MXN", "EUR"] },
            "date": { "type": "string" }
          },
          "required": ["merchant", "total", "currency"],
          "additionalProperties": false
        }
        """

    @State private var choices: [Model] = []
    @State private var selected: Model?
    @State private var schema = StructuredOutputDemoView.receiptSchema
    @State private var input = "Café Aura, 2 lattes and a croissant. Total $12.75 USD, paid on 20 Oct 2026."
    @State private var useGrammar = true
    @State private var output = ""
    @State private var detail = ""
    @State private var errorText: String?
    @State private var running: Task<Void, Never>?

    var body: some View {
        Group {
            if choices.isEmpty {
                ModelDownloadPlaceholder(
                    "A model for the JSON demo",
                    message: "The demo needs a downloaded text model.",
                    models: [RequiredModel(.qwen3_1_7b, purpose: "Writes the JSON reply")],
                    onReady: refreshChoices)
            } else {
                form
            }
        }
        .navigationTitle("Structured output")
        .onAppear(perform: refreshChoices)
        .onDisappear { running?.cancel() }
    }

    private var form: some View {
        Form {
            Section {
                Picker("Model", selection: $selected) {
                    ForEach(choices, id: \.id) { Text($0.displayName).tag(Optional($0)) }
                }
                Toggle("Constrain GGUF output with the schema grammar", isOn: $useGrammar)
            } footer: {
                Text(enforcementNote)
            }
            Section("JSON Schema") {
                TextEditor(text: $schema)
                    .font(.caption.monospaced())
                    .frame(minHeight: 160)
            }
            Section("Text to extract from") {
                TextField("Text", text: $input, axis: .vertical)
                if running == nil {
                    Button("Extract", action: run).disabled(selected == nil || input.isEmpty)
                } else {
                    Button("Stop", role: .destructive) { running?.cancel() }
                }
            }
            if !output.isEmpty || errorText != nil {
                Section("Result") {
                    if !output.isEmpty {
                        Text(output).font(.callout.monospaced()).textSelection(.enabled)
                    }
                    if !detail.isEmpty {
                        Text(detail).font(.caption).foregroundStyle(.secondary)
                    }
                    if let errorText { Text(errorText).foregroundStyle(.red) }
                }
            }
        }
        .formStyle(.grouped)
    }

    private var enforcementNote: String {
        guard let selected else { return "" }
        if selected.format == .gguf && useGrammar {
            return "GGUF: decoding follows a grammar built from the schema, and the reply is validated."
        }
        return "The reply is validated against the schema and re-prompted with the violations when it breaks it."
    }

    private func refreshChoices() {
        choices = ChatModels.downloaded(requiring: [.text])
        if selected.map({ choices.contains($0) }) != true { selected = choices.first }
    }

    private func run() {
        guard let model = selected else { return }
        output = ""
        detail = ""
        errorText = nil
        let profile = TonamoxProfile(id: "demo-json", displayName: "JSON demo",
                                  instructions: "Extract the receipt fields from the user's text.",
                                  model: model, sampling: .precise, outputSchema: .json(schema))
        let grammar = useGrammar
        running = Task {
            defer { running = nil }
            let start = ContinuousClock.now
            do {
                let session = try await TonamoxSession(profile: profile)
                session.usesSchemaGrammar = grammar
                for try await json in session.stream(input) {
                    output = Self.pretty(json)
                }
                detail = "Done in \((ContinuousClock.now - start).formatted(.units(allowed: [.seconds, .milliseconds])))."
            } catch is CancellationError {
                errorText = "Stopped."
            } catch {
                errorText = error.localizedDescription
            }
        }
    }

    private static func pretty(_ json: String) -> String {
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data),
              let formatted = try? JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]),
              let text = String(data: formatted, encoding: .utf8) else { return json }
        return text
    }
}
