import Foundation
import NaturalLanguage
import AuraCore

/// `aura ml` subcommands for custom Core ML models and Create ML training.
extension AuraCLI {

    static func runCoreMLDescribe(_ args: [String]) async throws {
        let parsed = try CLIArguments(args)
        guard let path = parsed.positionals.first else {
            throw CLIUsageError("aura ml coreml-describe <model.mlmodel|.mlpackage|.mlmodelc>")
        }
        let tool = CoreMLModelTool(modelAt: URL(fileURLWithPath: path))
        let info = try await tool.describe()
        print("id           \(tool.id)")
        printIfPresent("author", info.author)
        printIfPresent("description", info.shortDescription)
        printIfPresent("version", info.version)
        printIfPresent("license", info.license)
        print("inputs")
        info.inputs.forEach { print("  " + featureLine($0)) }
        print("outputs")
        info.outputs.forEach { print("  " + featureLine($0)) }
        if info.isClassifier {
            print("classifier   \(info.classLabels.count) labels: \(info.classLabels.joined(separator: ", "))")
        }
    }

    static func runCoreMLPredict(_ args: [String]) async throws {
        let parsed = try CLIArguments(args)
        guard let path = parsed.positionals.first, parsed.positionals.count > 1 else {
            throw CLIUsageError("aura ml coreml-predict <model> <input>=<value>... "
                                + "(numbers, text, or comma-separated numbers for a multi-array)")
        }
        let tool = CoreMLModelTool(modelAt: URL(fileURLWithPath: path))
        let specs = Dictionary(uniqueKeysWithValues: try await tool.describe().inputs.map { ($0.name, $0) })
        let pairs = try parsed.positionals.dropFirst().map { try featureInput($0, specs: specs) }
        var seen: Set<String> = []
        if let repeated = pairs.map(\.name).first(where: { !seen.insert($0).inserted }) {
            throw CLIUsageError("Input \(repeated) given more than once.")
        }
        let outputs = try await tool.predict(Dictionary(uniqueKeysWithValues: pairs.map { ($0.name, $0.value) }))
        for name in outputs.keys.sorted() {
            print("\(name)  \(describeValue(outputs[name]))")
        }
    }

    static func runTrainText(_ args: [String]) async throws {
        let usage = "aura ml train-text <examples.csv> --out <Model.mlmodel> [--text-column text] "
            + "[--label-column label] [--algorithm maxent|crf|static|bert] [--language es] "
            + "[--holdout 0.2 [--seed 7] | --no-validation]"
        let parsed = try CLIArguments(
            args,
            options: ["--out", "--text-column", "--label-column", "--algorithm", "--language", "--holdout", "--seed"],
            flags: ["--no-validation"])
        guard let csv = parsed.positionals.first, let out = parsed.value("--out") else {
            throw CLIUsageError(usage)
        }
        let trainer = TextClassifierTrainer()
        try await requireAvailable(trainer)
        let holdOut: TextClassifierTrainer.Validation
        if parsed.has("--no-validation") {
            holdOut = .disabled
        } else if let fraction = try parsed.double("--holdout") {
            holdOut = .holdOut(fraction: fraction, seed: try parsed.int("--seed") ?? 7)
        } else {
            holdOut = .automatic
        }
        let started = Date()
        let report = try await withStdoutOnStderr {
            try await trainer.train(
                csvAt: URL(fileURLWithPath: csv),
                textColumn: parsed.value("--text-column") ?? "text",
                labelColumn: parsed.value("--label-column") ?? "label",
                writingModelTo: URL(fileURLWithPath: out),
                algorithm: try algorithm(parsed.value("--algorithm")),
                language: parsed.value("--language").map(NLLanguage.init(rawValue:)),
                validation: holdOut)
        }
        print(report.modelURL.path)
        err("— \(report.exampleCount) examples, \(report.classLabels.count) labels "
            + "(\(report.classLabels.joined(separator: ", "))) in \(fixed(Date().timeIntervalSince(started))) s\n")
        err("  training accuracy   \(percent(report.trainingAccuracy))\n")
        err("  validation accuracy \(percent(report.validationAccuracy))\n")
    }

    static func runClassifyText(_ args: [String]) async throws {
        let parsed = try CLIArguments(args, options: ["--max"])
        guard parsed.positionals.count >= 2 else {
            throw CLIUsageError("aura ml classify-text <Model.mlmodel> \"<text>\" [--max N]")
        }
        let tool = TextClassifierTool(modelAt: URL(fileURLWithPath: parsed.positionals[0]))
        let text = parsed.positionals.dropFirst().joined(separator: " ")
        let result = try await tool.classify(text, maxHypotheses: try parsed.int("--max") ?? 3)
        print("label  \(result.label ?? "(none: blank text)")")
        for hypothesis in result.ranked() {
            print("\(fixed(hypothesis.probability))  \(hypothesis.label)")
        }
    }

    // MARK: - Helpers

    /// Create ML prints its training log to stdout; route it to stderr so stdout carries only the result.
    private static func withStdoutOnStderr<T>(_ body: () async throws -> T) async rethrows -> T {
        fflush(stdout)
        let saved = dup(STDOUT_FILENO)
        dup2(STDERR_FILENO, STDOUT_FILENO)
        defer {
            fflush(stdout)
            dup2(saved, STDOUT_FILENO)
            close(saved)
        }
        return try await body()
    }

    private static func algorithm(_ name: String?) throws -> TextClassifierTrainer.Algorithm {
        switch name?.lowercased() {
        case nil, "maxent": .maxEnt
        case "crf": .crf
        case "static": .transferLearning(.staticEmbedding)
        case "bert": .transferLearning(.bertEmbedding)
        case let other?: throw CLIUsageError("Unknown algorithm \(other): use maxent, crf, static or bert.")
        }
    }

    private static func featureInput(
        _ pair: String, specs: [String: CoreMLModelTool.FeatureSpec]
    ) throws -> (name: String, value: CoreMLModelTool.FeatureValue) {
        guard let separator = pair.firstIndex(of: "=") else {
            throw CLIUsageError("Inputs are <name>=<value>, got \(pair).")
        }
        let name = String(pair[..<separator])
        let raw = String(pair[pair.index(after: separator)...])
        guard let spec = specs[name] else {
            let known = specs.keys.sorted().joined(separator: ", ")
            throw CLIUsageError("The model has no input named \(name); inputs: \(known).")
        }
        switch spec.kind {
        case .string:
            return (name, .string(raw))
        case .int64:
            guard let number = Int(raw) else { throw CLIUsageError("\(name) expects an integer, got \(raw).") }
            return (name, .int(number))
        case .double:
            guard let number = Double(raw) else { throw CLIUsageError("\(name) expects a number, got \(raw).") }
            return (name, .double(number))
        case .multiArray:
            let numbers = try raw.split(separator: ",").map { token in
                let trimmed = token.trimmingCharacters(in: .whitespaces)
                guard let number = Double(trimmed) else {
                    throw CLIUsageError("\(name) expects comma-separated numbers, got \(trimmed).")
                }
                return number
            }
            return (name, .doubles(numbers))
        default:
            throw CLIUsageError(
                "\(name) is a \(spec.kind.rawValue) input; the CLI passes only numbers, text and multi-arrays.")
        }
    }

    private static func describeValue(_ value: CoreMLModelTool.FeatureValue?) -> String {
        switch value {
        case .string(let text)?: "\"\(text)\""
        case .int(let number)?: "\(number)"
        case .double(let number)?: fixed(number, digits: 4)
        case .doubles(let values)?: values.map { fixed($0, digits: 4) }.joined(separator: ", ")
        case .multiArray(let shape, let values)?:
            "shape \(shape) [\(values.prefix(8).map { fixed($0, digits: 4) }.joined(separator: ", "))"
                + "\(values.count > 8 ? ", …" : "")]"
        case .dictionary(let scores)?:
            scores.sorted { $0.value > $1.value }.prefix(5)
                .map { "\($0.key)=\(fixed($0.value))" }.joined(separator: "  ")
        case .strings(let items)?: items.joined(separator: ", ")
        case .image?: "(image)"
        case .unsupported(let kind)?: "(\(kind))"
        case nil: "(none)"
        }
    }

    private static func featureLine(_ spec: CoreMLModelTool.FeatureSpec) -> String {
        var line = "\(spec.name)  \(spec.kind.rawValue)"
        if !spec.shape.isEmpty { line += " \(spec.shape)" }
        if let dataType = spec.dataType { line += " \(dataType)" }
        if spec.isShapeFlexible { line += " (flexible)" }
        if spec.isOptional { line += " (optional)" }
        return line
    }

    private static func printIfPresent(_ label: String, _ value: String?) {
        guard let value, !value.isEmpty else { return }
        print(label.padding(toLength: 12, withPad: " ", startingAt: 0) + " " + value)
    }
}
