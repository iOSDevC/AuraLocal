import Foundation
import NaturalLanguage
import AuraCore

/// `aura ml <subcommand>` — one subcommand per on-device ML tool. Results go to stdout,
/// notes and summaries to stderr, like the rest of the CLI.
extension AuraCLI {

    static func runML(_ args: [String]) async throws {
        guard let subcommand = args.first else { printMLUsage(); exit(2) }
        let rest = Array(args.dropFirst())
        switch subcommand {
        case "classify-image":  try await runClassifyImage(rest)
        case "barcodes":        try await runBarcodes(rest)
        case "faces":           try await runFaces(rest)
        case "ocr-lines":       try await runOCRLines(rest)
        case "language":        try await runLanguage(rest)
        case "entities":        try await runEntities(rest)
        case "sentiment":       try await runSentiment(rest)
        case "similarity":      try await runSimilarity(rest)
        case "sounds":          try await runSounds(rest)
        case "coreml-describe": try await runCoreMLDescribe(rest)
        case "coreml-predict":  try await runCoreMLPredict(rest)
        case "train-text":      try await runTrainText(rest)
        case "classify-text":   try await runClassifyText(rest)
        case "help", "-h", "--help": printMLUsage()
        default:
            err("Unknown ml subcommand: \(subcommand)\n")
            printMLUsage(); exit(2)
        }
    }

    // MARK: - Vision

    static func runClassifyImage(_ args: [String]) async throws {
        let parsed = try CLIArguments(args, options: ["--max", "--min"])
        let data = try imageData(parsed, usage: "aura ml classify-image <image> [--max N] [--min 0.1]")
        let maxResults = try parsed.int("--max") ?? 5
        let minimum = try parsed.double("--min") ?? 0.1
        let tool = VisionImageClassificationTool()
        try await requireAvailable(tool)
        let labels = try tool.classify(inImageData: data, maxResults: maxResults, minimumConfidence: Float(minimum))
        if labels.isEmpty { err("(no label above \(fixed(minimum)))\n") }
        for label in labels {
            print("\(fixed(label.confidence))  \(label.identifier)")
        }
    }

    static func runBarcodes(_ args: [String]) async throws {
        let parsed = try CLIArguments(args, options: ["--symbology"], flags: ["--list"])
        let tool = VisionBarcodeTool()
        if parsed.has("--list") {
            let names = tool.supportedSymbologies().map(shortSymbology).sorted()
            names.forEach { print($0) }
            err("— \(names.count) symbologies\n")
            return
        }
        let data = try imageData(parsed, usage: "aura ml barcodes <image> [--symbology QR]... | --list")
        try await requireAvailable(tool)
        let filter = parsed.values("--symbology").map(rawSymbology)
        let codes = try tool.detectBarcodes(inImageData: data, symbologies: filter)
        if codes.isEmpty { err("(no barcode found)\n") }
        for code in codes {
            print("\(shortSymbology(code.symbology))  \(code.payload.map { "\"\($0)\"" } ?? "(binary payload)")  \(box(code.boundingBox))")
        }
    }

    static func runFaces(_ args: [String]) async throws {
        let parsed = try CLIArguments(args)
        let data = try imageData(parsed, usage: "aura ml faces <image>")
        let tool = VisionFaceDetectionTool()
        try await requireAvailable(tool)
        let faces = try tool.detectFaces(inImageData: data)
        if faces.isEmpty { err("(no face found)\n") }
        for (index, face) in faces.enumerated() {
            print("face \(index + 1)  confidence \(fixed(face.confidence))  \(box(face.boundingBox))  "
                  + "roll \(degrees(face.roll))  yaw \(degrees(face.yaw))  pitch \(degrees(face.pitch))")
        }
    }

    static func runOCRLines(_ args: [String]) async throws {
        let parsed = try CLIArguments(args, options: ["--lang"], flags: ["--fast"])
        let data = try imageData(parsed, usage: "aura ml ocr-lines <image> [--lang es-ES]... [--fast]")
        let lines = try VisionOCRTool().recognizeLines(
            inImageData: data, languages: parsed.values("--lang"), accurate: !parsed.has("--fast"))
        if lines.isEmpty { err("(no text found)\n") }
        for line in lines {
            print("\(fixed(line.confidence))  \(box(line.boundingBox))  \(line.text)")
        }
    }

    // MARK: - Language

    static func runLanguage(_ args: [String]) async throws {
        let parsed = try CLIArguments(args, options: ["--max", "--only"])
        let text = try inputText(parsed, usage: "aura ml language \"<text>\" [--max N] [--only es,en]")
        let constraints = parsed.value("--only")?.split(separator: ",").map(String.init) ?? []
        let result = NLLanguageIdentificationTool()
            .identify(text, maxHypotheses: try parsed.int("--max") ?? 3, constraints: constraints)
        print("dominant  \(result.dominantLanguage ?? "undetermined")")
        for hypothesis in result.hypotheses {
            print("\(fixed(hypothesis.probability))  \(hypothesis.language)")
        }
    }

    static func runEntities(_ args: [String]) async throws {
        let parsed = try CLIArguments(args, options: ["--lang"])
        let text = try inputText(parsed, usage: "aura ml entities \"<text>\" [--lang es]")
        let tool = NLEntityRecognitionTool()
        try await requireAvailable(tool)
        let entities = tool.entities(in: text, language: parsed.value("--lang"))
        if entities.isEmpty { err("(no people, places or organizations found)\n") }
        for entity in entities {
            let kind = entity.kind.rawValue.padding(toLength: 12, withPad: " ", startingAt: 0)
            print("\(kind)  \(entity.text)  [utf16 \(entity.range.location)+\(entity.range.length)]")
        }
    }

    static func runSentiment(_ args: [String]) async throws {
        let parsed = try CLIArguments(args, options: ["--lang"])
        let text = try inputText(parsed, usage: "aura ml sentiment \"<text>\" [--lang es]")
        let tool = NLSentimentTool()
        try await requireAvailable(tool)
        guard let score = tool.score(text, language: parsed.value("--lang")) else {
            err("(no sentiment model for this text's language)\n")
            return
        }
        print(fixed(score))
        err("— -1 negative … 1 positive; neutral text can score as low as -0.8, so only values near -1 are clearly negative\n")
    }

    static func runSimilarity(_ args: [String]) async throws {
        let parsed = try CLIArguments(args, options: ["--lang"])
        guard parsed.positionals.count == 2 else {
            throw CLIUsageError("aura ml similarity \"<text a>\" \"<text b>\" [--lang en]")
        }
        let language = NLLanguage(rawValue: parsed.value("--lang") ?? NLLanguage.english.rawValue)
        guard let distance = NLEmbeddingTool()
            .distance(parsed.positionals[0], parsed.positionals[1], language: language) else {
            err("(no sentence embedding for \(language.rawValue) on this device)\n")
            return
        }
        print(fixed(distance))
        err("— cosine distance with \(language.rawValue) sentence embeddings: 0 = same meaning, 2 = opposite\n")
    }

    // MARK: - Audio

    static func runSounds(_ args: [String]) async throws {
        let parsed = try CLIArguments(args, options: ["--max"], flags: ["--mean", "--list"])
        let tool = SoundClassificationTool()
        try await requireAvailable(tool)
        if parsed.has("--list") {
            let labels = try tool.knownSounds()
            labels.forEach { print($0) }
            err("— \(labels.count) sounds\n")
            return
        }
        guard let path = parsed.positionals.first else {
            throw CLIUsageError("aura ml sounds <audio file> [--max N] [--mean] | --list")
        }
        let sounds = try await tool.classify(
            audioFileAt: URL(fileURLWithPath: path),
            maxResults: try parsed.int("--max") ?? 5,
            aggregation: parsed.has("--mean") ? .mean : .peak)
        if sounds.isEmpty { err("(nothing analysed: the file is empty or has no audio track)\n") }
        for sound in sounds {
            print("\(fixed(sound.confidence))  \(sound.identifier)")
        }
        err("— \(parsed.has("--mean") ? "mean over" : "peak across") ~3 s analysis windows\n")
    }

    // MARK: - Custom Core ML models

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
        let inputs = try Dictionary(uniqueKeysWithValues: parsed.positionals.dropFirst().map { pair in
            try featureInput(pair, specs: specs)
        })
        let outputs = try await tool.predict(inputs)
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
        print("label  \(result.label ?? "(none)")")
        for hypothesis in result.ranked() {
            print("\(fixed(hypothesis.probability))  \(hypothesis.label)")
        }
    }

    // MARK: - Usage

    static func printMLUsage() {
        print("""
        aura ml — on-device machine learning tools (no model download, no network)

        VISION (images: PNG, JPEG, HEIC…; EXIF orientation honoured)
          aura ml classify-image <image> [--max N] [--min 0.1]   Label the scene/objects
          aura ml barcodes <image> [--symbology QR]... | --list  Read QR codes and barcodes
          aura ml faces <image>                                  Detect faces and head pose
          aura ml ocr-lines <image> [--lang es-ES]... [--fast]   Text line by line with boxes

        LANGUAGE (text argument, or - to read stdin)
          aura ml language "<text>" [--max N] [--only es,en]     Identify the language
          aura ml entities "<text>" [--lang es]                  People, places, organizations
          aura ml sentiment "<text>" [--lang es]                 Score from -1 to 1
          aura ml similarity "<a>" "<b>" [--lang en]             Sentence-embedding distance

        AUDIO
          aura ml sounds <audio file> [--max N] [--mean] | --list  Classify everyday sounds

        CUSTOM MODELS (Core ML / Create ML)
          aura ml coreml-describe <model>                        Inputs, outputs, labels, metadata
          aura ml coreml-predict <model> <input>=<value>...      Run one prediction
          aura ml train-text <csv> --out <Model.mlmodel> [...]   Train a text classifier (Create ML)
          aura ml classify-text <Model.mlmodel> "<text>"         Classify text with a trained model

        Boxes are normalized 0…1 with the origin at the bottom-left (Vision's convention).
        """)
    }

    // MARK: - Helpers

    private static func requireAvailable(_ tool: some SystemTool) async throws {
        if let reason = await tool.availability().reason {
            throw CLIUsageError.unavailable(tool.displayName, reason: reason)
        }
    }

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

    private static func imageData(_ parsed: CLIArguments, usage: String) throws -> Data {
        guard let path = parsed.positionals.first else { throw CLIUsageError(usage) }
        return try Data(contentsOf: URL(fileURLWithPath: path))
    }

    private static func inputText(_ parsed: CLIArguments, usage: String) throws -> String {
        guard !parsed.positionals.isEmpty else { throw CLIUsageError(usage) }
        if parsed.positionals == ["-"] {
            return String(decoding: FileHandle.standardInput.readDataToEndOfFile(), as: UTF8.self)
        }
        return parsed.positionals.joined(separator: " ")
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
    ) throws -> (String, CoreMLModelTool.FeatureValue) {
        guard let separator = pair.firstIndex(of: "=") else {
            throw CLIUsageError("Inputs are <name>=<value>, got \(pair).")
        }
        let name = String(pair[..<separator])
        let raw = String(pair[pair.index(after: separator)...])
        guard let spec = specs[name] else {
            throw CLIUsageError("The model has no input named \(name); inputs: \(specs.keys.sorted().joined(separator: ", ")).")
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
            let numbers = raw.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
            return (name, .doubles(numbers))
        default:
            throw CLIUsageError("\(name) is a \(spec.kind.rawValue) input; the CLI passes only numbers, text and multi-arrays.")
        }
    }

    private static func describeValue(_ value: CoreMLModelTool.FeatureValue?) -> String {
        switch value {
        case .string(let text)?: "\"\(text)\""
        case .int(let number)?: "\(number)"
        case .double(let number)?: fixed(number, digits: 4)
        case .doubles(let values)?: values.map { fixed($0, digits: 4) }.joined(separator: ", ")
        case .multiArray(let shape, let values)?:
            "shape \(shape) [\(values.prefix(8).map { fixed($0, digits: 4) }.joined(separator: ", "))\(values.count > 8 ? ", …" : "")]"
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

    private static let symbologyPrefix = "VNBarcodeSymbology"

    private static func shortSymbology(_ raw: String) -> String {
        raw.hasPrefix(symbologyPrefix) ? String(raw.dropFirst(symbologyPrefix.count)) : raw
    }

    private static func rawSymbology(_ name: String) -> String {
        name.hasPrefix(symbologyPrefix) ? name : symbologyPrefix + name
    }

    private static func box(_ rect: CGRect) -> String {
        "box x \(fixed(rect.minX)) y \(fixed(rect.minY)) w \(fixed(rect.width)) h \(fixed(rect.height))"
    }

    private static func degrees(_ radians: Double?) -> String {
        radians.map { "\(fixed($0 * 180 / .pi, digits: 0))°" } ?? "n/a"
    }

    private static func percent(_ fraction: Double?) -> String {
        fraction.map { "\(fixed($0 * 100, digits: 1)) %" } ?? "n/a"
    }

    static func fixed(_ value: Double, digits: Int = 2) -> String {
        String(format: "%.\(digits)f", value)
    }

    static func fixed(_ value: Float) -> String {
        fixed(Double(value))
    }
}

/// Positional arguments plus `--option value` (repeatable) and boolean `--flag`s.
struct CLIArguments {
    private(set) var positionals: [String] = []
    private var optionValues: [String: [String]] = [:]
    private var presentFlags: Set<String> = []

    init(_ args: [String], options: Set<String> = [], flags: Set<String> = []) throws {
        var unknown: [String] = []
        var index = 0
        while index < args.count {
            let arg = args[index]
            if options.contains(arg), index + 1 < args.count {
                optionValues[arg, default: []].append(args[index + 1])
                index += 1
            } else if flags.contains(arg) {
                presentFlags.insert(arg)
            } else if arg.hasPrefix("--") {
                unknown.append(arg)
            } else {
                positionals.append(arg)
            }
            index += 1
        }
        if let first = unknown.first {
            throw CLIUsageError("Unknown or incomplete option \(first).")
        }
    }

    func has(_ flag: String) -> Bool { presentFlags.contains(flag) }
    func value(_ option: String) -> String? { optionValues[option]?.last }
    func values(_ option: String) -> [String] { optionValues[option] ?? [] }

    func int(_ option: String) throws -> Int? {
        guard let raw = value(option) else { return nil }
        guard let number = Int(raw) else { throw CLIUsageError("\(option) expects an integer, got \(raw).") }
        return number
    }

    func double(_ option: String) throws -> Double? {
        guard let raw = value(option) else { return nil }
        guard let number = Double(raw) else { throw CLIUsageError("\(option) expects a number, got \(raw).") }
        return number
    }
}

struct CLIUsageError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }

    static func unavailable(_ tool: String, reason: String) -> CLIUsageError {
        CLIUsageError("\(tool) is unavailable here: \(reason)")
    }

    var errorDescription: String? { message }
}
