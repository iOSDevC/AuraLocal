import Foundation
import NaturalLanguage
import AuraCore

/// `aura ml` subcommands backed by the system Vision, NaturalLanguage and SoundAnalysis tools.
extension AuraCLI {

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
            let payload = code.payload.map { "\"\($0)\"" } ?? "(binary payload)"
            print("\(shortSymbology(code.symbology))  \(payload)  \(box(code.boundingBox))")
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
        err("— -1 negative … 1 positive; neutral text scored from -0.8 to 0.4 depending on the language\n")
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
        let maxResults = try parsed.int("--max") ?? 5
        let sounds = try await tool.classify(
            audioFileAt: URL(fileURLWithPath: path),
            maxResults: maxResults,
            aggregation: parsed.has("--mean") ? .mean : .peak)
        if sounds.isEmpty && maxResults > 0 { err("(nothing analysed: no audio could be decoded from the file)\n") }
        for sound in sounds {
            print("\(fixed(sound.confidence))  \(sound.identifier)")
        }
        err("— \(parsed.has("--mean") ? "mean over" : "peak across") 3 s analysis windows "
            + "(one window as long as the file when it is shorter)\n")
    }

    // MARK: - Formatting

    private static let symbologyPrefix = "VNBarcodeSymbology"

    private static func shortSymbology(_ raw: String) -> String {
        raw.hasPrefix(symbologyPrefix) ? String(raw.dropFirst(symbologyPrefix.count)) : raw
    }

    private static func rawSymbology(_ name: String) -> String {
        name.hasPrefix(symbologyPrefix) ? name : symbologyPrefix + name
    }

    private static func degrees(_ radians: Double?) -> String {
        radians.map { "\(fixed($0 * 180 / .pi, digits: 0))°" } ?? "n/a"
    }
}
