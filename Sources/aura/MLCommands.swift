import Foundation
import AuraCore

/// `aura ml <subcommand>` — one subcommand per on-device ML tool. Results go to stdout,
/// notes and summaries to stderr, like the rest of the CLI.
extension AuraCLI {

    private static let mlSubcommands: [String: @MainActor ([String]) async throws -> Void] = [
        "classify-image": runClassifyImage,
        "barcodes": runBarcodes,
        "faces": runFaces,
        "ocr-lines": runOCRLines,
        "language": runLanguage,
        "entities": runEntities,
        "sentiment": runSentiment,
        "similarity": runSimilarity,
        "sounds": runSounds,
        "coreml-describe": runCoreMLDescribe,
        "coreml-predict": runCoreMLPredict,
        "train-text": runTrainText,
        "classify-text": runClassifyText
    ]

    static func runML(_ args: [String]) async throws {
        guard let subcommand = args.first else { printMLUsage(); exit(2) }
        if ["help", "-h", "--help"].contains(subcommand) {
            printMLUsage()
            return
        }
        guard let run = mlSubcommands[subcommand] else {
            err("Unknown ml subcommand: \(subcommand)\n")
            printMLUsage(); exit(2)
        }
        try await run(Array(args.dropFirst()))
    }

    static func printMLUsage() {
        print("""
        aura ml — on-device machine learning tools (nothing is uploaded)

        VISION (images: PNG, JPEG, HEIC…; EXIF orientation honoured)
          aura ml classify-image <image> [--max N] [--min 0.1]   Label the scene/objects
          aura ml barcodes <image> [--symbology QR]... | --list  Read QR codes and barcodes
          aura ml faces <image>                                  Detect faces and head pose
          aura ml ocr-lines <image> [--lang es-ES]... [--fast]   Text line by line with boxes

        LANGUAGE (language, entities and sentiment read stdin when the text is -)
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

    static func requireAvailable(_ tool: some SystemTool) async throws {
        if let reason = await tool.availability().reason {
            throw CLIUsageError.unavailable(tool.displayName, reason: reason)
        }
    }

    static func imageData(_ parsed: CLIArguments, usage: String) throws -> Data {
        guard let path = parsed.positionals.first else { throw CLIUsageError(usage) }
        return try Data(contentsOf: URL(fileURLWithPath: path))
    }

    static func inputText(_ parsed: CLIArguments, usage: String) throws -> String {
        guard !parsed.positionals.isEmpty else { throw CLIUsageError(usage) }
        guard parsed.positionals == ["-"] else {
            return parsed.positionals.joined(separator: " ")
        }
        guard let text = String(data: FileHandle.standardInput.readDataToEndOfFile(), encoding: .utf8) else {
            throw CLIUsageError("stdin is not UTF-8 text.")
        }
        return text
    }

    static func box(_ rect: CGRect) -> String {
        "box x \(fixed(rect.minX)) y \(fixed(rect.minY)) w \(fixed(rect.width)) h \(fixed(rect.height))"
    }

    static func percent(_ fraction: Double?) -> String {
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
