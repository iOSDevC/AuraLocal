import Foundation
import AuraCore
import ReceiptSpike

/// `aura receipt` — the Recibo spike harness. The rule-based baseline always runs; with `--model` a local model
/// at temperature 0 fills the same fields from the same OCR text, so both are scored side by side.
extension AuraCLI {

    static let receiptUsage = """
        usage: aura receipt extract <image>... [--model <id | .gguf path>] [--lang es-MX]...
               aura receipt eval <labels.csv> [--images <dir>] [--model <id | .gguf path>] [--lang es-MX]...
                                 [--min-total 0.9] [--min-date 0.9] [--min-merchant 0.8] [--min-tax 0.8]
                                 [--report <out.json>]

        Labels CSV: a header naming file, merchant, date (YYYY-MM-DD), total and optionally tax and currency;
        an empty cell means the receipt does not show the field. --images defaults to the CSV's folder.
        eval exits 1 when the model (or, without --model, the baseline) misses a --min-* threshold.
        """

    static func runReceipt(_ args: [String]) async throws {
        guard let subcommand = args.first else { throw CLIUsageError(receiptUsage) }
        let rest = Array(args.dropFirst())
        switch subcommand {
        case "extract": try await runReceiptExtract(rest)
        case "eval": try await runReceiptEval(rest)
        case "help", "-h", "--help": print(receiptUsage)
        default: throw CLIUsageError(receiptUsage)
        }
    }

    // MARK: - extract

    static func runReceiptExtract(_ args: [String]) async throws {
        let parsed = try CLIArguments(args, options: ["--model", "--lang"])
        guard !parsed.positionals.isEmpty else { throw CLIUsageError(receiptUsage) }
        let reader = try await ReceiptReader(modelArgument: parsed.value("--model"), languages: parsed.values("--lang"))
        for path in parsed.positionals {
            let result = try await reader.read(URL(fileURLWithPath: path))
            if parsed.positionals.count > 1 { print("== \(path)") }
            print("baseline  \(describe(result.baseline))")
            if let model = result.model {
                print("model     \(describe(model.fields))  (\(seconds(model.seconds)))")
                print("anchored  \(describe(model.anchors))")
            }
        }
    }

    // MARK: - eval

    static func runReceiptEval(_ args: [String]) async throws {
        let options: Set<String> = ["--images", "--model", "--lang", "--report",
                                    "--min-total", "--min-date", "--min-merchant", "--min-tax"]
        let parsed = try CLIArguments(args, options: options)
        guard let csvPath = parsed.positionals.first else { throw CLIUsageError(receiptUsage) }
        let csvURL = URL(fileURLWithPath: csvPath)
        let labels = try ReceiptLabels.parse(String(contentsOf: csvURL, encoding: .utf8))
        guard !labels.isEmpty else { throw CLIUsageError("\(csvPath) has no labelled receipts.") }
        let folder = parsed.value("--images").map { URL(fileURLWithPath: $0) } ?? csvURL.deletingLastPathComponent()
        var thresholds: [ReceiptField: Double] = [:]
        for field in ReceiptField.allCases {
            if let minimum = try parsed.double("--min-\(field.rawValue)") { thresholds[field] = minimum }
        }

        let reader = try await ReceiptReader(modelArgument: parsed.value("--model"), languages: parsed.values("--lang"))
        var baseline = ReceiptScorecard()
        var model = ReceiptScorecard()
        var rows: [[String: Any]] = []
        for (index, label) in labels.enumerated() {
            let result = try await reader.read(folder.appendingPathComponent(label.file))
            let baselineHits = baseline.add(result.baseline, label: label.fields)
            var row: [String: Any] = ["file": label.file, "label": json(label.fields),
                                      "baseline": json(result.baseline), "baselineCorrect": json(baselineHits)]
            var line = "[\(index + 1)/\(labels.count)] \(label.file)  baseline \(marks(baselineHits))"
            if let read = result.model {
                let hits = model.add(read.fields, label: label.fields)
                row["model"] = json(read.fields)
                row["modelCorrect"] = json(hits)
                row["anchors"] = json(read.anchors)
                row["seconds"] = read.seconds
                line += "  model \(marks(hits))  \(seconds(read.seconds))"
            }
            print(line)
            rows.append(row)
        }

        print("\nfield     baseline   model      threshold")
        for field in ReceiptField.allCases {
            let modelColumn = reader.hasModel ? percent(model.accuracy(field)) : "—"
            let threshold = thresholds[field].map { percent($0) } ?? "—"
            print(field.rawValue.padding(toLength: 10, withPad: " ", startingAt: 0)
                  + percent(baseline.accuracy(field)).padding(toLength: 11, withPad: " ", startingAt: 0)
                  + modelColumn.padding(toLength: 11, withPad: " ", startingAt: 0) + threshold)
        }
        let gated = reader.hasModel ? model : baseline
        let failures = gated.failures(against: thresholds)

        if let reportPath = parsed.value("--report") {
            var summary: [String: Any] = [
                "receipts": labels.count,
                "thresholds": Dictionary(uniqueKeysWithValues: thresholds.map { ($0.key.rawValue, $0.value) }),
                "baselineAccuracy": accuracies(baseline), "failedFields": failures.map(\.rawValue),
                "receiptsDetail": rows,
            ]
            if reader.hasModel {
                summary["model"] = parsed.value("--model")
                summary["modelAccuracy"] = accuracies(model)
            }
            let data = try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: reportPath))
        }
        guard !thresholds.isEmpty else { return }
        if failures.isEmpty {
            print("\ngate: PASS (\(reader.hasModel ? "model" : "baseline"))")
        } else {
            print("\ngate: FAIL — \(failures.map(\.rawValue).joined(separator: ", ")) below threshold")
            exit(1)
        }
    }

    // MARK: - Formatting

    private static func describe(_ fields: ReceiptFields) -> String {
        let total = fields.total.map { "\($0)" } ?? "—"
        let tax = fields.tax.map { "\($0)" } ?? "—"
        return "merchant \(fields.merchant ?? "—") | date \(fields.date ?? "—") | total \(total) "
            + "\(fields.currency ?? "") | tax \(tax)"
    }

    private static func describe(_ anchors: ReceiptAnchors) -> String {
        func mark(_ value: Bool?) -> String { value.map { $0 ? "yes" : "NO" } ?? "—" }
        return "merchant \(mark(anchors.merchant)) | date \(mark(anchors.date)) | total \(mark(anchors.total)) "
            + "| tax \(mark(anchors.tax))"
    }

    private static func marks(_ hits: [ReceiptField: Bool]) -> String {
        ReceiptField.allCases.map { "\($0.rawValue) \(hits[$0] == true ? "✓" : "✗")" }.joined(separator: " ")
    }

    private static func seconds(_ value: Double) -> String { "\(fixed(value, digits: 1)) s" }

    private static func accuracies(_ card: ReceiptScorecard) -> [String: Double] {
        Dictionary(uniqueKeysWithValues: ReceiptField.allCases.map { ($0.rawValue, card.accuracy($0)) })
    }

    private static func json(_ fields: ReceiptFields) -> [String: Any] {
        ["merchant": fields.merchant ?? NSNull(), "date": fields.date ?? NSNull(),
         "total": fields.total.map { NSDecimalNumber(decimal: $0) } ?? NSNull(),
         "tax": fields.tax.map { NSDecimalNumber(decimal: $0) } ?? NSNull(), "currency": fields.currency ?? NSNull()]
    }

    private static func json(_ hits: [ReceiptField: Bool]) -> [String: Bool] {
        Dictionary(uniqueKeysWithValues: hits.map { ($0.key.rawValue, $0.value) })
    }

    private static func json(_ anchors: ReceiptAnchors) -> [String: Any] {
        ["merchant": anchors.merchant ?? NSNull(), "date": anchors.date ?? NSNull(),
         "total": anchors.total ?? NSNull(), "tax": anchors.tax ?? NSNull()]
    }
}

// MARK: - ReceiptReader

/// Reads one receipt image: Vision's document reader, the baseline, and the model when one was given. The model is
/// loaded once and reused for every receipt.
@MainActor
final class ReceiptReader {
    struct ModelRead {
        let fields: ReceiptFields
        let anchors: ReceiptAnchors
        let seconds: Double
    }

    struct Reading {
        let baseline: ReceiptFields
        let model: ModelRead?
    }

    private let session: AuraSession?
    private let languages: [String]
    var hasModel: Bool { session != nil }

    init(modelArgument: String?, languages: [String]) async throws {
        self.languages = languages
        let tool = VisionDocumentTool()
        if let reason = await tool.availability().reason {
            throw CLIUsageError.unavailable(tool.displayName, reason: reason)
        }
        if let argument = modelArgument {
            let model = try Self.model(for: argument)
            session = try await AuraSession(profile: ReceiptPrompt.profile(model: model))
        } else {
            session = nil
        }
    }

    /// A catalog id (`qwen35_2b_text`), or a `.gguf` path or URL for the spike's GGUF fallback.
    static func model(for argument: String) throws -> Model {
        if let model = ModelRegistry.shared.model(id: argument) { return model }
        let source = argument.hasPrefix("/") || argument.hasPrefix(".")
            ? URL(fileURLWithPath: argument).absoluteString : argument
        guard let model = Model.fromURL(source) else {
            throw CLIUsageError("\(argument) is neither a catalog model id nor a .gguf path or URL.")
        }
        return model
    }

    func read(_ image: URL) async throws -> Reading {
        let document = try await Self.document(in: try Data(contentsOf: image), languages: languages)
        let baseline = ReceiptHeuristic.extract(from: document)
        guard let session else { return Reading(baseline: baseline, model: nil) }
        let start = Date()
        var reply = ""
        for try await chunk in session.stream(ReceiptPrompt.prompt(for: document)) { reply += chunk }
        let fields = try ReceiptFields.decode(reply)
        return Reading(baseline: baseline, model: ModelRead(
            fields: fields, anchors: ReceiptAnchors.check(fields, in: document),
            seconds: Date().timeIntervalSince(start)))
    }

    /// The document with the most text; an image with none reads as an empty document.
    private static func document(in data: Data, languages: [String]) async throws -> VisionDocumentTool.Document {
        #if compiler(>=6.2)
        if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) {
            let documents = try await VisionDocumentTool().readDocuments(inImageData: data, languages: languages)
            return documents.max { $0.text.count < $1.text.count } ?? VisionDocumentTool.Document(text: "")
        }
        #endif
        // init has already checked availability, so an earlier system never gets here.
        return VisionDocumentTool.Document(text: "")
    }
}
