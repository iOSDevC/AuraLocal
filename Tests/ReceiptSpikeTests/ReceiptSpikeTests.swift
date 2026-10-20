import XCTest
import CoreGraphics
import TonamoxCore
@testable import ReceiptSpike

final class ReceiptSpikeTests: XCTestCase {

    private static let utc = TimeZone(identifier: "UTC")!

    private static func day(_ year: Int, _ month: Int, _ day: Int) -> Date {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = utc
        return calendar.date(from: DateComponents(year: year, month: month, day: day))!
    }

    private static func document(_ lines: [String], title: String? = nil,
                                 detected: [VisionDocumentTool.DetectedData] = []) -> VisionDocumentTool.Document {
        VisionDocumentTool.Document(
            text: lines.joined(separator: "\n"), title: title,
            lines: lines.map { VisionOCRTool.RecognizedLine(text: $0, confidence: 1, boundingBox: .zero) },
            detections: detected)
    }

    /// A Mexican supermarket receipt as Vision would read it.
    private static let plaza = document([
        "SUPERMERCADO LA PLAZA", "Fecha: 03/03/2026 14:05", "Leche entera 1L   25.50", "Pan dulce x 2   12.00",
        "SUBTOTAL   37.50", "IVA 16%   6.00", "TOTAL   $43.50", "Gracias por su compra",
    ], detected: [
        .init(kind: .date, text: "03/03/2026 14:05", startDate: day(2026, 3, 3)),
        .init(kind: .moneyAmount, text: "$43.50", amount: Decimal(string: "43.50"), currency: "MXN"),
    ])

    // MARK: - Numbers and words

    func testTokensReadBothDecimalConventions() {
        func value(_ token: String) -> ReceiptNumber? { ReceiptText.decimal(fromToken: token) }
        XCTAssertEqual(value("12.75"), ReceiptNumber(value: Decimal(string: "12.75")!, hasFraction: true))
        XCTAssertEqual(value("12,75"), ReceiptNumber(value: Decimal(string: "12.75")!, hasFraction: true))
        XCTAssertEqual(value("1,234.50"), ReceiptNumber(value: Decimal(string: "1234.50")!, hasFraction: true))
        XCTAssertEqual(value("1.234,50"), ReceiptNumber(value: Decimal(string: "1234.50")!, hasFraction: true))
        XCTAssertEqual(value("1.234"), ReceiptNumber(value: 1234, hasFraction: false))
        XCTAssertEqual(value("2026"), ReceiptNumber(value: 2026, hasFraction: false))
    }

    func testNumbersInALineKeepThousandsTogether() {
        XCTAssertEqual(ReceiptText.numbers(in: "TOTAL $1,234.50 MXN").map(\.value), [Decimal(string: "1234.50")!])
        XCTAssertEqual(ReceiptText.numbers(in: "Pan dulce x 2   12.00").map(\.value), [2, 12])
        XCTAssertEqual(ReceiptText.numbers(in: "IVA 16%   6,00").map(\.hasFraction), [false, true])
    }

    func testKeywordsMatchWholeWordsWithoutAccentsOrCase() {
        XCTAssertTrue(ReceiptText.containsWords("IVA 16%", ReceiptHeuristic.taxWords))
        XCTAssertTrue(ReceiptText.containsWords("I.V.A. incluido", ReceiptHeuristic.taxWords))
        XCTAssertFalse(ReceiptText.containsWords("Tarjeta Activa", ReceiptHeuristic.taxWords))
        XCTAssertTrue(ReceiptText.containsWords("IMPORTE A PAGAR", ReceiptHeuristic.totalWords))
        XCTAssertFalse(ReceiptText.containsWords("SUBTOTAL 37.50", ReceiptHeuristic.totalWords))
        XCTAssertEqual(ReceiptText.key("Café, S.A. de C.V."), "cafesadecv")
    }

    // MARK: - Baseline

    func testBaselineReadsAMexicanReceipt() {
        let fields = ReceiptHeuristic.extract(from: Self.plaza, timeZone: Self.utc)
        XCTAssertEqual(fields, ReceiptFields(merchant: "SUPERMERCADO LA PLAZA", date: "2026-03-03",
                                             total: Decimal(string: "43.50"), tax: 6, currency: "MXN"))
    }

    func testBaselineFallsBackToTheLargestAmountAndPrefersTheTitle() {
        let receipt = Self.document(["Coffee Bar", "Latte 4.50", "Muffin 3.25", "Thank you"], title: "Coffee Bar Inc.")
        let fields = ReceiptHeuristic.extract(from: receipt, timeZone: Self.utc)
        XCTAssertEqual(fields.total, Decimal(string: "4.50"))
        XCTAssertNil(fields.tax)
        XCTAssertNil(fields.date)
        XCTAssertEqual(fields.merchant, "Coffee Bar Inc.")
    }

    func testBaselineTakesTheLastTotalLineNotTheSubtotalOrTax() {
        let receipt = Self.document(["Shop", "Total items 3", "Subtotal 20.00", "Tax 1.60", "Total 21.60",
                                     "Cash 30.00", "Change 8.40"])
        let fields = ReceiptHeuristic.extract(from: receipt, timeZone: Self.utc)
        XCTAssertEqual(fields.total, Decimal(string: "21.60"))
        XCTAssertEqual(fields.tax, Decimal(string: "1.60"))
    }

    // MARK: - Prompt and schema

    func testPromptCarriesTheLinesAndTheDetectedCandidates() {
        let prompt = ReceiptPrompt.prompt(for: Self.plaza, timeZone: Self.utc)
        XCTAssertTrue(prompt.contains("TOTAL   $43.50"))
        XCTAssertTrue(prompt.contains("Amounts found in the text: $43.50 (43.5 MXN)"), prompt)
        XCTAssertTrue(prompt.contains("Dates found in the text: 03/03/2026 14:05 (2026-03-03)"), prompt)
    }

    func testSchemaCompilesAndAcceptsOnlyFullReplies() throws {
        let validator = try OutputSchemaValidator(.json(ReceiptPrompt.schema))
        let full = #"{"merchant": "OXXO", "date": "2026-03-03", "total": 43.5, "tax": null, "currency": "MXN"}"#
        XCTAssertEqual(validator.validate(full), [])
        XCTAssertFalse(validator.validate(#"{"merchant": "OXXO", "total": 43.5}"#).isEmpty)
        XCTAssertFalse(validator.validate(#"{"merchant": "OXXO", "date": "3 March", "total": "43.50", "tax": null, "currency": "MXN"}"#).isEmpty)

        let fields = try ReceiptFields.decode(full)
        XCTAssertEqual(fields.total, Decimal(string: "43.50"))
        XCTAssertNil(fields.tax)
    }

    func testProfileIsGreedyWithTheSchema() {
        let profile = ReceiptPrompt.profile(model: .qwen3_0_6b)
        XCTAssertEqual(profile.sampling.temperature, 0)
        XCTAssertEqual(profile.outputSchema, .json(ReceiptPrompt.schema))
        XCTAssertTrue(profile.tools.isEmpty)
    }

    // MARK: - Anchors

    func testAnchorsFindValuesPrintedOnTheReceipt() {
        let printed = ReceiptFields(merchant: "La Plaza", date: "2026-03-03", total: Decimal(string: "43.50"), tax: 6)
        XCTAssertEqual(ReceiptAnchors.check(printed, in: Self.plaza, timeZone: Self.utc),
                       ReceiptAnchors(merchant: true, date: true, total: true, tax: true))

        let invented = ReceiptFields(merchant: "Walmart", date: "2026-03-13", total: Decimal(string: "99.99"))
        XCTAssertEqual(ReceiptAnchors.check(invented, in: Self.plaza, timeZone: Self.utc),
                       ReceiptAnchors(merchant: false, date: false, total: false, tax: nil))
    }

    func testADateIsAnchoredByItsSpellingWithoutDataDetection() {
        let receipt = Self.document(["Tienda", "13/03/26", "Total 5.00"])
        XCTAssertTrue(ReceiptAnchors.dateAnchored("2026-03-13", in: receipt, timeZone: Self.utc))
        XCTAssertFalse(ReceiptAnchors.dateAnchored("2026-03-03", in: receipt, timeZone: Self.utc),
                       "03/03 must not match inside 13/03")
    }

    // MARK: - Labels and scoring

    func testLabelsParseQuotedFieldsEmptyCellsAndCRLF() throws {
        let csv = "file,merchant,date,total,tax,currency\r\n"
            + "a.jpg,\"OXXO, Tienda 12\",2026-03-03,43.50,6.00,mxn\r\n"
            + "b.png,\"Say \"\"hi\"\" Café\",,12,,\r\n\r\n"
        let labels = try ReceiptLabels.parse(csv)
        XCTAssertEqual(labels, [
            ReceiptLabel(file: "a.jpg", fields: ReceiptFields(merchant: "OXXO, Tienda 12", date: "2026-03-03",
                                                              total: Decimal(string: "43.50"), tax: 6,
                                                              currency: "MXN")),
            ReceiptLabel(file: "b.png", fields: ReceiptFields(merchant: "Say \"hi\" Café", total: 12)),
        ])
    }

    func testLabelsRejectMissingColumnsAndBadValues() {
        XCTAssertThrowsError(try ReceiptLabels.parse("file,merchant,total\na.jpg,X,1.00")) { error in
            XCTAssertEqual(error as? ReceiptLabelError, .missingColumn("date"))
        }
        XCTAssertThrowsError(try ReceiptLabels.parse("file,merchant,date,total\na.jpg,X,03/03/2026,1.00")) { error in
            XCTAssertEqual(error as? ReceiptLabelError, .badValue(line: 2, column: "date", value: "03/03/2026"))
        }
        XCTAssertThrowsError(try ReceiptLabels.parse("file,merchant,date,total\na.jpg,X,2026-03-03,\"43,50\"")) { error in
            XCTAssertEqual(error as? ReceiptLabelError, .badValue(line: 2, column: "total", value: "43,50"))
        }
    }

    func testScorecardCountsFieldsAndAppliesThresholds() {
        var card = ReceiptScorecard()
        let truth = ReceiptFields(merchant: "OXXO Tienda 123", date: "2026-03-03", total: Decimal(string: "43.50"))
        let hits = card.add(ReceiptFields(merchant: "OXXO", date: "2026-03-03", total: Decimal(string: "43.5")),
                            label: truth)
        XCTAssertEqual(hits, [.merchant: true, .date: true, .total: true, .tax: true])
        card.add(ReceiptFields(merchant: "Walmart", date: "2026-03-04", total: Decimal(string: "43.50"), tax: 6),
                 label: truth)
        XCTAssertEqual(card.receipts, 2)
        XCTAssertEqual(card.accuracy(.total), 1)
        XCTAssertEqual(card.accuracy(.merchant), 0.5)
        XCTAssertEqual(card.accuracy(.tax), 0.5, "a tax the receipt does not show must stay empty")
        XCTAssertEqual(card.failures(against: [.total: 0.9, .date: 0.9, .merchant: 0.5]), [.date])
        XCTAssertEqual(ReceiptScorecard().accuracy(.total), 0)
    }

    func testAMissFailsEveryFieldEvenOneTheReceiptDoesNotShow() {
        var card = ReceiptScorecard()
        XCTAssertEqual(card.addMiss(), [.merchant: false, .date: false, .total: false, .tax: false])
        card.add(ReceiptFields(merchant: "OXXO"), label: ReceiptFields(merchant: "OXXO"))
        XCTAssertEqual(card.receipts, 2)
        XCTAssertEqual(card.accuracy(.merchant), 0.5)
        XCTAssertEqual(card.accuracy(.tax), 0.5)
    }
}
