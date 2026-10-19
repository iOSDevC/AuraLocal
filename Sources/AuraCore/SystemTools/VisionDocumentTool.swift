import Foundation
import Vision
import CoreGraphics
import ImageIO
#if compiler(>=6.2) && canImport(DataDetection)
import DataDetection
#endif

/// Structured document reading via Vision's `RecognizeDocumentsRequest` (iOS 26, macOS 26, visionOS 26): the
/// page's text, paragraphs and tables, plus the amounts, dates and contacts Data Detection finds in it. A receipt's
/// total and line items come back as values and cells instead of lines to re-parse. Earlier systems report the tool
/// unavailable; ``VisionOCRTool`` reads them line by line. Resilient: an image with no document returns an empty
/// list (not an error); a bad image throws a typed error.
public struct VisionDocumentTool: SystemTool {
    public let id = "system.vision.document"
    public let displayName = "Document reader (Vision)"
    public let summary = "Read a document's paragraphs, tables, amounts and dates with Apple's Vision framework (iOS 26, macOS 26) — no model download, works offline."
    public let category = SystemToolCategory.vision

    public init() {}

    public enum ToolError: LocalizedError {
        case invalidImage
        public var errorDescription: String? {
            switch self {
            case .invalidImage: "The provided data is not a decodable image."
            }
        }
    }

    /// One document Vision found in the image. Boxes are normalized 0…1 with Vision's bottom-left origin.
    public struct Document: Sendable, Equatable {
        /// Every recognized line, in reading order, joined by newlines.
        public let text: String
        public let title: String?
        public let paragraphs: [Paragraph]
        /// The same lines ``VisionOCRTool/recognizeLines(in:languages:accurate:)`` returns, with their boxes.
        public let lines: [VisionOCRTool.RecognizedLine]
        public let tables: [Table]
        /// Matches found in ``text``, in document order.
        public let detections: [DetectedData]

        /// The ``DetectedData/DataKind/moneyAmount`` matches, e.g. a receipt's subtotal, tax and total.
        public var amounts: [DetectedData] { detections.filter { $0.kind == .moneyAmount } }

        /// For fixtures and tests; ``VisionDocumentTool`` builds these from Vision's observations.
        public init(text: String, title: String? = nil, paragraphs: [Paragraph] = [],
                    lines: [VisionOCRTool.RecognizedLine] = [], tables: [Table] = [],
                    detections: [DetectedData] = []) {
            self.text = text
            self.title = title
            self.paragraphs = paragraphs
            self.lines = lines
            self.tables = tables
            self.detections = detections
        }
    }

    public struct Paragraph: Sendable, Equatable {
        public let text: String
        public let boundingBox: CGRect

        public init(text: String, boundingBox: CGRect) {
            self.text = text
            self.boundingBox = boundingBox
        }
    }

    public struct Table: Sendable, Equatable {
        /// Cells row by row, as Vision groups them. A cell spanning rows or columns says so in its ranges.
        public let rows: [[Cell]]
        public let boundingBox: CGRect

        /// Each row's cell texts, for a caller that needs only the grid.
        public var textRows: [[String]] { rows.map { $0.map(\.text) } }

        public init(rows: [[Cell]], boundingBox: CGRect) {
            self.rows = rows
            self.boundingBox = boundingBox
        }
    }

    public struct Cell: Sendable, Equatable {
        public let text: String
        public let rowRange: ClosedRange<Int>
        public let columnRange: ClosedRange<Int>
        public let boundingBox: CGRect

        public init(text: String, rowRange: ClosedRange<Int>, columnRange: ClosedRange<Int>, boundingBox: CGRect) {
            self.text = text
            self.rowRange = rowRange
            self.columnRange = columnRange
            self.boundingBox = boundingBox
        }
    }

    /// A span of the document's text that Data Detection matched to a kind of data.
    public struct DetectedData: Sendable, Equatable {
        public enum DataKind: String, Sendable, CaseIterable {
            case moneyAmount, date, emailAddress, phoneNumber, postalAddress, link
            case flightNumber, shipmentTrackingNumber, paymentIdentifier, measurement
        }

        public let kind: DataKind
        /// The matched span of ``Document/text``; empty when Vision gives no range for the match.
        public let text: String
        /// `nil` when Vision gives no region for the span.
        public let boundingBox: CGRect?
        /// ``DataKind/moneyAmount`` only.
        public let amount: Decimal?
        /// ISO 4217 code (`USD`, `MXN`…), ``DataKind/moneyAmount`` only.
        public let currency: String?
        /// The start of a ``DataKind/date`` match.
        public let startDate: Date?

        public init(kind: DataKind, text: String, boundingBox: CGRect? = nil, amount: Decimal? = nil,
                    currency: String? = nil, startDate: Date? = nil) {
            self.kind = kind
            self.text = text
            self.boundingBox = boundingBox
            self.amount = amount
            self.currency = currency
            self.startDate = startDate
        }
    }

    public func availability() async -> SystemToolAvailability {
        #if compiler(>=6.2)
        if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) {
            return .available
        }
        #endif
        return .unavailable(reason: "Vision document reading needs iOS 26, macOS 26 or visionOS 26.")
    }
}

#if compiler(>=6.2)
@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
public extension VisionDocumentTool {

    /// Read the documents in a `CGImage`, one ``Document`` per document Vision finds. `languages` (BCP-47) hints
    /// recognition; empty = automatic, as in ``VisionOCRTool``.
    func readDocuments(in image: CGImage, languages: [String] = []) async throws -> [Document] {
        try await Self.documents(in: image, orientation: nil, languages: languages)
    }

    /// Read raw image bytes (PNG/JPEG/HEIC/…). Boxes follow the image's EXIF orientation, i.e. the upright image as
    /// displayed. Throws ``ToolError/invalidImage`` if undecodable.
    func readDocuments(inImageData data: Data, languages: [String] = []) async throws -> [Document] {
        guard let decoded = VisionImageInput.decode(data) else {
            throw ToolError.invalidImage
        }
        return try await Self.documents(in: decoded.image, orientation: decoded.orientation, languages: languages)
    }

    private static func documents(
        in image: CGImage, orientation: CGImagePropertyOrientation?, languages: [String]
    ) async throws -> [Document] {
        var request = RecognizeDocumentsRequest()
        // Without languages Vision assumes en-US and silently misses non-Latin scripts.
        if languages.isEmpty {
            request.textRecognitionOptions.automaticallyDetectLanguage = true
        } else {
            request.textRecognitionOptions.recognitionLanguages = languages.map { Locale.Language(identifier: $0) }
        }
        return try await request.perform(on: image, orientation: orientation).map { document(from: $0.document) }
    }

    private static func document(from container: DocumentObservation.Container) -> Document {
        Document(
            text: container.text.transcript,
            title: container.title.map(\.transcript),
            paragraphs: container.paragraphs.map {
                Paragraph(text: $0.transcript, boundingBox: $0.boundingRegion.boundingBox.cgRect)
            },
            lines: container.text.lines.compactMap(line(from:)),
            tables: container.tables.map(table(from:)),
            detections: container.text.detectedData.compactMap { detection(from: $0.match, in: container.text) })
    }

    private static func line(from observation: RecognizedTextObservation) -> VisionOCRTool.RecognizedLine? {
        guard let best = observation.topCandidates(1).first else { return nil }
        return VisionOCRTool.RecognizedLine(text: best.string, confidence: best.confidence,
                                            boundingBox: observation.boundingBox.cgRect)
    }

    private static func table(from table: DocumentObservation.Container.Table) -> Table {
        Table(rows: table.rows.map { row in
                  row.map { cell in
                      Cell(text: cell.content.text.transcript, rowRange: cell.rowRange,
                           columnRange: cell.columnRange,
                           boundingBox: cell.content.text.boundingRegion.boundingBox.cgRect)
                  }
              },
              boundingBox: table.boundingRegion.boundingBox.cgRect)
    }

    private static func detection(
        from match: DataDetector.Match, in text: DocumentObservation.Container.Text
    ) -> DetectedData? {
        let transcript = text.transcript
        var span = ""
        var box: CGRect?
        if let range = match.range, range.lowerBound >= transcript.startIndex, range.upperBound <= transcript.endIndex {
            span = String(transcript[range])
            box = text.boundingRegion(for: range)?.boundingBox.cgRect
        }
        func found(_ kind: DetectedData.DataKind, amount: Decimal? = nil, currency: String? = nil,
                   startDate: Date? = nil) -> DetectedData {
            DetectedData(kind: kind, text: span, boundingBox: box, amount: amount, currency: currency,
                         startDate: startDate)
        }
        switch match.details {
        case .moneyAmount(let money): return found(.moneyAmount, amount: money.amount, currency: money.currency.identifier)
        case .calendarEvent(let event): return found(.date, startDate: event.startDate)
        case .emailAddress: return found(.emailAddress)
        case .phoneNumber: return found(.phoneNumber)
        case .postalAddress: return found(.postalAddress)
        case .link: return found(.link)
        case .flightNumber: return found(.flightNumber)
        case .shipmentTrackingNumber: return found(.shipmentTrackingNumber)
        case .paymentIdentifier: return found(.paymentIdentifier)
        case .measurement: return found(.measurement)
        @unknown default: return nil
        }
    }
}
#endif
