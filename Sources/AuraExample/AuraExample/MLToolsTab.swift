import SwiftUI
import PhotosUI
import UniformTypeIdentifiers
import CoreGraphics
import CoreText
import CoreImage
import CoreImage.CIFilterBuiltins
import ImageIO
import AVFoundation
import AuraCore

// MARK: - MLToolsTab
//
// Usage reference for AuraCore's on-device ML tools: every tool is an Apple framework
// (Vision, NaturalLanguage, SoundAnalysis, Core ML, Create ML) — no download, no network.

struct MLToolsTab: View {
    private enum Pane: String, CaseIterable, Identifiable {
        case tools = "Tools", text = "Text", image = "Image", audio = "Audio", train = "Train"
        var id: String { rawValue }
    }

    @State private var selection: Pane = .tools

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                Picker("Section", selection: $selection) {
                    ForEach(Pane.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)
                .padding(.top, 8)

                switch selection {
                case .tools: MLToolListView()
                case .text: MLTextToolsView()
                case .image: MLImageToolsView()
                case .audio: MLAudioToolsView()
                case .train: MLTrainingView()
                }
            }
            .navigationTitle("On-device ML")
        }
    }
}

// MARK: - Tools (discovery)

private struct MLToolListView: View {
    @State private var tools: [SystemToolRegistry.ToolInfo] = []

    var body: some View {
        List {
            ForEach(SystemToolCategory.allCases, id: \.self) { category in
                Section(category.displayName) {
                    let members = tools.filter { $0.category == category }
                    if category == .customModel && members.isEmpty {
                        Text("CoreMLModelTool and TextClassifierTool wrap one model file each, so they are created with its URL instead of being listed here. The Train section builds one.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    ForEach(members) { MLToolRow(tool: $0) }
                }
            }
        }
        .task { tools = await SystemToolRegistry.discover() }
    }
}

private struct MLToolRow: View {
    let tool: SystemToolRegistry.ToolInfo

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label(tool.displayName, systemImage: tool.isAvailable ? "checkmark.circle.fill" : "xmark.circle")
                .foregroundStyle(tool.isAvailable ? .green : .orange)
                .font(.subheadline.weight(.semibold))
            Text(tool.id).font(.caption.monospaced()).foregroundStyle(.secondary)
            Text(tool.availability.reason ?? tool.summary).font(.caption)
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Text (language, entities, sentiment)

private nonisolated struct TextAnalysis: Sendable {
    let languageGuess: NLLanguageIdentificationTool.Identification
    let entities: MLOutcome<[NLEntityRecognitionTool.Entity]>
    let sentiment: MLOutcome<Double?>

    static func make(for text: String) async -> TextAnalysis {
        let entityTool = NLEntityRecognitionTool()
        let sentimentTool = NLSentimentTool()
        return TextAnalysis(
            languageGuess: NLLanguageIdentificationTool().identify(text, maxHypotheses: 3),
            entities: await MLOutcome.run(entityTool) { entityTool.entities(in: text) },
            sentiment: await MLOutcome.run(sentimentTool) { sentimentTool.score(text) })
    }
}

private struct MLTextToolsView: View {
    @State private var text = "Tim Cook presentó el nuevo iPhone en Cupertino. La presentación fue excelente y el público quedó encantado."
    @State private var analysis: TextAnalysis?
    @State private var isRunning = false

    var body: some View {
        Form {
            Section("Text") {
                TextEditor(text: $text)
                    .frame(minHeight: 110)
                    .font(.body)
                Button {
                    analyze()
                } label: {
                    Label("Analyze", systemImage: "text.magnifyingglass")
                }
                .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isRunning)
            }
            if let analysis {
                Section("Language — NLLanguageIdentificationTool") {
                    LabeledContent("Dominant", value: analysis.languageGuess.dominantLanguage ?? "undetermined")
                    ForEach(analysis.languageGuess.hypotheses, id: \.language) { hypothesis in
                        ProgressView(value: hypothesis.probability) {
                            Text("\(hypothesis.language)  \(MLFormat.percent(hypothesis.probability))").font(.caption)
                        }
                    }
                }
                Section("Entities — NLEntityRecognitionTool") {
                    MLOutcomeView(outcome: analysis.entities) { entities in
                        if entities.isEmpty {
                            Text("No people, places or organizations found.").foregroundStyle(.secondary)
                        }
                        ForEach(entities, id: \.range.location) { entity in
                            LabeledContent(entity.text, value: entity.kind.rawValue)
                        }
                    }
                }
                Section("Sentiment — NLSentimentTool") {
                    MLOutcomeView(outcome: analysis.sentiment) { score in
                        if let score {
                            Gauge(value: score, in: -1...1) {
                                Text("Score")
                            } currentValueLabel: {
                                Text(MLFormat.fixed(score))
                            }
                            Text("Scores lean negative: treat only values near -1 as clearly negative.")
                                .font(.caption).foregroundStyle(.secondary)
                        } else {
                            Text("No sentiment model for this language (en, es, fr, de, it and pt have one).")
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
        .formStyle(.grouped)
    }

    private func analyze() {
        let input = text
        isRunning = true
        Task {
            analysis = await Task.detached(priority: .userInitiated) { await TextAnalysis.make(for: input) }.value
            isRunning = false
        }
    }
}

// MARK: - Image (classification, barcodes, faces, OCR lines)

private nonisolated struct ImageAnalysis: Sendable {
    let labels: MLOutcome<[VisionImageClassificationTool.Classification]>
    let barcodes: MLOutcome<[VisionBarcodeTool.DetectedBarcode]>
    let faces: MLOutcome<[VisionFaceDetectionTool.DetectedFace]>
    let lines: MLOutcome<[VisionOCRTool.RecognizedLine]>

    /// Uses the `Data` overloads so EXIF orientation is honoured and boxes match the upright preview.
    static func make(for data: Data) async -> ImageAnalysis {
        let classifier = VisionImageClassificationTool()
        let reader = VisionBarcodeTool()
        let detector = VisionFaceDetectionTool()
        let ocr = VisionOCRTool()
        return ImageAnalysis(
            labels: await MLOutcome.run(classifier) { try classifier.classify(inImageData: data, maxResults: 5) },
            barcodes: await MLOutcome.run(reader) { try reader.detectBarcodes(inImageData: data) },
            faces: await MLOutcome.run(detector) { try detector.detectFaces(inImageData: data) },
            lines: await MLOutcome.run(ocr) { try ocr.recognizeLines(inImageData: data) })
    }

    var boxes: [MLBox] {
        (barcodes.output ?? []).map { MLBox(rect: $0.boundingBox, tint: .blue) }
            + (faces.output ?? []).map { MLBox(rect: $0.boundingBox, tint: .green) }
            + (lines.output ?? []).map { MLBox(rect: $0.boundingBox, tint: .orange) }
    }
}

private struct MLImageToolsView: View {
    @State private var pickerItem: PhotosPickerItem?
    @State private var showingImporter = false
    @State private var imageData: Data?
    @State private var preview: CGImage?
    @State private var analysis: ImageAnalysis?
    @State private var isRunning = false
    @State private var errorText: String?

    var body: some View {
        Form {
            Section("Image") {
                if let preview {
                    Image(decorative: preview, scale: 1)
                        .resizable()
                        .scaledToFit()
                        .overlay { MLBoxOverlay(boxes: analysis?.boxes ?? []) }
                        .frame(maxHeight: 280)
                        .frame(maxWidth: .infinity)
                }
                HStack {
                    PhotosPicker(selection: $pickerItem, matching: .images) {
                        Label("Photos", systemImage: "photo")
                    }
                    Button { showingImporter = true } label: { Label("Files", systemImage: "folder") }
                    Button { load(MLSamples.receiptWithQRCode()) } label: {
                        Label("Sample", systemImage: "qrcode")
                    }
                }
                .buttonStyle(.bordered)
                Button {
                    analyze()
                } label: {
                    Label(isRunning ? "Analyzing…" : "Analyze", systemImage: "eye")
                }
                .disabled(imageData == nil || isRunning)
                if let errorText {
                    Text(errorText).foregroundStyle(.red).font(.caption)
                }
            }
            if let analysis {
                Section("Classification — VisionImageClassificationTool") {
                    MLOutcomeView(outcome: analysis.labels) { labels in
                        if labels.isEmpty { Text("No label above 10 %.").foregroundStyle(.secondary) }
                        ForEach(labels, id: \.identifier) { label in
                            LabeledContent(label.identifier, value: MLFormat.percent(Double(label.confidence)))
                        }
                    }
                }
                Section("Barcodes — VisionBarcodeTool (blue)") {
                    MLOutcomeView(outcome: analysis.barcodes) { codes in
                        if codes.isEmpty { Text("No barcode found.").foregroundStyle(.secondary) }
                        ForEach(Array(codes.enumerated()), id: \.offset) { _, code in
                            LabeledContent(code.payload ?? "(binary payload)",
                                           value: code.symbology.replacingOccurrences(of: "VNBarcodeSymbology", with: ""))
                        }
                    }
                }
                Section("Faces — VisionFaceDetectionTool (green)") {
                    MLOutcomeView(outcome: analysis.faces) { faces in
                        Text(faces.isEmpty ? "No face found." : "\(faces.count) face(s) — detection only, no identity.")
                            .foregroundStyle(faces.isEmpty ? .secondary : .primary)
                    }
                }
                Section("Text lines — VisionOCRTool.recognizeLines (orange)") {
                    MLOutcomeView(outcome: analysis.lines) { lines in
                        if lines.isEmpty { Text("No text found.").foregroundStyle(.secondary) }
                        ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
                            LabeledContent(line.text, value: MLFormat.percent(Double(line.confidence)))
                        }
                    }
                }
            }
        }
        .formStyle(.grouped)
        .onChange(of: pickerItem) { _, item in
            Task { load(try? await item?.loadTransferable(type: Data.self)) }
        }
        .fileImporter(isPresented: $showingImporter, allowedContentTypes: [.image]) { result in
            load(try? MLSamples.readImported(result.get()))
        }
    }

    private func load(_ data: Data?) {
        guard let data else { return }
        analysis = nil
        guard let image = MLSamples.uprightPreview(of: data) else {
            errorText = "That file is not a readable image."
            return
        }
        errorText = nil
        imageData = data
        preview = image
    }

    private func analyze() {
        guard let data = imageData else { return }
        isRunning = true
        Task {
            analysis = await Task.detached(priority: .userInitiated) { await ImageAnalysis.make(for: data) }.value
            isRunning = false
        }
    }
}

// MARK: - Audio (sound classification)

private struct MLAudioToolsView: View {
    @State private var showingImporter = false
    @State private var audioURL: URL?
    @State private var usesMean = false
    @State private var sounds: MLOutcome<[SoundClassificationTool.Classification]>?
    @State private var isRunning = false

    var body: some View {
        Form {
            Section("Audio file") {
                LabeledContent("File", value: audioURL?.lastPathComponent ?? "none")
                HStack {
                    Button { showingImporter = true } label: { Label("Files", systemImage: "folder") }
                    Button { audioURL = try? MLSamples.toneFile() } label: {
                        Label("440 Hz tone", systemImage: "waveform")
                    }
                }
                .buttonStyle(.bordered)
                Picker("Score", selection: $usesMean) {
                    Text("Peak (occurs anywhere)").tag(false)
                    Text("Mean (share of the file)").tag(true)
                }
                Button {
                    classify()
                } label: {
                    Label(isRunning ? "Listening…" : "Classify sounds", systemImage: "ear")
                }
                .disabled(audioURL == nil || isRunning)
            }
            if let sounds {
                Section("SoundClassificationTool") {
                    MLOutcomeView(outcome: sounds) { results in
                        if results.isEmpty { Text("Nothing analysed: the file has no audio.").foregroundStyle(.secondary) }
                        ForEach(results, id: \.identifier) { sound in
                            ProgressView(value: sound.confidence) {
                                Text("\(sound.identifier)  \(MLFormat.percent(sound.confidence))").font(.caption)
                            }
                        }
                    }
                }
            }
        }
        .formStyle(.grouped)
        .fileImporter(isPresented: $showingImporter, allowedContentTypes: [.audio]) { result in
            audioURL = try? MLSamples.copyImported(result.get())
            sounds = nil
        }
    }

    private func classify() {
        guard let url = audioURL else { return }
        let scoring: SoundClassificationTool.Aggregation = usesMean ? .mean : .peak
        isRunning = true
        Task {
            let tool = SoundClassificationTool()
            sounds = await MLOutcome.run(tool) {
                try await tool.classify(audioFileAt: url, maxResults: 5, aggregation: scoring)
            }
            isRunning = false
        }
    }
}

// MARK: - Train (Create ML → Core ML → classify)

private struct MLTrainingView: View {
    var body: some View {
        #if os(macOS)
        MLTrainingDemo()
        #else
        ContentUnavailableView(
            "Training demo runs on macOS",
            systemImage: "hammer",
            description: Text("TextClassifierTrainer reports its own availability; the iOS and visionOS Simulators lack Create ML."))
        #endif
    }
}

#if os(macOS)
private struct MLTrainingDemo: View {
    @State private var usesBERT = true
    @State private var trainingReport: TextClassifierTrainer.Report?
    @State private var modelDescription: CoreMLModelTool.ModelInfo?
    @State private var classifier: TextClassifierTool?
    @State private var query = "Pagué el taxi del hotel a la estación"
    @State private var prediction: TextClassifierTool.Classification?
    @State private var isTraining = false
    @State private var errorText: String?

    var body: some View {
        Form {
            Section("Dataset") {
                Text("\(MLSamples.expenses.count) labelled expenses in Spanish and English: food, transport, housing.")
                DisclosureGroup("Examples") {
                    ForEach(Array(MLSamples.expenses.enumerated()), id: \.offset) { _, example in
                        LabeledContent(example.text, value: example.label).font(.caption)
                    }
                }
            }
            Section("Train — TextClassifierTrainer") {
                Picker("Algorithm", selection: $usesBERT) {
                    Text("BERT transfer learning (~10 s, generalises)").tag(true)
                    Text("Maximum entropy (instant, needs more data)").tag(false)
                }
                Button {
                    Task { await train() }
                } label: {
                    Label(isTraining ? "Training…" : "Train model", systemImage: "hammer")
                }
                .disabled(isTraining)
                if let trainingReport {
                    LabeledContent("Labels", value: trainingReport.classLabels.joined(separator: ", "))
                    LabeledContent("Training accuracy", value: trainingReport.trainingAccuracy.map(MLFormat.percent) ?? "n/a")
                    LabeledContent("Validation accuracy (20 % held out)",
                                   value: trainingReport.validationAccuracy.map(MLFormat.percent) ?? "n/a")
                    Text(trainingReport.modelURL.path).font(.caption.monospaced()).textSelection(.enabled)
                }
                if let errorText {
                    Text(errorText).foregroundStyle(.red).font(.caption)
                }
            }
            if let classifier {
                Section("Classify — TextClassifierTool") {
                    TextField("Expense", text: $query)
                    Button("Classify") {
                        Task { prediction = try? await classifier.classify(query, maxHypotheses: 3) }
                    }
                    if let prediction {
                        ForEach(prediction.ranked(), id: \.label) { hypothesis in
                            ProgressView(value: hypothesis.probability) {
                                Text("\(hypothesis.label)  \(MLFormat.percent(hypothesis.probability))").font(.caption)
                            }
                        }
                    }
                }
            }
            if let modelDescription {
                Section("Inspect — CoreMLModelTool.describe()") {
                    ForEach(modelDescription.inputs, id: \.name) { LabeledContent("Input \($0.name)", value: $0.kind.rawValue) }
                    ForEach(modelDescription.outputs, id: \.name) { LabeledContent("Output \($0.name)", value: $0.kind.rawValue) }
                    LabeledContent("Author", value: modelDescription.author ?? "—")
                }
            }
        }
        .formStyle(.grouped)
    }

    private func train() async {
        isTraining = true
        defer { isTraining = false }
        errorText = nil
        let destination = FileManager.default.temporaryDirectory
            .appendingPathComponent("AuraExample", isDirectory: true)
            .appendingPathComponent("Expenses.mlmodel")
        do {
            let result = try await TextClassifierTrainer().train(
                examples: MLSamples.expenses,
                writingModelTo: destination,
                algorithm: usesBERT ? .transferLearning(.bertEmbedding) : .maxEnt,
                validation: .holdOut(fraction: 0.2, seed: 7))
            trainingReport = result
            // A new tool per training run: an instance keeps the model it compiled first.
            let trained = TextClassifierTool(modelAt: result.modelURL)
            classifier = trained
            prediction = try await trained.classify(query, maxHypotheses: 3)
            modelDescription = try await CoreMLModelTool(modelAt: result.modelURL).describe()
        } catch {
            errorText = error.localizedDescription
        }
    }
}
#endif

// MARK: - Shared pieces

/// One tool's result, or why it could not run on this device.
nonisolated enum MLOutcome<Value: Sendable>: Sendable {
    case done(Value)
    case unavailable(String)
    case failed(String)

    var output: Value? {
        if case .done(let value) = self { return value }
        return nil
    }

    /// Checks `availability()` first: some tools report unavailable (e.g. in the Simulator)
    /// instead of failing loudly.
    static func run(_ tool: some SystemTool, _ body: () async throws -> Value) async -> MLOutcome<Value> {
        if let reason = await tool.availability().reason { return .unavailable(reason) }
        do {
            return .done(try await body())
        } catch {
            return .failed(error.localizedDescription)
        }
    }
}

private struct MLOutcomeView<Value: Sendable, Content: View>: View {
    let outcome: MLOutcome<Value>
    @ViewBuilder let content: (Value) -> Content

    var body: some View {
        switch outcome {
        case .done(let value): content(value)
        case .unavailable(let reason): Label(reason, systemImage: "slash.circle").foregroundStyle(.orange)
        case .failed(let reason): Label(reason, systemImage: "xmark.octagon").foregroundStyle(.red)
        }
    }
}

nonisolated struct MLBox: Sendable {
    let rect: CGRect
    let tint: Color
}

/// Draws Vision boxes (normalized, bottom-left origin) over the displayed image.
private struct MLBoxOverlay: View {
    let boxes: [MLBox]

    var body: some View {
        GeometryReader { geometry in
            ForEach(Array(boxes.enumerated()), id: \.offset) { _, box in
                Rectangle()
                    .stroke(box.tint, lineWidth: 2)
                    .frame(width: box.rect.width * geometry.size.width,
                           height: box.rect.height * geometry.size.height)
                    .position(x: box.rect.midX * geometry.size.width,
                              y: (1 - box.rect.midY) * geometry.size.height)
            }
        }
    }
}

nonisolated enum MLFormat {
    static func percent(_ fraction: Double) -> String {
        String(format: "%.0f %%", fraction * 100)
    }

    static func fixed(_ value: Double) -> String {
        String(format: "%.2f", value)
    }
}

/// Offline sample inputs so every section can be tried without picking a file.
nonisolated enum MLSamples {
    static let expenses: [TextClassifierTrainer.Example] = expensePairs.map(makeExample)

    private static let expensePairs: [(text: String, label: String)] = [
        ("Almuerzo en el restaurante del centro", "food"), ("Compra semanal en el supermercado", "food"),
        ("Café y croissant en la panadería", "food"), ("Cena con amigos, pizza y cervezas", "food"),
        ("Pedido de comida a domicilio", "food"), ("Fruta y verdura del mercado", "food"),
        ("Lunch at the sushi place", "food"), ("Groceries at the supermarket", "food"),
        ("Coffee and a bagel downtown", "food"), ("Dinner delivery from the Thai restaurant", "food"),
        ("Billete de metro mensual", "transport"), ("Taxi al aeropuerto", "transport"),
        ("Gasolina para el coche", "transport"), ("Tren de alta velocidad a Barcelona", "transport"),
        ("Parking en el centro comercial", "transport"), ("Uber to the office", "transport"),
        ("Monthly bus pass", "transport"), ("Filled up the car with gas", "transport"),
        ("Train ticket to Boston", "transport"), ("Airport parking for three days", "transport"),
        ("Alquiler del piso de octubre", "housing"), ("Factura de la luz", "housing"),
        ("Recibo del agua y la basura", "housing"), ("Cuota de la comunidad de vecinos", "housing"),
        ("Reparación de la caldera", "housing"), ("Monthly rent payment", "housing"),
        ("Electricity bill", "housing"), ("Internet and cable for the apartment", "housing"),
        ("Plumber fixed the kitchen sink", "housing"), ("Home insurance renewal", "housing"),
    ]

    private static func makeExample(_ pair: (text: String, label: String)) -> TextClassifierTrainer.Example {
        TextClassifierTrainer.Example(text: pair.text, label: pair.label)
    }

    /// A white canvas with two lines of text and a QR code, encoded as PNG.
    static func receiptWithQRCode() -> Data? {
        let size = CGSize(width: 900, height: 420)
        guard let canvas = CGContext(
            data: nil, width: Int(size.width), height: Int(size.height), bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let code = qrCode("https://github.com/iOSDevC/AuraLocal") else { return nil }
        canvas.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        canvas.fill(CGRect(origin: .zero, size: size))
        draw("CAFÉ CENTRAL  4,50 EUR", in: canvas, at: CGPoint(x: 40, y: 300))
        draw("Tarjeta ****1234", in: canvas, at: CGPoint(x: 40, y: 220))
        canvas.interpolationQuality = .none
        canvas.draw(code, in: CGRect(x: 560, y: 40, width: 300, height: 300))
        guard let image = canvas.makeImage() else { return nil }
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output, UTType.png.identifier as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(destination, image, nil)
        return CGImageDestinationFinalize(destination) ? output as Data : nil
    }

    private static func qrCode(_ text: String) -> CGImage? {
        let generator = CIFilter.qrCodeGenerator()
        generator.message = Data(text.utf8)
        generator.correctionLevel = "M"
        guard let output = generator.outputImage else { return nil }
        return CIContext().createCGImage(output, from: output.extent)
    }

    private static func draw(_ text: String, in canvas: CGContext, at point: CGPoint) {
        let font = CTFontCreateWithName("Helvetica" as CFString, 36, nil)
        let attributes: [CFString: Any] = [
            kCTFontAttributeName: font,
            kCTForegroundColorAttributeName: CGColor(red: 0, green: 0, blue: 0, alpha: 1),
        ]
        guard let attributed = CFAttributedStringCreate(nil, text as CFString, attributes as CFDictionary) else { return }
        canvas.textPosition = point
        CTLineDraw(CTLineCreateWithAttributedString(attributed), canvas)
    }

    /// Upright (EXIF-applied) preview, so Vision boxes from the `Data` overloads line up with it.
    static func uprightPreview(of data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: 1600,
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }

    /// A 3 s, 440 Hz sine tone as a 16 kHz mono WAV in the temporary directory.
    static func toneFile() throws -> URL {
        let sampleRate = 16_000.0
        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(sampleRate * 3)),
              let samples = buffer.floatChannelData?[0] else {
            throw CocoaError(.fileWriteUnknown)
        }
        buffer.frameLength = buffer.frameCapacity
        let step = Float(2 * Double.pi * 440 / sampleRate)
        var phase: Float = 0
        for index in 0..<Int(buffer.frameLength) {
            samples[index] = 0.5 * sin(phase)
            phase += step
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("aura-tone-440hz.wav")
        try? FileManager.default.removeItem(at: url)
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
        file.close()
        return url
    }

    static func readImported(_ url: URL) throws -> Data {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        return try Data(contentsOf: url)
    }

    /// Copies a picked file out of its security scope so later async reads keep working.
    static func copyImported(_ url: URL) throws -> URL {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let copy = FileManager.default.temporaryDirectory
            .appendingPathComponent("aura-ml-\(UUID().uuidString)")
            .appendingPathExtension(url.pathExtension)
        try FileManager.default.copyItem(at: url, to: copy)
        return copy
    }
}
