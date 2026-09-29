import SwiftUI
import PhotosUI
import UniformTypeIdentifiers
import AuraCore

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

struct MLTextToolsView: View {
    @State private var text = "Tim Cook presentó el nuevo iPhone en Cupertino. "
        + "La presentación fue excelente y el público quedó encantado."
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
                            Text("Neutral text scored from -0.8 to 0.4 depending on the language.")
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

struct MLImageToolsView: View {
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
                            LabeledContent(
                                code.payload ?? "(binary payload)",
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

struct MLAudioToolsView: View {
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
                        if results.isEmpty {
                            Text("Nothing analysed: no audio could be decoded from the file.")
                                .foregroundStyle(.secondary)
                        }
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
