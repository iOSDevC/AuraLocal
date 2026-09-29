import SwiftUI
import AuraCore

// MARK: - Train (Create ML → Core ML → classify)

struct MLTrainingView: View {
    var body: some View {
        #if os(macOS)
        MLTrainingDemo()
        #else
        ContentUnavailableView(
            "Training demo runs on macOS",
            systemImage: "hammer",
            description: Text("TextClassifierTrainer reports its own availability; "
                              + "the iOS and visionOS Simulators lack Create ML."))
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
                    Text("BERT transfer learning (a few seconds)").tag(true)
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
                    LabeledContent("Training accuracy",
                                   value: trainingReport.trainingAccuracy.map(MLFormat.percent) ?? "n/a")
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
                        Task { await classify(with: classifier) }
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
                    ForEach(modelDescription.inputs, id: \.name) {
                        LabeledContent("Input \($0.name)", value: $0.kind.rawValue)
                    }
                    ForEach(modelDescription.outputs, id: \.name) {
                        LabeledContent("Output \($0.name)", value: $0.kind.rawValue)
                    }
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

    private func classify(with trained: TextClassifierTool) async {
        do {
            prediction = try await trained.classify(query, maxHypotheses: 3)
            errorText = nil
        } catch {
            prediction = nil
            errorText = error.localizedDescription
        }
    }
}
#endif
