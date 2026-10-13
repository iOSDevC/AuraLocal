import SwiftUI
import PhotosUI
import AuraCore
import AuraUI
import AuraVoice
import AuraDocs
import AuraAppleIntelligence
import AuraAgents

// MARK: - ContentView
//
// §3.2 sidebarAdaptable: Single TabView that adapts to sidebar (macOS/iPad),
// bottom tab bar (iPhone), or ornament (visionOS).
// §3.2.1 TabSection: Groups tabs logically for density-adaptive display.

struct ContentView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        TabView {
            // MARK: - AI Chat section
            Tab("Text", systemImage: "text.bubble") {
                // Models sits under More on iPhone; this is the one-tap way to search.
                TextChatTab(offersHuggingFaceSearch: true)
            }

            Tab("Voice", systemImage: "mic.fill") {
                VoiceTab()
            }

            // MARK: - Documents section
            TabSection("Documents") {
                Tab("Library", systemImage: "doc.text.magnifyingglass") {
                    DocsTab()
                }

                Tab("Vision", systemImage: "eye") {
                    VisionTab()
                }

                Tab("OCR", systemImage: "doc.viewfinder") {
                    OCRTab()
                }
            }

            // MARK: - Tools section
            TabSection("Tools") {
                Tab("Models", systemImage: "square.stack.3d.up") {
                    ModelsTab()
                }

                Tab("Hybrid", systemImage: "cloud.bolt") {
                    HybridTab()
                }

                Tab("ML", systemImage: "brain") {
                    MLToolsTab()
                }

                Tab("Image", systemImage: "photo.on.rectangle.angled") {
                    ImageGenTab()
                }

                // AgentCrew — requires iOS 26+ with Apple Intelligence
                if #available(iOS 26, macOS 26, *) {
                    Tab("Agents", systemImage: "cpu") {
                        AgentCrewInlineView()
                            .environmentObject(appState)
                    }
                }
            }
        }
        .tabViewStyle(.sidebarAdaptable)
    }
}

// MARK: - AgentCrew Inline View
//
// §Fix #5: Flattened from nested TabView into inline NavigationStack sub-views.
// Previously AgentCrewMainView had its own TabView creating a confusing
// tab-within-tab hierarchy.

@available(iOS 26, macOS 26, *)
struct AgentCrewInlineView: View {
    @EnvironmentObject var appState: AppState
    @State private var selectedSection: AgentSection = .pipeline
    // Present the escalation consent sheet if a crew step escalates to the cloud.
    @ObservedObject private var consentGate = HybridSettings.shared.consent

    enum AgentSection: String, CaseIterable, Identifiable {
        case pipeline   = "Pipeline"
        case documents  = "Documents"
        case history    = "History"
        var id: String { rawValue }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                Picker("Section", selection: $selectedSection) {
                    ForEach(AgentSection.allCases) { section in
                        Text(section.rawValue).tag(section)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)
                .padding(.top, 8)

                switch selectedSection {
                case .pipeline:
                    PipelineTab()
                        .environmentObject(appState)
                case .documents:
                    DocumentsTab()
                        .environmentObject(appState)
                case .history:
                    HistoryTab()
                        .environmentObject(appState)
                }
            }
            .navigationTitle("Agent Crew")
            .task { await appState.setup() }
            .sheet(item: $consentGate.pending) { request in
                ConsentSheet(request: request, gate: consentGate)
            }
        }
    }
}

// MARK: - Models Tab

struct ModelsTab: View {
    private let profile = HardwareProfile.current()
    @State private var testTarget: DomainTestTarget?
    @State private var showsSearch = false
    @State private var showsDownloads = false

    /// Only show models that can actually run on this device.
    private var runnableText: [Model] { runnable(Model.textModels) }
    private var runnableVision: [Model] { runnable(Model.visionModels) }
    private var runnableOCR: [Model] { runnable(Model.specializedModels) }

    /// Models added from Hugging Face are managed under Downloads, not in the catalog sections.
    private func runnable(_ models: [Model]) -> [Model] {
        let added = Set(DownloadRecords.standard.models.map(\.id))
        return models.filter {
            !added.contains($0.id) && HardwareAnalyzer.assess($0, profile: profile).fitLevel.isRunnable
        }
    }

    private func color(for domain: Model.Domain) -> Color {
        switch domain {
        case .code:     .purple
        case .security: .red
        case .finance:  .mint
        case .medicine: .pink
        }
    }

    var body: some View {
        // Read so the sections are rebuilt when a download finishes or a model is deleted.
        let _ = ModelDownloadCenter.shared.revision
        NavigationStack {
            List {
                Section {
                    Button("Search Hugging Face", systemImage: "magnifyingglass") { showsSearch = true }
                    Button("Downloads", systemImage: "arrow.down.circle") { showsDownloads = true }
                } header: {
                    Text("Hugging Face")
                } footer: {
                    Text("Search shows what runs on this device. Downloaded models come first in the Text and Vision pickers.")
                }

                // Domain-specialized picks (code / security / finance / medicine).
                ForEach(Model.Domain.allCases) { domain in
                    let models = runnable(Model.models(in: domain))
                    if !models.isEmpty {
                        ModelSection(title: domain.displayName, icon: domain.systemImage, color: color(for: domain), models: models) { model in
                            testTarget = DomainTestTarget(model: model, domain: domain)
                        }
                    }
                }

                if !runnableOCR.isEmpty {
                    ModelSection(title: "Specialized OCR", icon: "doc.viewfinder", color: .orange, models: runnableOCR)
                }
                if !runnableVision.isEmpty {
                    ModelSection(title: "Vision",          icon: "eye",            color: .blue,   models: runnableVision)
                }
                if !runnableText.isEmpty {
                    ModelSection(title: "Text",            icon: "text.bubble",    color: .green,  models: runnableText)
                }

                // Device info
                Section {
                    LabeledContent("Device", value: profile.deviceName)
                        .font(.subheadline)
                    LabeledContent("Memory", value: String(format: "%.1f GB", profile.totalMemoryGB))
                        .font(.subheadline)
                    LabeledContent("Available") {
                        // Distinguish a real reading from the estimate. The OS reports "unknown"
                        // exactly when the process is at its jetsam limit, and every safety check
                        // treats that as pressure — so a rosy number here without the caveat would
                        // contradict what the engine is actually doing.
                        if HardwareProfile.availableMemoryBytes() != nil {
                            Text(String(format: "%.1f GB", profile.availableMemoryGB))
                        } else {
                            VStack(alignment: .trailing, spacing: 2) {
                                Text(String(format: "~%.1f GB", profile.availableMemoryGB))
                                Text("estimated — OS reported unknown")
                                    .font(.caption2).foregroundStyle(.orange)
                            }
                        }
                    }
                    .font(.subheadline)
                    LabeledContent("Compatible models", value: "\(runnableText.count + runnableVision.count + runnableOCR.count)")
                        .font(.subheadline)
                } header: {
                    Label("Device", systemImage: "iphone")
                } footer: {
                    Text("Only models compatible with this device are shown. Download uses parallel chunks for large files.")
                }
            }
            .navigationTitle("Models")
            .sheet(item: $testTarget) { target in
                DomainTestView(model: target.model, domain: target.domain)
            }
            .sheet(isPresented: $showsSearch) { HFSearchScreen() }
            .sheet(isPresented: $showsDownloads) { DownloadsScreen() }
        }
    }
}

#Preview { ContentView().environmentObject(AppState()) }
