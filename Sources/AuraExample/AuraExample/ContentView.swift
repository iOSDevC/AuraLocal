import SwiftUI
import PhotosUI
import AuraCore
import AuraUI
import AuraVoice
import AuraDocs
import AuraAppleIntelligence

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
                TextChatTab()
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
        }
    }
}

// MARK: - Models Tab

struct ModelsTab: View {
    var body: some View {
        NavigationStack {
            List {
                ModelSection(title: "Text",            icon: "text.bubble",    color: .green,  models: Model.textModels)
                ModelSection(title: "Vision",          icon: "eye",            color: .blue,   models: Model.visionModels)
                ModelSection(title: "Specialized OCR", icon: "doc.viewfinder", color: .orange, models: Model.specializedModels)
            }
            .navigationTitle("Models")
        }
    }
}

#Preview { ContentView().environmentObject(AppState()) }
