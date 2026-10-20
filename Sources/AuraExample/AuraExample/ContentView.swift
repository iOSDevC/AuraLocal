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
// Each section finds and downloads its own models; Models only manages what is on disk.

struct ContentView: View {
    @EnvironmentObject var appState: AppState
    @AppStorage("auraExample.selectedTab") private var storedTab = AppTab.chat
    /// Starts on the remembered tab, or on the one a `-auraExample.openTab <tab>` launch argument names. The
    /// argument has its own key: one on the stored key would win every read and pin the tab.
    @State private var selectedTab: AppTab
    @State private var imageState = ImageGenState()

    init() {
        let defaults = UserDefaults.standard
        let name = defaults.string(forKey: "auraExample.openTab") ?? defaults.string(forKey: "auraExample.selectedTab")
        let restored = name.flatMap(AppTab.init(rawValue:)) ?? .chat
        _selectedTab = State(initialValue: AppTab.shown.contains(restored) ? restored : .chat)
    }

    var body: some View {
        TabView(selection: $selectedTab) {
            // iPhone shows the first four tabs in the bar; the rest sit under More.
            Tab("Chat", systemImage: "bubble.left.and.text.bubble.right", value: AppTab.chat) {
                ChatTab()
            }

            Tab("Voice", systemImage: "mic.fill", value: AppTab.voice) {
                VoiceTab()
            }

            Tab("Image", systemImage: "photo.on.rectangle.angled", value: AppTab.image) {
                ImageGenTab(state: imageState)
            }

            Tab("Library", systemImage: "doc.text.magnifyingglass", value: AppTab.library) {
                DocsTab()
            }

            // MARK: - Tools section
            TabSection("Tools") {
                Tab("Models", systemImage: "internaldrive", value: AppTab.models) {
                    ModelsTab(imageState: imageState)
                }

                Tab("Hybrid", systemImage: "cloud.bolt", value: AppTab.hybrid) {
                    HybridTab()
                }

                Tab("ML", systemImage: "brain", value: AppTab.ml) {
                    MLToolsTab()
                }

                // AgentCrew needs Apple Intelligence (OS 26); its long pipeline suits a Mac, not a phone.
                #if !os(iOS)
                if #available(macOS 26, visionOS 26, *) {
                    Tab("Agents", systemImage: "cpu", value: AppTab.agents) {
                        AgentCrewInlineView()
                            .environmentObject(appState)
                    }
                }
                #endif
            }
        }
        .tabViewStyle(.sidebarAdaptable)
        #if os(iOS)
        .background(KeyboardDismissOnTap())
        #endif
        .onChange(of: selectedTab) { _, tab in storedTab = tab }
    }
}

enum AppTab: String {
    case chat, voice, image, library, models, hybrid, ml, agents

    /// Tabs this platform shows; a remembered tab outside it falls back to Chat.
    static var shown: Set<AppTab> {
        #if os(iOS)
        [.chat, .voice, .image, .library, .models, .hybrid, .ml]
        #else
        [.chat, .voice, .image, .library, .models, .hybrid, .ml, .agents]
        #endif
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

#Preview { ContentView().environmentObject(AppState()) }
