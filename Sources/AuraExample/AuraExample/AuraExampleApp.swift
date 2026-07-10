import SwiftUI
import AuraCore

@main
struct AuraExampleApp: App {
    @StateObject private var appState = AppState()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
        }
        .defaultSize(width: 900, height: 700)
#if os(macOS)
        .windowResizability(.contentSize)
#endif
        .commands {
            AppCommands(appState: appState)
        }
        .onChange(of: scenePhase) { _, newPhase in
            switch newPhase {
            case .background:
                // §2.4 / Common Mistake #10: Release GPU resources when backgrounded
                Task { @MainActor in
                    ModelManager.shared.evictAll()
                }
            case .inactive:
                // Pause heavy inference if needed — no full eviction
                break
            case .active:
                break
            @unknown default:
                break
            }
        }

#if os(macOS)
        // Top-level Settings scene (⌘,) — hybrid policy, BYOK keys, download tokens.
        Settings {
            HybridSettingsView(isEmbedded: true)
        }
#endif
    }
}
