import SwiftUI
import AuraAppleIntelligence

// MARK: - AgentCrewMainView (Legacy)
//
// Preserved for backward compatibility.
// The primary integration now goes through AgentCrewInlineView in ContentView.
// §Fix #24: Wraps content in AIAvailabilityGate to show clear messaging
// when Apple Intelligence is unavailable.

@available(iOS 26, macOS 26, *)
struct AgentCrewMainView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        AIAvailabilityGate {
            AgentCrewInlineView()
                .environmentObject(appState)
        }
    }
}
