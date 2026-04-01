import SwiftUI
import AuraCore
import AuraUI

// MARK: - AppCommands
//
// §5.1: Global Menu Bar commands for macOS.
// FocusedValue keys are defined in AuraUI/FocusedActions.swift
// so that both AuraUI views and this host app can share them.
// Uses @FocusedValue to dispatch to the active window's view.

struct AppCommands: Commands {
    let appState: AppState
    @FocusedValue(\.newConversationAction) var newConversation
    @FocusedValue(\.sendMessageAction)     var sendMessage
    @FocusedValue(\.toggleSidebarAction)   var toggleSidebar

    var body: some Commands {
        // Replace default New Item
        CommandGroup(replacing: .newItem) {
            Button("New Conversation") {
                newConversation?()
            }
            .keyboardShortcut("n", modifiers: .command)
            .disabled(newConversation == nil)
        }

        CommandMenu("Chat") {
            Button("Send Message") {
                sendMessage?()
            }
            .keyboardShortcut(.return, modifiers: .command)
            .disabled(sendMessage == nil)

            Divider()

            Button("Toggle Conversations Sidebar") {
                toggleSidebar?()
            }
            .keyboardShortcut("l", modifiers: .command)
            .disabled(toggleSidebar == nil)
        }
    }
}
