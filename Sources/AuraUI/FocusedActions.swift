import SwiftUI

// MARK: - Focused Action Keys
//
// §5.1 FocusedValue: Reimplements the responder chain in SwiftUI.
// Defined in AuraUI so both library views and the host app can share them.
// The host app's Commands read these; views attach their concrete actions.

public struct NewConversationActionKey: FocusedValueKey {
    public typealias Value = () -> Void
}

public struct SendMessageActionKey: FocusedValueKey {
    public typealias Value = () -> Void
}

public struct ToggleSidebarActionKey: FocusedValueKey {
    public typealias Value = () -> Void
}

public extension FocusedValues {
    var newConversationAction: (() -> Void)? {
        get { self[NewConversationActionKey.self] }
        set { self[NewConversationActionKey.self] = newValue }
    }
    var sendMessageAction: (() -> Void)? {
        get { self[SendMessageActionKey.self] }
        set { self[SendMessageActionKey.self] = newValue }
    }
    var toggleSidebarAction: (() -> Void)? {
        get { self[ToggleSidebarActionKey.self] }
        set { self[ToggleSidebarActionKey.self] = newValue }
    }
}
