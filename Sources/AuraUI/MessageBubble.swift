import SwiftUI
import AuraCore

// MARK: - MessageBubble
//
// §7.1: Minimum touch target respected via padding.
// Added .accessibilityLabel for screen readers.

public struct MessageBubble: View {
    let message: ChatMessage

    public var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 48) }
            Text(message.text)
                .padding(12)
                .background(
                    message.role == .user
                    ? Color.blue.opacity(0.15)
                    : Color.secondaryGroupedBackground,
                    in: RoundedRectangle(cornerRadius: 16)
                )
                .font(.subheadline)
                .textSelection(.enabled)
                .accessibilityLabel("\(message.role == .user ? "You" : "Assistant"): \(message.text)")
            if message.role == .assistant { Spacer(minLength: 48) }
        }
    }
}

// MARK: - StreamingBubble

public struct StreamingBubble: View {
    let text: String

    public var body: some View {
        HStack {
            Text(text.isEmpty ? "…" : text)
                .padding(12)
                .background(Color.secondaryGroupedBackground, in: RoundedRectangle(cornerRadius: 16))
                .font(.subheadline)
                .opacity(text.isEmpty ? 0.4 : 1)
                .textSelection(.enabled)
                .accessibilityLabel(text.isEmpty ? "Generating response" : "Assistant is typing: \(text)")
            Spacer(minLength: 48)
        }
    }
}
