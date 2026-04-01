import SwiftUI
import AVFoundation
import AuraCore

// MARK: - VoiceButton

/// A ready-to-use SwiftUI button that drives a VoiceSession.
///
/// Drop it anywhere in your view hierarchy:
/// ```swift
/// VoiceButton(llm: llm)
/// ```
///
/// Or with a shared session for full control:
/// ```swift
/// @StateObject var session = VoiceSession(llm: llm)
///
/// VoiceButton(session: session)
///
/// Text(session.response)
/// ```
public struct VoiceButton: View {
    @StateObject private var ownedSession: VoiceSession
    @ObservedObject private var session: VoiceSession

    @State private var permissionDenied = false
    @State private var pulse = false

    // MARK: - Init

    /// Convenience init — creates and owns its VoiceSession internally.
    public init(
        llm: AuraLocal,
        conversationID: UUID? = nil,
        config: VoiceSession.Config = .init()
    ) {
        let s = VoiceSession(llm: llm, conversationID: conversationID, config: config)
        _ownedSession = StateObject(wrappedValue: s)
        _session = ObservedObject(wrappedValue: s)
    }

    /// Init with an externally-owned session (for sharing state with parent).
    public init(session: VoiceSession) {
        let placeholder = session   // reuse same instance
        _ownedSession = StateObject(wrappedValue: placeholder)
        _session = ObservedObject(wrappedValue: session)
    }

    // MARK: - Body

    public var body: some View {
        Button {
            handleTap()
        } label: {
            ZStack {
                // Pulse ring when listening
                if case .listening = session.state {
                    Circle()
                        .stroke(Color.red.opacity(0.3), lineWidth: 3)
                        .scaleEffect(pulse ? 1.6 : 1.0)
                        .opacity(pulse ? 0 : 0.6)
                        .animation(.easeOut(duration: 1).repeatForever(autoreverses: false), value: pulse)
                }

                Circle()
                    .fill(buttonColor.opacity(0.15))
                    .frame(width: 64, height: 64)

                Image(systemName: iconName)
                    .font(.system(size: 26, weight: .medium))
                    .foregroundStyle(buttonColor)
                    .symbolEffect(.bounce, value: session.state == .idle)
            }
        }
        .buttonStyle(.plain)
        .disabled(permissionDenied)
        .onAppear {
            Task {
                let granted = await session.requestPermissions()
                permissionDenied = !granted
            }
        }
        .onChange(of: isListening) { _, listening in
            pulse = listening
        }
        .alert("Microphone Access Required", isPresented: $permissionDenied) {
            Button("Open Settings") { openSettings() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Please allow microphone and speech recognition access in Settings.")
        }
    }

    // MARK: - Computed

    private var isListening: Bool {
        if case .listening = session.state { return true }
        return false
    }

    private var iconName: String {
        switch session.state {
        case .idle:                return "mic"
        case .listening:           return "mic.fill"
        case .thinking:            return "ellipsis.bubble"
        case .speaking:            return "speaker.wave.2.fill"
        case .error:               return "exclamationmark.triangle"
        }
    }

    private var buttonColor: Color {
        switch session.state {
        case .idle:                return .primary
        case .listening:           return .red
        case .thinking:            return .blue
        case .speaking:            return .green
        case .error:               return .orange
        }
    }

    // MARK: - Actions

    private func handleTap() {
        switch session.state {
        case .idle:
            Task { try? await session.startListening() }
        case .listening:
            Task { await session.stopListening() }
        case .thinking, .speaking:
            session.interrupt()
        case .error:
            session.cancel()
        }
    }

    private func openSettings() {
        #if os(iOS)
        if let url = URL(string: UIApplication.openSettingsURLString) {
            UIApplication.shared.open(url)
        }
        #elseif os(macOS)
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
            NSWorkspace.shared.open(url)
        }
        #endif
    }
}

// MARK: - VoiceChatView

/// A full voice chat interface — transcript, response, and VoiceButton combined.
///
/// ```swift
/// VoiceChatView(llm: llm)
/// ```
public struct VoiceChatView: View {
    @StateObject private var session: VoiceSession
    private let llm: AuraLocal

    public init(llm: AuraLocal, conversationID: UUID? = nil, config: VoiceSession.Config = .init()) {
        self.llm = llm
        _session = StateObject(wrappedValue: VoiceSession(
            llm: llm,
            conversationID: conversationID,
            config: config
        ))
    }

    public var body: some View {
        VStack(spacing: 24) {
            Spacer()

            // Status label
            statusLabel

            // Transcript bubble
            if !session.transcript.isEmpty {
                ChatBubble(text: session.transcript, role: .user)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }

            // Response bubble
            if !session.response.isEmpty {
                ChatBubble(text: session.response, role: .assistant)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }

            Spacer()

            VoiceButton(session: session)
                .padding(.bottom, 32)
        }
        .animation(.spring(duration: 0.35), value: session.transcript)
        .animation(.spring(duration: 0.35), value: session.response)
        .padding()
    }

    @ViewBuilder
    private var statusLabel: some View {
        switch session.state {
        case .idle:
            Text("Tap to speak")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        case .listening:
            Label("Listening…", systemImage: "waveform")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.red)
        case .thinking(let partial) where partial.isEmpty:
            Label("Thinking…", systemImage: "ellipsis.bubble")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.blue)
        case .thinking:
            Label("Generating…", systemImage: "ellipsis.bubble.fill")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.blue)
        case .speaking:
            Label("Speaking…", systemImage: "speaker.wave.2")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.green)
        case .error(let msg):
            Label(msg, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .multilineTextAlignment(.center)
        }
    }
}

// MARK: - ChatBubble

private struct ChatBubble: View {
    enum Role { case user, assistant }
    let text: String
    let role: Role

    var body: some View {
        HStack {
            if role == .user { Spacer(minLength: 48) }
            Text(text)
                .padding(12)
                .background(
                    role == .user ? Color.blue.opacity(0.12) : Color.secondary.opacity(0.1),
                    in: RoundedRectangle(cornerRadius: 16)
                )
                .font(.body)
            if role == .assistant { Spacer(minLength: 48) }
        }
    }
}
