import SwiftUI
import AuraCore

// MARK: - Chat View
public struct ChatView: View {
    @ObservedObject var vm: TextChatViewModel
    @Binding var selectedModel: Model
    @State private var prompt: String = ""
    @FocusState private var promptFocused: Bool

    public var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(vm.messages) { msg in
                            MessageBubble(message: msg).id(msg.id)
                        }
                        if vm.isStreaming {
                            StreamingBubble(text: vm.streamingText).id("streaming")
                        }
                    }
                    .padding()
                }
                // §Fix #10: Replace onTapGesture keyboard dismiss with proper API
                .scrollDismissesKeyboard(.interactively)
                .onChange(of: vm.streamingText) { _, _ in
                    withAnimation { proxy.scrollTo("streaming", anchor: .bottom) }
                }
                .onChange(of: vm.messages.count) { _, _ in
                    withAnimation { proxy.scrollTo(vm.messages.last?.id, anchor: .bottom) }
                }
            }

            Divider()

            if !vm.progress.isEmpty {
                HStack(spacing: 6) {
                    ProgressView().scaleEffect(0.7)
                    Text(vm.progress).font(.caption).foregroundStyle(.secondary)
                }
                .padding(.horizontal)
                .padding(.top, 6)
            }

            inputBar
        }
        .background(Color.groupedBackground)
        // §5.1: Wire send action for Cmd+Return via FocusedValue
        .focusedValue(\.sendMessageAction) {
            sendCurrentPrompt()
        }
    }

    // MARK: - Input Bar

    private var inputBar: some View {
        HStack(spacing: 10) {
            TextField("Ask something...", text: $prompt, axis: .vertical)
                .lineLimit(1...4)
                .padding(10)
                .background(Color.tertiaryGroupedBackground, in: RoundedRectangle(cornerRadius: 12))
                .focused($promptFocused)
                .onSubmit { sendCurrentPrompt() }

            sendButton
        }
        .padding(12)
        .background(Color.secondaryGroupedBackground)
    }

    // MARK: - Send Button
    //
    // §7.1: Minimum 60pt for visionOS; 44pt for iOS.
    // §4.2: .hoverEffect() for iPadOS pointer magnetism and visionOS gaze highlight.

    private var sendButton: some View {
        Button {
            sendCurrentPrompt()
        } label: {
            Image(systemName: vm.isStreaming ? "stop.circle.fill" : "arrow.up.circle.fill")
                .font(.system(size: 28))
                .foregroundStyle(vm.isStreaming ? .red : .blue)
                .frame(minWidth: 44, minHeight: 44)
                .contentShape(RoundedRectangle(cornerRadius: 22))
        }
        .hoverEffect()
        .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !vm.isStreaming)
        .accessibilityLabel(vm.isStreaming ? "Stop generating" : "Send message")
        .accessibilityAction(named: "Send message") { sendCurrentPrompt() }
#if os(macOS)
        .keyboardShortcut(.return, modifiers: .command)
#endif
    }

    // MARK: - Send Logic

    private func sendCurrentPrompt() {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !vm.isStreaming else { return }
        prompt = ""
        promptFocused = false
        Task { await vm.send(text, model: selectedModel) }
    }
}
