import SwiftUI
import PhotosUI
import TonamoxCore
import TonamoxUI
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - ChatTab
//
// One chat for every downloaded model: text with history, an image attachment when the model takes images, and
// document extraction when it has a document prompt. Capabilities come from `Model.capabilities`. Models are
// picked from what is on disk; the model sheet's catalog and Hugging Face search download more.

struct ChatTab: View {
    @StateObject private var vm = ChatViewModel()
    @AppStorage("auraExample.chatModelID") private var storedModelID = ""
    @State private var selectedModel: Model?
    @State private var prompt = ""
    @State private var attachment: PlatformImage?
    @State private var pickerItem: PhotosPickerItem?
    @State private var sheet: ChatSheet?
    /// Downloaded model ids when the model sheet opened, to pick up what it added.
    @State private var modelsBeforeSheet: Set<String>?
    @FocusState private var promptFocused: Bool

    var body: some View {
        // Read so the picker is rebuilt when a download finishes or a model is deleted.
        let revision = ModelDownloadCenter.shared.revision
        NavigationStack {
            Group {
                if let selectedModel {
                    conversation(with: selectedModel)
                } else {
                    noModel
                }
            }
            .navigationTitle(vm.activeConversation?.title ?? "Chat")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar { toolbar }
            .sheet(item: $sheet, onDismiss: adoptNewDownload) { sheet in
                switch sheet {
                case .conversations: ChatConversationsSheet(vm: vm, onSelect: restore)
                case .models(let start):
                    ChatModelSheet(selectedModel: selectedModel, start: start) { model in
                        // An explicit pick wins over adopting a fresh download on dismiss.
                        modelsBeforeSheet = nil
                        select(model)
                    }
                case .domainTest(let model, let domain): DomainTestView(model: model, domain: domain)
                }
            }
            .onAppear { if selectedModel == nil { selectedModel = ChatModels.preferred(id: storedModelID) } }
            .onChange(of: revision) { resetSelectionIfRemoved() }
            .onChange(of: pickerItem) { _, item in loadAttachment(item) }
            .focusedValue(\.newConversationAction) { startNewConversation() }
            .focusedValue(\.toggleSidebarAction) { sheet = .conversations }
            .focusedValue(\.sendMessageAction) { submit() }
        }
    }

    // MARK: Layout

    private func conversation(with model: Model) -> some View {
        VStack(spacing: 0) {
            modelHeader(model)
            Divider()
            ChatTranscript(entries: vm.entries, streamingText: vm.streamingText, isBusy: vm.isBusy,
                           emptyHint: emptyHint(for: model.capabilities))
            Divider()
            if !vm.progress.isEmpty {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text(vm.progress).font(.caption).foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal)
                .padding(.top, 6)
            }
            composer(model)
        }
        .background(ChatColors.background)
    }

    private var noModel: some View {
        ContentUnavailableView {
            Label("No chat model on this device", systemImage: "cpu")
        } description: {
            Text("Download one from the catalog or from Hugging Face. Models that take images can also describe "
                 + "photos and read documents.")
        } actions: {
            Button("Browse the catalog", systemImage: "square.stack.3d.up") { present(.models(start: .catalog)) }
                .buttonStyle(.borderedProminent)
            Button("Find on Hugging Face", systemImage: "magnifyingglass") {
                present(.models(start: .huggingFace))
            }
        }
    }

    /// The selected model and what it can do; tapping it opens the model sheet.
    private func modelHeader(_ model: Model) -> some View {
        Button {
            present(.models(start: nil))
        } label: {
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 4) {
                        Text(model.displayName)
                            .font(.subheadline.weight(.semibold))
                            .lineLimit(1)
                        Image(systemName: "chevron.up.chevron.down")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    CapabilityBadges(capabilities: model.capabilities)
                }
                Spacer(minLength: 0)
            }
            .contentShape(Rectangle())
            .padding(.horizontal)
            .padding(.vertical, 8)
        }
        .buttonStyle(.plain)
        .disabled(vm.isBusy)
        .accessibilityLabel("Model: \(model.displayName). \(model.capabilities.summary)")
        .accessibilityHint("Opens the model list; picking another model starts a new conversation")
    }

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        ToolbarItem(placement: .navigation) {
            Button("Conversations", systemImage: "sidebar.left") { sheet = .conversations }
        }
        if let model = selectedModel, let domain = model.domain {
            ToolbarItem(placement: .primaryAction) {
                Button("\(domain.displayName) test", systemImage: domain.systemImage) {
                    sheet = .domainTest(model, domain)
                }
                .disabled(vm.isBusy)
            }
        }
        ToolbarItem(placement: .primaryAction) {
            Button("New conversation", systemImage: "square.and.pencil") { startNewConversation() }
                .disabled(vm.isBusy || selectedModel == nil)
        }
    }

    // MARK: Composer

    private func composer(_ model: Model) -> some View {
        let caps = model.capabilities
        return VStack(alignment: .leading, spacing: 8) {
            if let attachment {
                AttachmentPreview(image: attachment) { self.attachment = nil; pickerItem = nil }
            }
            HStack(alignment: .bottom, spacing: 10) {
                PhotosPicker(selection: $pickerItem, matching: .images) {
                    Image(systemName: "photo.badge.plus")
                        .font(.title3)
                        .frame(minWidth: 44, minHeight: 44)
                }
                .disabled(!caps.contains(.vision) || vm.isBusy)
                .accessibilityLabel("Attach an image")
                .accessibilityHint(caps.contains(.vision) ? "Adds a photo to the next message"
                                   : "\(model.displayName) does not take images")

                if caps.contains(.documentExtraction) {
                    Button {
                        extract(model)
                    } label: {
                        Image(systemName: "doc.text.viewfinder")
                            .font(.title3)
                            .frame(minWidth: 44, minHeight: 44)
                    }
                    .disabled(attachment == nil || vm.isBusy)
                    .accessibilityLabel("Extract document")
                    .accessibilityHint("Reads the attached image with the model's document prompt")
                }

                TextField(placeholder(for: caps), text: $prompt, axis: .vertical)
                    .lineLimit(1...4)
                    .padding(10)
                    .background(ChatColors.field, in: RoundedRectangle(cornerRadius: 12))
                    .focused($promptFocused)
                    .onSubmit { submit() }

                sendButton(caps)
            }
        }
        .padding(12)
        .background(ChatColors.bar)
    }

    /// What this model can be asked, before the first message.
    private func emptyHint(for caps: ModelCapabilities) -> String {
        if caps.contains(.documentExtraction) {
            return "Attach a photo of a receipt or document, then tap the document button to extract it."
        }
        if caps.contains(.vision) {
            return "Ask anything, or attach a photo and ask about it."
        }
        return "Ask anything. To ask about photos, pick a model with the Vision badge."
    }

    private func placeholder(for caps: ModelCapabilities) -> String {
        if attachment != nil { return "Ask about the image…" }
        return caps.contains(.text) ? "Ask something…" : "Attach an image first"
    }

    private func sendButton(_ caps: ModelCapabilities) -> some View {
        Button {
            if vm.isBusy { vm.stop() } else { submit() }
        } label: {
            Image(systemName: vm.isBusy ? "stop.circle.fill" : "arrow.up.circle.fill")
                .font(.system(size: 28))
                .foregroundStyle(vm.isBusy ? Color.red : Color.accentColor)
                .frame(minWidth: 44, minHeight: 44)
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .disabled(!vm.isBusy && !canSend(caps))
        .accessibilityLabel(vm.isBusy ? "Stop generating" : "Send message")
        #if os(macOS)
        .keyboardShortcut(.return, modifiers: .command)
        #endif
    }

    private func canSend(_ caps: ModelCapabilities) -> Bool {
        if attachment != nil { return caps.contains(.vision) }
        return caps.contains(.text) && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    // MARK: Actions

    private func submit() {
        guard let model = selectedModel, !vm.isBusy, canSend(model.capabilities) else { return }
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        prompt = ""
        promptFocused = false
        if let image = attachment {
            clearAttachment()
            vm.send(text, image: image, model: model)
        } else {
            vm.send(text, model: model)
        }
    }

    private func extract(_ model: Model) {
        guard let image = attachment, !vm.isBusy else { return }
        clearAttachment()
        vm.extractDocument(image, model: model)
    }

    private func clearAttachment() {
        attachment = nil
        pickerItem = nil
    }

    private func loadAttachment(_ item: PhotosPickerItem?) {
        guard let item else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self) {
                attachment = PlatformImage(data: data)
            }
        }
    }

    /// Picking a model starts a new conversation, as the conversation records its model.
    private func select(_ model: Model) {
        guard model != selectedModel else { return }
        selectedModel = model
        storedModelID = model.id
        if !model.capabilities.contains(.vision) { clearAttachment() }
        Task { await vm.newConversation(model: model) }
    }

    private func startNewConversation() {
        guard let model = selectedModel else { return }
        Task { await vm.newConversation(model: model) }
    }

    /// Opening a past conversation switches to its model when that is on disk; the store keeps its repo id.
    private func restore(_ conversation: Conversation) {
        Task {
            await vm.selectConversation(conversation)
            guard selectedModel?.rawValue != conversation.model else { return }
            if let match = ChatModels.downloaded().first(where: {
                $0.rawValue == conversation.model && ModelBlocker.check($0) == nil
            }) {
                selectedModel = match
                storedModelID = match.id
                if !match.capabilities.contains(.vision) { clearAttachment() }
            }
        }
    }

    private func present(_ next: ChatSheet) {
        modelsBeforeSheet = Set(ChatModels.downloaded().map(\.id))
        sheet = next
    }

    /// A model downloaded while the model sheet was open becomes the chat model.
    private func adoptNewDownload() {
        guard let before = modelsBeforeSheet else { return }
        modelsBeforeSheet = nil
        let added = ChatModels.downloaded().filter { !before.contains($0.id) && ModelBlocker.check($0) == nil }
        if let model = added.first { select(model) }
    }

    /// A deleted model leaves the registry or the disk; the chat moves to another downloaded model.
    private func resetSelectionIfRemoved() {
        if let current = selectedModel, ModelRegistry.shared.model(id: current.id) != nil, current.isDownloaded {
            return
        }
        selectedModel = ChatModels.preferred(id: storedModelID)
        if selectedModel?.capabilities.contains(.vision) != true { clearAttachment() }
    }
}

// MARK: - Sheets

enum ChatSheet: Identifiable {
    case conversations
    case models(start: ChatModelRoute?)
    case domainTest(Model, Model.Domain)

    var id: String {
        switch self {
        case .conversations: "conversations"
        case .models: "models"
        case .domainTest(let model, _): "domainTest-\(model.id)"
        }
    }
}

// MARK: - Transcript

private struct ChatTranscript: View {
    let entries: [ChatEntry]
    let streamingText: String
    let isBusy: Bool
    let emptyHint: String

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(entries) { ChatBubble(entry: $0).id($0.id) }
                    if isBusy {
                        StreamingChatBubble(text: streamingText).id("streaming")
                    }
                }
                .padding()
            }
            .scrollDismissesKeyboard(.interactively)
            .overlay {
                if entries.isEmpty && !isBusy {
                    Text(emptyHint)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(32)
                }
            }
            .onChange(of: streamingText) { _, _ in
                withAnimation { proxy.scrollTo("streaming", anchor: .bottom) }
            }
            .onChange(of: entries.count) { _, _ in
                withAnimation { proxy.scrollTo(entries.last?.id, anchor: .bottom) }
            }
        }
    }
}

private struct ChatBubble: View {
    let entry: ChatEntry

    var body: some View {
        if entry.isNotice {
            Text(entry.text)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity)
        } else {
            bubble
        }
    }

    private var bubble: some View {
        HStack {
            if entry.role == .user { Spacer(minLength: 48) }
            VStack(alignment: .leading, spacing: 8) {
                if let image = entry.image {
                    Image(platformImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: 220, maxHeight: 220)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                        .accessibilityLabel("Attached image")
                }
                if entry.isError {
                    Label(entry.text, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                } else {
                    Text(entry.text)
                }
            }
            .font(.subheadline)
            .textSelection(.enabled)
            .padding(12)
            .background(entry.role == .user ? Color.accentColor.opacity(0.15) : ChatColors.bubble,
                        in: RoundedRectangle(cornerRadius: 16))
            .accessibilityElement(children: .combine)
            .accessibilityLabel("\(entry.role == .user ? "You" : "Assistant"): \(entry.text)")
            if entry.role != .user { Spacer(minLength: 48) }
        }
    }
}

private struct StreamingChatBubble: View {
    let text: String

    var body: some View {
        HStack {
            Text(text.isEmpty ? "…" : text)
                .font(.subheadline)
                .textSelection(.enabled)
                .padding(12)
                .background(ChatColors.bubble, in: RoundedRectangle(cornerRadius: 16))
                .opacity(text.isEmpty ? 0.4 : 1)
                .accessibilityLabel(text.isEmpty ? "Generating response" : "Assistant is typing: \(text)")
            Spacer(minLength: 48)
        }
    }
}

private struct AttachmentPreview: View {
    let image: PlatformImage
    let onRemove: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(platformImage: image)
                .resizable()
                .scaledToFill()
                .frame(width: 64, height: 64)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .accessibilityLabel("Attached image")
            Button("Remove image", systemImage: "xmark.circle.fill", action: onRemove)
                .labelStyle(.iconOnly)
                .foregroundStyle(.secondary)
                .buttonStyle(.plain)
            Spacer()
        }
    }
}

// MARK: - Colors

enum ChatColors {
    #if os(iOS)
    static let background = Color(.systemGroupedBackground)
    static let bubble = Color(.secondarySystemGroupedBackground)
    static let bar = Color(.secondarySystemGroupedBackground)
    static let field = Color(.tertiarySystemGroupedBackground)
    #else
    static let background = Color(nsColor: .windowBackgroundColor)
    static let bubble = Color(nsColor: .controlBackgroundColor)
    static let bar = Color(nsColor: .controlBackgroundColor)
    static let field = Color(nsColor: .textBackgroundColor)
    #endif
}
