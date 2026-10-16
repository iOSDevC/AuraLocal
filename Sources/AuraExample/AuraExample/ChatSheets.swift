import SwiftUI
import AuraCore
import AuraUI

// MARK: - Conversations

/// Past conversations; picking one reopens it, swiping deletes it.
struct ChatConversationsSheet: View {
    @ObservedObject var vm: ChatViewModel
    let onSelect: (Conversation) -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                ForEach(vm.conversations) { conversation in
                    Button {
                        onSelect(conversation)
                        dismiss()
                    } label: {
                        VStack(alignment: .leading, spacing: 3) {
                            Text(conversation.title)
                                .font(.subheadline.weight(.medium))
                                .foregroundStyle(.primary)
                                .lineLimit(1)
                            Text("\(conversation.model.components(separatedBy: "/").last ?? conversation.model) · "
                                 + "\(conversation.turnCount) turns")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.vertical, 2)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .disabled(vm.isBusy)
                    .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                        // A disabled destructive swipe still slides the row away, so none is offered here.
                        if !vm.isReplying(in: conversation) {
                            Button("Delete", systemImage: "trash", role: .destructive) {
                                Task { await vm.deleteConversation(conversation) }
                            }
                        }
                    }
                    .listRowBackground(vm.activeConversation?.id == conversation.id
                                       ? Color.accentColor.opacity(0.08) : nil)
                }
            }
            .overlay {
                if vm.conversations.isEmpty {
                    ContentUnavailableView("No conversations", systemImage: "bubble.left",
                                           description: Text("Send a message to start one."))
                }
            }
            .navigationTitle("Conversations")
            .task { await vm.loadConversations() }
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
        #if os(macOS)
        .frame(minWidth: 420, minHeight: 480)
        #endif
    }
}

// MARK: - Model picker

/// Where the model sheet opens.
enum ChatModelRoute: Hashable {
    case catalog
    case huggingFace
}

/// Chat's model picker: the downloaded models with their capability badges, plus the catalog and a Hugging Face
/// search to download more. Downloads keep going after the sheet closes.
struct ChatModelSheet: View {
    let selectedModel: Model?
    let onSelect: (Model) -> Void
    @State private var path: [ChatModelRoute]
    @Environment(\.dismiss) private var dismiss

    init(selectedModel: Model?, start: ChatModelRoute?, onSelect: @escaping (Model) -> Void) {
        self.selectedModel = selectedModel
        self.onSelect = onSelect
        _path = State(initialValue: start.map { [$0] } ?? [])
    }

    var body: some View {
        // Read so the list is rebuilt when a download finishes or a model is deleted.
        let _ = ModelDownloadCenter.shared.revision
        let downloaded = ChatModels.downloaded()
        NavigationStack(path: $path) {
            List {
                Section {
                    NavigationLink(value: ChatModelRoute.catalog) {
                        Label("Browse the catalog", systemImage: "square.stack.3d.up")
                    }
                    NavigationLink(value: ChatModelRoute.huggingFace) {
                        Label("Find on Hugging Face", systemImage: "magnifyingglass")
                    }
                } footer: {
                    Text("Both list only models that run on this device. A finished download appears below.")
                }
                Section("On this device") {
                    ForEach(downloaded) { model in
                        row(for: model)
                    }
                    if downloaded.isEmpty {
                        Text("Nothing downloaded yet.").foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Chat model")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .navigationDestination(for: ChatModelRoute.self) { route in
                switch route {
                case .catalog: ChatCatalogList()
                case .huggingFace: ChatModelFinder()
                }
            }
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
        #if os(macOS)
        .frame(minWidth: 560, minHeight: 600)
        #endif
    }

    private func row(for model: Model) -> some View {
        let blocker = ModelBlocker.check(model)
        return Button {
            onSelect(model)
            dismiss()
        } label: {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.displayName)
                        .font(.body.weight(.medium))
                        .foregroundStyle(.primary)
                    CapabilityBadges(capabilities: model.capabilities)
                    Text(details(model, blocker: blocker))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
                if model == selectedModel {
                    Image(systemName: "checkmark")
                        .foregroundStyle(Color.accentColor)
                        .accessibilityHidden(true)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(blocker != nil)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(model.displayName). \(model.capabilities.summary). \(details(model, blocker: blocker))")
        .accessibilityAddTraits(model == selectedModel ? .isSelected : [])
    }

    /// Format and catalog size, written as the catalog rows write it.
    private func details(_ model: Model, blocker: ModelBlocker?) -> String {
        let megabytes = model.approximateSizeMB
        let size = megabytes >= 1000 ? String(format: "%.1f GB", Double(megabytes) / 1024)
            : megabytes > 0 ? "\(megabytes) MB" : "size unknown"
        let base = "\(model.format == .gguf ? "GGUF" : "MLX") · \(size)"
        return blocker.map { "\(base) · \($0.explanation)" } ?? base
    }
}

// MARK: - Catalog

/// AuraLocal's catalog of chat models that fit this device, with download, cancel and delete per row.
struct ChatCatalogList: View {
    @State private var testTarget: DomainTestTarget?
    @AppStorage("auraExample.showsUncensored") private var showsUncensored = false

    var body: some View {
        // Read so the sections are rebuilt when a download finishes or a model is deleted.
        let _ = ModelDownloadCenter.shared.revision
        let text = ChatModels.catalog(Model.textModels.filter { $0.domain == nil })
        let vision = ChatModels.catalog(Model.visionModels.filter { $0.domain == nil })
        let documents = ChatModels.catalog(Model.specializedModels.filter { $0.domain == nil })
        List {
            Section {
                Text("Downloads land where the backends load them from. Downloaded models appear in Chat's model "
                     + "list; Voice uses the text ones.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } footer: {
                #if targetEnvironment(simulator)
                Text("The Simulator lists GGUF models only: MLX needs a device.")
                #endif
            }
            if !text.isEmpty {
                ModelSection(title: "Text", icon: "text.bubble", color: .green, models: text)
            }
            if !vision.isEmpty {
                ModelSection(title: "Vision", icon: "eye", color: .blue, models: vision)
            }
            if !documents.isEmpty {
                ModelSection(title: "Documents (OCR)", icon: "doc.viewfinder", color: .orange, models: documents)
            }
            ForEach(Model.Domain.allCases) { domain in
                let models = ChatModels.catalog(Model.models(in: domain))
                if !models.isEmpty {
                    ModelSection(title: domain.displayName, icon: domain.systemImage,
                                 color: Self.color(for: domain), models: models) { model in
                        testTarget = DomainTestTarget(model: model, domain: domain)
                    }
                }
            }
            Section {
                Toggle("Show uncensored models", isOn: $showsUncensored)
            } footer: {
                Text("Abliterated and uncensored fine-tunes have their refusal training removed. AuraLocal leaves them "
                     + "out of its model lists unless an app asks for Model.uncensoredModels.")
            }
            if showsUncensored {
                let uncensored = ChatModels.catalog(Model.uncensoredModels)
                if !uncensored.isEmpty {
                    ModelSection(title: "Uncensored", icon: "exclamationmark.shield", color: .gray, models: uncensored)
                }
            }
        }
        .navigationTitle("Catalog")
        .sheet(item: $testTarget) { target in
            DomainTestView(model: target.model, domain: target.domain)
        }
    }

    private static func color(for domain: Model.Domain) -> Color {
        switch domain {
        case .code:     .purple
        case .security: .red
        case .finance:  .mint
        case .medicine: .pink
        }
    }
}

// MARK: - Hugging Face

/// Hugging Face search for chat models judged against this device: text generation, image + text and any-to-any
/// repos (plus undeclared ones), in MLX or GGUF.
struct ChatModelFinder: View {
    @State private var search = HFModelSearch(query: "qwen", devices: [DevicePreset.thisDevice()],
                                              compatibleOnly: true, pipelineFilter: .chat)
    @State private var selection: String?
    @State private var hasSearched = false

    var body: some View {
        @Bindable var downloads = ModelDownloadCenter.shared
        HFModelSearchList(search: search, selection: $selection)
            .navigationTitle("Hugging Face")
            .navigationDestination(item: $selection) { HFRepoDetailView(repoID: $0, search: search) }
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Menu("Options", systemImage: "ellipsis.circle") {
                        Toggle("Keep the screen awake while downloading", isOn: $downloads.keepsScreenAwake)
                    }
                }
            }
            .task {
                // Coming back from a repo's detail must not repeat the search.
                guard !hasSearched else { return }
                hasSearched = true
                search.search()
            }
    }
}
