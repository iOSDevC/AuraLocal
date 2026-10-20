import SwiftUI
import TonamoxCore
import TonamoxUI
import TonamoxDocs
import TonamoxImageGen

// MARK: - ModelsTab
//
// Storage only: every model file on disk, finished or paused, with its size, the section that uses it and Delete.
// Searching and downloading live in the sections (Chat, Image, Library). Locations come from the TonamoxCore,
// TonamoxImageGen and TonamoxDocs APIs, never from paths written here.

struct ModelsTab: View {
    let imageState: ImageGenState
    /// Library and Agents load their models through `ModelManager`; a load reads the files Delete removes.
    @ObservedObject private var loader = ModelManager.shared
    @State private var items: [StoredModel] = []
    @State private var freeBytes: Int64?
    @State private var notice: String?
    @State private var pendingDelete: StoredModel?

    private let profile = HardwareProfile.current()

    var body: some View {
        let refreshKey = StorageRefreshKey(revision: ModelDownloadCenter.shared.revision,
                                           activeLoads: loader.states.values.filter(\.isActive).count,
                                           imageDownloading: imageState.isDownloading,
                                           imageDownloaded: imageState.isDownloaded,
                                           imageModels: imageState.availableModels.count)
        NavigationStack {
            List {
                summarySection
                if let notice {
                    Section { Text(notice).font(.callout).foregroundStyle(.secondary) }
                }
                if items.isEmpty {
                    Section {
                        Label("Nothing downloaded yet.", systemImage: "internaldrive")
                            .foregroundStyle(.secondary)
                    }
                }
                ForEach(StoredModel.Usage.allCases) { usage in
                    let rows = items.filter { $0.usedIn == usage }
                    if !rows.isEmpty {
                        Section {
                            ForEach(rows) { row(for: $0) }
                        } header: {
                            Label(usage.title, systemImage: usage.systemImage)
                        } footer: {
                            if let footer = usage.footer(paused: rows.contains { !$0.isComplete && !isBusy($0) }) {
                                Text(footer)
                            }
                        }
                    }
                }
                deviceSection
            }
            .navigationTitle("Models")
            .refreshable { refresh() }
            .onAppear { refresh() }
            .onChange(of: refreshKey) { refresh() }
            .confirmationDialog(pendingDelete.map { "Delete \($0.title)?" } ?? "", isPresented: isConfirming,
                                titleVisibility: .visible, presenting: pendingDelete) { item in
                Button("Delete", role: .destructive) { Task { await delete(item) } }
            } message: { item in
                Text(item.deleteMessage)
            }
        }
    }

    // MARK: Sections

    private var summarySection: some View {
        Section {
            LabeledContent("Used by models", value: StoredModel.bytesText(items.map(\.sizeBytes).reduce(0, +)))
            LabeledContent("Free on this device", value: freeBytes.map { StoredModel.bytesText($0) } ?? "Unknown")
        } footer: {
            Text("Download chat models in Chat (Browse the catalog, Find on Hugging Face), image models in Image "
                 + "(Find on Hugging Face) and the Library's embedding model with Import embedding model.")
        }
    }

    private func row(for item: StoredModel) -> some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(item.title)
                    .font(.subheadline.weight(.medium))
                    .lineLimit(2)
                    .truncationMode(.middle)
                Text(item.detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if let capabilities = item.capabilities {
                    CapabilityBadges(capabilities: capabilities)
                }
                HStack(spacing: 8) {
                    status(of: item)
                    Text(StoredModel.bytesText(item.sizeBytes)).monospacedDigit()
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            Button("Delete \(item.title)", systemImage: "trash", role: .destructive) { pendingDelete = item }
                .labelStyle(.iconOnly)
                .buttonStyle(.borderless)
                .foregroundStyle(.red)
                .disabled(!item.isDeletable || isBusy(item))
                .frame(minWidth: 44, minHeight: 44)
        }
        .accessibilityElement(children: .contain)
        .swipeActions {
            Button("Delete", systemImage: "trash", role: .destructive) { pendingDelete = item }
                .disabled(!item.isDeletable || isBusy(item))
        }
    }

    @ViewBuilder
    private func status(of item: StoredModel) -> some View {
        let busy = isBusy(item)
        if item.isComplete {
            Label("Complete", systemImage: "checkmark.circle")
            if busy { Label("In use", systemImage: "bolt.circle") }
        } else if busy {
            Label("Downloading", systemImage: "arrow.down.circle.dotted")
        } else {
            Label("Paused", systemImage: "pause.circle")
        }
    }

    private var deviceSection: some View {
        Section {
            LabeledContent("Device", value: profile.deviceName)
            LabeledContent("Memory", value: String(format: "%.1f GB", profile.totalMemoryGB))
            LabeledContent("Available") {
                // A missing reading means the process is at its jetsam limit, which the engine treats as pressure.
                if let available = HardwareProfile.availableMemoryBytes() {
                    Text(String(format: "%.1f GB", Double(available) / 1_073_741_824))
                } else {
                    VStack(alignment: .trailing, spacing: 2) {
                        Text(String(format: "~%.1f GB", profile.availableMemoryGB))
                        Text("estimated — OS reported unknown")
                            .font(.caption2).foregroundStyle(.orange)
                    }
                }
            }
            #if os(iOS)
            // What fit verdicts are judged against on iOS: it does not shrink while a model is loaded.
            if let limit = HardwareProfile.memoryLimitBytes() {
                LabeledContent("App memory limit", value: String(format: "%.1f GB", Double(limit) / 1_073_741_824))
            }
            #endif
        } header: {
            Label("Device", systemImage: "iphone")
        }
        .font(.subheadline)
    }

    private var isConfirming: Binding<Bool> {
        Binding { pendingDelete != nil } set: { if !$0 { pendingDelete = nil } }
    }

    // MARK: Disk

    /// A download writing the file, a load reading it, or an image run using the model.
    private func isBusy(_ item: StoredModel) -> Bool {
        switch item.origin {
        case .chat(let model): model.map { ModelDownloadCenter.shared.isDownloading($0) || isLoading($0) } ?? false
        case .image(let model): imageState.isBusy(with: model)
        case .embeddings: false
        }
    }

    /// Matched by file: the load may use the catalog entry while the row shows the search record, or the reverse.
    private func isLoading(_ model: Model) -> Bool {
        let file = DownloadRecords.key(for: model)
        return loader.states.contains { $0.value.isActive && DownloadRecords.key(for: $0.key) == file }
    }

    private func refresh() {
        freeBytes = ModelDownloads.availableDiskBytes()
        items = StoredModel.scan()
    }

    private func delete(_ item: StoredModel) async {
        do {
            switch item.origin {
            case .chat(let model):
                guard let model else { return }
                try ModelDownloadCenter.shared.delete(model)
            case .image(let model):
                try await imageState.delete(model)
            case .embeddings(let folder):
                try FileManager.default.removeItem(at: folder)
            }
            notice = "Deleted \(item.title), freed \(StoredModel.bytesText(item.sizeBytes))."
        } catch {
            notice = "Could not delete \(item.title): \(error.localizedDescription)"
        }
        refresh()
    }
}

/// What the storage list reloads on: downloads finishing or deletes in Chat and Image, and loads that download.
private struct StorageRefreshKey: Equatable {
    let revision: Int
    let activeLoads: Int
    let imageDownloading: Bool
    let imageDownloaded: Bool
    let imageModels: Int
}

// MARK: - StoredModel

/// One model's files on disk.
struct StoredModel: Identifiable {
    enum Source {
        /// `nil` when nothing on disk says which model the file is; it can't be deleted from here.
        case chat(Model?)
        case image(CoreMLDiffusionModel)
        case embeddings(URL)
    }

    /// The section that uses the files.
    enum Usage: CaseIterable, Identifiable {
        case chat, image, library

        var id: Self { self }

        var title: String {
            switch self {
            case .chat: "Chat & Voice"
            case .image: "Image"
            case .library: "Library"
            }
        }

        var systemImage: String {
            switch self {
            case .chat: "bubble.left.and.text.bubble.right"
            case .image: "photo.on.rectangle.angled"
            case .library: "doc.text.magnifyingglass"
            }
        }

        /// `paused`: the section lists an unfinished download.
        func footer(paused: Bool) -> String? {
            switch self {
            case .chat:
                paused ? "Resume a paused download in Chat: Browse the catalog, or find the repo again on Hugging Face."
                    : nil
            case .image: "Core ML may keep a compiled copy in the app's caches until the system clears them."
            case .library: "Without it the Library uses TF-IDF and re-indexes its documents the next time it opens."
            }
        }
    }

    let id: String
    let title: String
    let detail: String
    let usedIn: Usage
    let origin: Source
    let sizeBytes: Int64
    let isComplete: Bool
    /// Badges for a chat model that a record or the registry identifies; `nil` for other files.
    let capabilities: ModelCapabilities?

    var isDeletable: Bool {
        if case .chat(nil) = origin { return false }
        return true
    }

    var deleteMessage: String {
        let freed = "Frees \(Self.bytesText(sizeBytes))."
        switch origin {
        case .chat: return freed + " Unloads the model first; download it again from Chat."
        case .image(let model):
            return freed + (model.isBuiltIn ? " Download it again from Image."
                            : " Removes it from Image's model list; the built-in model is selected there.")
        case .embeddings: return freed + " The Library falls back to TF-IDF."
        }
    }

    static func bytesText(_ bytes: Int64) -> String {
        bytes.formatted(.byteCount(style: .file))
    }

    // MARK: Scan

    static func scan() -> [StoredModel] {
        let imageModels = CoreMLDiffusionModelStore.standard.all
        let imageRepos = Set(imageModels.map { $0.repoID.lowercased() })
        let modelsRoot = DownloadedModels.modelsDirectory.standardizedFileURL.path + "/"
        let finished = DownloadedModels.scan().map { ggufItem($0) }
        let paused = DownloadedModels.partialFiles()
            .filter { $0.standardizedFileURL.path.hasPrefix(modelsRoot) }
            .compactMap { partialItem($0, modelsRoot: modelsRoot) }
        // Image folders download into the same hub layout as MLX snapshots; they are listed under Image.
        let snapshots = DownloadedModels.scanSnapshots()
            .filter { !imageRepos.contains($0.repoID.lowercased()) }
            .map { snapshotItem($0) }
        let chat = (finished + paused + snapshots)
            .sorted { $0.title.localizedCaseInsensitiveCompare($1.title) == .orderedAscending }
        return chat + imageModels.compactMap { imageItem($0) } + embeddingItems()
    }

    private static func ggufItem(_ file: DownloadedGGUF) -> StoredModel {
        let known = knownGGUF(repoID: file.repoID, filename: file.filename)
        let model = known ?? unknownGGUF(repoID: file.repoID, filename: file.filename)
        return StoredModel(id: file.id, title: model?.displayName ?? file.displayName,
                           detail: "GGUF · \(file.filename)", usedIn: .chat, origin: .chat(model),
                           sizeBytes: file.sizeBytes, isComplete: true,
                           capabilities: known?.capabilities)
    }

    /// `<models>/<org>/<repo>/<file>.gguf.partial`; anything deeper is not a chat model's file.
    private static func partialItem(_ partial: URL, modelsRoot: String) -> StoredModel? {
        let relative = partial.standardizedFileURL.path.dropFirst(modelsRoot.count).split(separator: "/")
        guard relative.count == 3 else { return nil }
        let repoID = "\(relative[0])/\(relative[1])"
        let filename = partial.deletingPathExtension().lastPathComponent
        let known = knownGGUF(repoID: repoID, filename: filename)
        let model = known ?? unknownGGUF(repoID: repoID, filename: filename)
        return StoredModel(id: partial.path, title: model?.displayName ?? "\(relative[1]) · \(filename)",
                           detail: "GGUF · \(filename)", usedIn: .chat, origin: .chat(model),
                           sizeBytes: fileSize(partial), isComplete: false, capabilities: known?.capabilities)
    }

    private static func snapshotItem(_ snapshot: DownloadedSnapshot) -> StoredModel {
        let known = knownMLX(repoID: snapshot.repoID)
        let model = known ?? unknownMLX(repoID: snapshot.repoID)
        return StoredModel(id: snapshot.id, title: model?.displayName ?? snapshot.repoID,
                           detail: "MLX · \(snapshot.repoID)", usedIn: .chat, origin: .chat(model),
                           sizeBytes: snapshot.sizeBytes, isComplete: snapshot.isComplete,
                           capabilities: known?.capabilities)
    }

    private static func imageItem(_ model: CoreMLDiffusionModel) -> StoredModel? {
        let size = folderSize(model.resourcesDirectory)
        guard size > 0 else { return nil }
        let provenance = model.isBuiltIn ? "built in" : "\(model.repoID) · \(model.folder)"
        return StoredModel(id: "image:\(model.id)", title: model.displayName,
                           detail: "Core ML · \(model.imageSide)×\(model.imageSide) · \(provenance)", usedIn: .image,
                           origin: .image(model), sizeBytes: size, isComplete: model.isDownloaded, capabilities: nil)
    }

    private static func embeddingItems() -> [StoredModel] {
        let folder = AutoEmbeddingProvider.defaultModelBundleURL
        let size = folderSize(folder)
        guard size > 0 else { return [] }
        return [StoredModel(id: "embeddings:\(folder.path)", title: folder.lastPathComponent,
                            detail: "Core ML text embeddings", usedIn: .library, origin: .embeddings(folder),
                            sizeBytes: size, isComplete: true, capabilities: nil)]
    }

    // MARK: File identity

    /// The search record, else a registry entry for the same file. Only these say what the model does.
    private static func knownGGUF(repoID: String, filename: String) -> Model? {
        DownloadRecords.standard.model(repoID: repoID, ggufFilename: filename)
            ?? ModelRegistry.shared.models.first {
                $0.format == .gguf && $0.repoID == repoID && $0.ggufFilename == filename && $0.localFileURL == nil
            }
    }

    /// An unsized stand-in for a file nothing identifies, good only for deleting it; its category is a guess.
    private static func unknownGGUF(repoID: String, filename: String) -> Model? {
        Model.fromURL("https://huggingface.co/\(repoID)/resolve/main/\(filename)")
    }

    private static func knownMLX(repoID: String) -> Model? {
        DownloadRecords.standard.model(repoID: repoID, ggufFilename: nil)
            ?? ModelRegistry.shared.models.first { $0.format == .mlx && $0.repoID == repoID }
    }

    private static func unknownMLX(repoID: String) -> Model? {
        CatalogEntry(id: repoID, repoID: repoID, displayName: repoID, modelCategory: .text, weightFormat: .mlx,
                     approximateSizeMB: 0, isUncensored: false, ggufFilename: nil, numLayers: 0, kvHeads: 0,
                     headDim: 0, maxContextLength: nil).decodedModel()
    }

    // MARK: Sizes

    private static func fileSize(_ url: URL) -> Int64 {
        let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values?.isRegularFile == true else { return 0 }
        return Int64(values?.fileSize ?? 0)
    }

    /// Bytes of the regular files under `folder`; 0 when it does not exist.
    private static func folderSize(_ folder: URL) -> Int64 {
        let files = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: [.isRegularFileKey,
                                                                                            .fileSizeKey])
        return (files?.allObjects ?? []).compactMap { $0 as? URL }.map { fileSize($0) }.reduce(0, +)
    }
}
