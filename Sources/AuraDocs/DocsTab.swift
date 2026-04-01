import SwiftUI
import UniformTypeIdentifiers
import AuraCore

// MARK: - DocsTab

/// Drop-in SwiftUI tab for document management and RAG chat.
///
/// Provides file import, indexing progress, document list with swipe-to-export/delete,
/// and a grounded chat sheet powered by ``DocumentLibrary``.
public struct DocsTab: View {
    @StateObject private var vm = DocsViewModel()
    @State private var showFilePicker  = false
    @State private var showChat        = false
    @State private var showExportSheet = false
    
    public init() {}
    
    public var body: some View {
        NavigationStack {
            Group {
                if vm.documents.isEmpty && !vm.isIndexing {
                    emptyState
                } else {
                    documentList
                }
            }
            .navigationTitle("Documents")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button { showFilePicker = true } label: {
                        Image(systemName: "plus")
                    }
                    .disabled(vm.isIndexing || !vm.isReady)
                }
                ToolbarItem(placement: .primaryAction) {
                    Button { showChat = true } label: {
                        Image(systemName: "bubble.left.and.bubble.right")
                    }
                    .disabled(vm.documents.isEmpty || !vm.isReady)
                }
                if !vm.documents.isEmpty {
                    ToolbarItem(placement: .navigation) {
                        Button { showExportSheet = true } label: {
                            Image(systemName: "square.and.arrow.up")
                        }
                        .disabled(vm.isExporting)
                    }
                }
            }
            .fileImporter(
                isPresented: $showFilePicker,
                allowedContentTypes: supportedTypes,
                allowsMultipleSelection: true
            ) { result in
                if case .success(let urls) = result {
                    Task { await vm.index(urls: urls) }
                }
            }
            .sheet(isPresented: $showChat) {
                if let chat = vm.documentChat {
                    DocumentChatSheet(chat: chat)
                }
            }
            .sheet(isPresented: $showExportSheet) {
                ExportSheet(vm: vm)
            }
            .task { await vm.setup() }
        }
    }
    
    // MARK: - Subviews
    
    private var emptyState: some View {
        VStack(spacing: 16) {
            if !vm.isReady {
                ProgressView()
                Text(vm.progress)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Image(systemName: "doc.badge.plus")
                    .font(.system(size: 48))
                    .foregroundStyle(.secondary)
                Text("No documents indexed")
                    .font(.headline)
                Text("Tap + to add PDF, Word, text or image files")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                Button("Add Documents") { showFilePicker = true }
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding()
    }
    
    private var documentList: some View {
        List {
            // Indexing progress
            if vm.isIndexing {
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Image(systemName: "doc.badge.gearshape")
                                .foregroundStyle(.blue)
                            Text(vm.indexingFile.isEmpty ? "Processing..." : vm.indexingFile)
                                .font(.subheadline.weight(.medium))
                                .lineLimit(1)
                            Spacer()
                            Text("\(Int(vm.indexingProgress * 100))%")
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        ProgressView(value: vm.indexingProgress)
                            .tint(.blue)
                        Text(vm.progress)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    .padding(.vertical, 6)
                }
            }
            
            // Document list
            Section {
                ForEach(vm.documents) { doc in
                    DocumentRow(document: doc)
                        .swipeActions(edge: .leading, allowsFullSwipe: false) {
                            Button {
                                Task { await vm.exportSingle(document: doc) }
                            } label: {
                                Label("Export", systemImage: "square.and.arrow.up")
                            }
                            .tint(.blue)
                        }
                        .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                            Button(role: .destructive) {
                                Task { await vm.remove(document: doc) }
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                }
            } header: {
                Text("\(vm.documents.count) document\(vm.documents.count == 1 ? "" : "s")")
            }
        }
    }
    
    private var supportedTypes: [UTType] {
        [.pdf, .text, .plainText,
         UTType(filenameExtension: "docx") ?? .data,
         UTType(filenameExtension: "md")   ?? .text,
         .png, .jpeg, .heic, .tiff]
    }
}

// MARK: - DocumentRow

private struct DocumentRow: View {
    let document: IndexedDocument
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: iconName)
                .font(.title2)
                .foregroundStyle(iconColor)
                .frame(width: 36)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(document.title)
                    .font(.subheadline.weight(.medium))
                    .lineLimit(1)
                HStack(spacing: 6) {
                    Text("\(document.chunkCount) chunks")
                    Text(".")
                    Text(document.indexedAt, style: .relative)
                }
                .font(.caption2)
                .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }
    
    private var ext: String { document.url.pathExtension.lowercased() }
    
    private var iconName: String {
        switch ext {
            case "pdf":                        return "doc.fill"
            case "docx":                       return "doc.richtext"
            case "txt", "md", "markdown":      return "doc.text"
            case "png", "jpg", "jpeg", "heic": return "photo"
            default:                           return "doc"
        }
    }
    
    private var iconColor: Color {
        switch ext {
            case "pdf":       return .red
            case "docx":      return .blue
            case "txt", "md": return .primary
            default:          return .orange
        }
    }
}

// MARK: - ExportSheet

struct ExportSheet: View {
    @ObservedObject var vm: DocsViewModel
    @Environment(\.dismiss) private var dismiss
    
    @State private var format: ExportFormat      = .jsonlGz
    @State private var includeEmbeddings         = false
    @State private var exportAll                 = true
    @State private var selectedIDs: Set<UUID>    = []
    
    var body: some View {
        NavigationStack {
            Form {
                Section("Format") {
                    Picker("Format", selection: $format) {
                        Text("JSONL.GZ").tag(ExportFormat.jsonlGz)
                        Text("JSONL").tag(ExportFormat.jsonl)
                    }
                    .pickerStyle(.segmented)
                    Toggle("Include embedding vectors", isOn: $includeEmbeddings)
                }
                
                Section("Documents") {
                    Toggle("Export all (\(vm.documents.count))", isOn: $exportAll)
                    if !exportAll {
                        ForEach(vm.documents) { doc in
                            HStack {
                                Image(systemName: selectedIDs.contains(doc.id)
                                      ? "checkmark.circle.fill" : "circle")
                                .foregroundStyle(selectedIDs.contains(doc.id) ? .blue : .secondary)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(doc.title).font(.subheadline)
                                    Text("\(doc.chunkCount) chunks")
                                        .font(.caption2).foregroundStyle(.secondary)
                                }
                            }
                            .contentShape(Rectangle())
                            .onTapGesture {
                                if selectedIDs.contains(doc.id) { selectedIDs.remove(doc.id) }
                                else { selectedIDs.insert(doc.id) }
                            }
                        }
                    }
                }
                
                if vm.isExporting {
                    Section {
                        HStack(spacing: 12) {
                            ProgressView()
                            Text(vm.exportProgress)
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                
                if !vm.exportedURLs.isEmpty {
                    Section("Exported Files") {
                        ForEach(vm.exportedURLs, id: \.self) { url in
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(url.lastPathComponent)
                                        .font(.subheadline)
                                        .lineLimit(1)
                                    if let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                                        Text(ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file))
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                                Spacer()
                                ShareLink(item: url) {
                                    Image(systemName: "square.and.arrow.up")
                                        .foregroundStyle(.blue)
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Export")
#if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
#endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
                ToolbarItem(placement: .primaryAction) {
                    Button("Export") {
                        let ids = exportAll
                        ? vm.documents.map(\.id)
                        : Array(selectedIDs)
                        Task {
                            await vm.exportDocuments(
                                ids:               ids,
                                format:            format,
                                includeEmbeddings: includeEmbeddings
                            )
                        }
                    }
                    .fontWeight(.semibold)
                    .disabled(vm.isExporting || (!exportAll && selectedIDs.isEmpty))
                }
            }
        }
    }
}

// MARK: - DocumentChatSheet

struct DocumentChatSheet: View {
    @ObservedObject var chat: DocumentChat
    @State private var prompt = ""
    @FocusState private var focused: Bool
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(chat.messages) { msg in
                                DocMessageBubble(message: msg)
                                    .id(msg.id)
                            }
                            if chat.isThinking {
                                HStack {
                                    ProgressView()
                                    Text("Searching documents...")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                    Spacer()
                                }
                                .padding()
                                .id("thinking")
                            }
                        }
                        .padding()
                    }
                    .onChange(of: chat.messages.count) { _, _ in
                        withAnimation { proxy.scrollTo(chat.messages.last?.id, anchor: .bottom) }
                    }
                    .onChange(of: chat.isThinking) { _, v in
                        if v { withAnimation { proxy.scrollTo("thinking", anchor: .bottom) } }
                    }
                }
                
                Divider()
                
                HStack(spacing: 10) {
                    TextField("Ask about your documents...", text: $prompt, axis: .vertical)
                        .lineLimit(1...4)
                        .padding(10)
                        .background(Color.tertiaryGroupedBackground, in: RoundedRectangle(cornerRadius: 12))
                        .focused($focused)
                    
                    Button {
                        let q = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !q.isEmpty, !chat.isThinking else { return }
                        prompt = ""
                        focused = false
                        Task { try? await chat.send(q) }
                    } label: {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 32))
                            .foregroundStyle(chat.isThinking ? .gray : .blue)
                    }
                    .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || chat.isThinking)
                }
                .padding(12)
                .background(Color.secondaryGroupedBackground)
            }
            .background(Color.groupedBackground)
            .navigationTitle("Document Chat")
#if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
#endif
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

// MARK: - DocMessageBubble

private struct DocMessageBubble: View {
    let message: DocumentChatMessage
    @State private var showSources = false
    
    var body: some View {
        VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
            HStack {
                if message.role == .user { Spacer(minLength: 48) }
                Text(message.text)
                    .padding(12)
                    .background(
                        message.role == .user
                        ? Color.blue.opacity(0.12)
                        : Color.secondaryGroupedBackground,
                        in: RoundedRectangle(cornerRadius: 16)
                    )
                    .font(.subheadline)
                if message.role == .assistant { Spacer(minLength: 48) }
            }
            
            if message.role == .assistant, !message.sources.isEmpty {
                Button {
                    withAnimation(.spring(duration: 0.3)) { showSources.toggle() }
                } label: {
                    Label(
                        showSources
                        ? "Hide sources"
                        : "\(message.sources.count) source\(message.sources.count == 1 ? "" : "s")",
                        systemImage: "doc.text.magnifyingglass"
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .padding(.leading, 4)
                
                if showSources {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(Array(message.sources.enumerated()), id: \.offset) { _, src in
                            SourceCard(source: src)
                        }
                    }
                    .transition(.move(edge: .top).combined(with: .opacity))
                }
            }
        }
    }
}

private struct SourceCard: View {
    let source: DocumentAnswer.SourceReference
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Label(
                    source.pageNumber > 0
                    ? "\(source.documentTitle) p.\(source.pageNumber)"
                    : source.documentTitle,
                    systemImage: "doc.text"
                )
                .font(.caption.weight(.medium))
                Spacer()
                Text(String(format: "%.0f%%", source.score * 100))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Text(source.excerpt)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(3)
        }
        .padding(8)
        .background(Color.tertiaryGroupedBackground, in: RoundedRectangle(cornerRadius: 10))
    }
}

// MARK: - DocsViewModel

@MainActor
final class DocsViewModel: ObservableObject {
    @Published var documents:          [IndexedDocument] = []
    @Published var isIndexing         = false
    @Published var isReady            = false
    @Published var isExporting        = false
    @Published var progress           = ""
    @Published var exportProgress     = ""
    @Published var indexingFile       = ""
    @Published var indexingProgress:  Double = 0
    @Published var exportedURLs:      [URL] = []
    private(set) var documentChat: DocumentChat?
    
    private let library = DocumentLibrary.shared
    private let store   = ConversationStore.shared
    
    // MARK: - Setup
    
    func setup() async {
        progress = "Loading model..."
        do {
            // Reuse already-loaded models from other tabs via ModelManager
            let llm = try await ModelManager.shared.load(.qwen3_1_7b) { [weak self] p in
                Task { @MainActor [weak self] in self?.progress = p }
            }
            let vlm = try await ModelManager.shared.load(.fastVLM_0_5b_fp16) { [weak self] p in
                Task { @MainActor [weak self] in self?.progress = p }
            }
            let embedder = AutoEmbeddingProvider()
            
            await library.configure(embeddingProvider: embedder, llm: llm, visionLLM: vlm)
            try await library.open()
            
            documentChat = DocumentChat(library: library, llm: llm, store: store)
            documents    = try await library.allDocuments()
            
            let backend = await embedder.backendName()
            progress = "Ready - \(backend)"
            isReady  = true
        } catch {
            progress = "Error: \(error.localizedDescription)"
        }
    }
    
    // MARK: - Indexing
    
    func index(urls: [URL]) async {
        isIndexing = true
        for url in urls {
            indexingFile     = url.deletingPathExtension().lastPathComponent
            indexingProgress = 0
            do {
                let doc = try await library.add(url: url) { [weak self] p in
                    self?.progress = p
                    if let pct = Self.parsePercent(from: p) {
                        self?.indexingProgress = pct
                    } else if p.hasPrefix("Parsing") {
                        self?.indexingProgress = 0.05
                    } else if p.hasPrefix("Chunking") {
                        self?.indexingProgress = 0.15
                    } else if p.contains("indexed") {
                        self?.indexingProgress = 1.0
                    }
                }
                if !documents.contains(where: { $0.id == doc.id }) {
                    documents.insert(doc, at: 0)
                }
                await library.refreshCorpus()
            } catch {
                progress = "Error: \(error.localizedDescription)"
            }
        }
        isIndexing       = false
        indexingFile     = ""
        indexingProgress = 0
        progress         = ""
    }
    
    // MARK: - Export
    
    /// Quick export from swipe action — single document as JSONL.GZ.
    func exportSingle(document: IndexedDocument) async {
        isExporting    = true
        exportProgress = "Exporting \(document.title)..."
        do {
            let dest = exportDirectory()
            let url  = try await library.export(
                documentID: document.id,
                to:         dest,
                format:     .jsonlGz
            )
            exportedURLs.insert(url, at: 0)
            exportProgress = "Done: \(url.lastPathComponent)"
        } catch {
            exportProgress = "Error: \(error.localizedDescription)"
        }
        isExporting = false
    }
    
    /// Batch export from ExportSheet.
    func exportDocuments(
        ids:               [UUID],
        format:            ExportFormat,
        includeEmbeddings: Bool
    ) async {
        guard !ids.isEmpty else { return }
        isExporting   = true
        exportedURLs  = []
        let dest      = exportDirectory()
        
        for (i, id) in ids.enumerated() {
            let title = documents.first(where: { $0.id == id })?.title ?? id.uuidString
            exportProgress = "Exporting \(i + 1)/\(ids.count): \(title)..."
            do {
                let url = try await library.export(
                    documentID:        id,
                    to:                dest,
                    format:            format,
                    includeEmbeddings: includeEmbeddings
                )
                exportedURLs.append(url)
            } catch {
                exportProgress = "Error \(title): \(error.localizedDescription)"
            }
        }
        
        exportProgress = "\(exportedURLs.count) file\(exportedURLs.count == 1 ? "" : "s") exported"
        isExporting    = false
    }
    
    // MARK: - Delete
    
    func remove(document: IndexedDocument) async {
        try? await library.removeDocument(id: document.id)
        documents.removeAll { $0.id == document.id }
    }
    
    // MARK: - Helpers
    
    private func exportDirectory() -> URL {
        FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AuraExports", isDirectory: true)
    }
    
    private static func parsePercent(from text: String) -> Double? {
        guard text.contains("%"),
              let range = text.range(of: #"(\d+)%"#, options: .regularExpression),
              let num = Double(text[range].dropLast())
        else { return nil }
        return 0.15 + (num / 100.0) * 0.85
    }
}

// MARK: - Cross-platform colors

private extension Color {
    static var groupedBackground: Color {
#if os(iOS)
        Color(.systemGroupedBackground)
#else
        Color(nsColor: .windowBackgroundColor)
#endif
    }
    static var secondaryGroupedBackground: Color {
#if os(iOS)
        Color(.secondarySystemGroupedBackground)
#else
        Color(nsColor: .controlBackgroundColor)
#endif
    }
    static var tertiaryGroupedBackground: Color {
#if os(iOS)
        Color(.tertiarySystemGroupedBackground)
#else
        Color(nsColor: .textBackgroundColor)
#endif
    }
}
