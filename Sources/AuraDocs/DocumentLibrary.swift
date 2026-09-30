import Foundation
import AuraCore

// MARK: - IndexedDocument

/// A document that has been parsed, chunked, and embedded into the vector store.
public struct IndexedDocument: Identifiable, Sendable {
    public let id: UUID
    public let title: String
    public let url: URL
    public let chunkCount: Int
    public let indexedAt: Date
}

// MARK: - DocumentAnswer

/// The result of a RAG query against the ``DocumentLibrary``.
///
/// Contains the LLM-generated answer together with the source chunks
/// that were used as grounding context, ranked by relevance score.
public struct DocumentAnswer: Sendable {
    /// LLM-generated answer grounded in the retrieved document chunks.
    public let text: String
    /// Source chunks used to generate the answer, ranked by relevance.
    public let sources: [SourceReference]

    /// A reference to a specific document chunk that contributed to an answer.
    public struct SourceReference: Sendable {
        /// Title of the source document.
        public let documentTitle: String
        /// 1-based page number (0 if the format has no page concept).
        public let pageNumber: Int
        /// First 200 characters of the chunk text.
        public let excerpt: String
        /// Cosine similarity score (0–1) between the query and this chunk.
        public let score: Float
    }
}

// MARK: - DocumentLibrary

/// Manages a local library of indexed documents for RAG queries.
///
/// ```swift
/// // Setup (once)
/// let library = DocumentLibrary(
///     embeddingProvider: OpenAIEmbeddingProvider(apiKey: "sk-..."),
///     llm: llm
/// )
/// try await library.open()
///
/// // Index documents
/// try await library.add(url: pdfURL)
/// try await library.add(url: docxURL)
///
/// // Ask a question
/// let answer = try await library.ask("What is the contract amount?")
/// print(answer.text)
/// print(answer.sources.map { "[\($0.documentTitle) p.\($0.pageNumber)]" })
/// ```
public actor DocumentLibrary {
    
    // MARK: - Singleton
    
    public static let shared = DocumentLibrary()
    
    // MARK: - Dependencies
    
    private var embeddingProvider: (any EmbeddingProvider)?
    private var llm: AuraLocal?
    private var visionLLM: AuraLocal?
    
    private let vectorStore: VectorStore
    private let chunker:     DocumentChunker
    
    /// The provider identity the stored vectors are known to match; `nil` until checked.
    private var matchedIdentity: EmbeddingIndexIdentity?
    /// The latest identity check or re-embed; each one waits for the previous.
    private var indexJob: Task<Void, Error>?
    private static let embedBatchSize = 50

    // MARK: - Init
    
    public init(
        directory: URL? = nil,
        chunkTargetTokens: Int = 512,
        chunkOverlapFraction: Double = 0.1
    ) {
        let base = directory ?? FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AuraLocal/docs", isDirectory: true)
        
        self.vectorStore = VectorStore(directory: base)
        self.chunker     = DocumentChunker(
            targetTokens:    chunkTargetTokens,
            overlapFraction: chunkOverlapFraction
        )
    }
    
    // MARK: - Configuration
    
    /// Set the embedding provider and LLM instances used for indexing and querying.
    ///
    /// Must be called before ``add(url:onProgress:)`` or ``ask(_:topK:maxContextTokens:systemPrompt:)``.
    public func configure(
        embeddingProvider: any EmbeddingProvider,
        llm: AuraLocal,
        visionLLM: AuraLocal? = nil
    ) {
        self.embeddingProvider = embeddingProvider
        self.llm               = llm
        self.visionLLM         = visionLLM
        self.matchedIdentity   = nil
    }
    
    // MARK: - Lifecycle
    
    /// Open the underlying SQLite vector store. Must be called once before any indexing or querying.
    public func open() async throws {
        try await vectorStore.open()
    }

    /// Close the vector store database connection.
    public func close() async {
        await vectorStore.close()
    }
    
    // MARK: - Indexing
    
    /// Index a document from a file URL. Safe to call multiple times — skips if already indexed.
    @discardableResult
    public func add(
        url: URL,
        onProgress: @escaping @MainActor (String) -> Void = { _ in }
    ) async throws -> IndexedDocument {
        guard let embedder = embeddingProvider else {
            throw DocumentError.libraryNotReady
        }
        
        let docID = deterministicID(for: url)
        
        // Skip if already indexed
        if try await vectorStore.documentExists(id: docID) {
            await onProgress("'\(url.lastPathComponent)' already indexed.")
            let docs = try await allDocuments()
            if let existing = docs.first(where: { $0.id == docID }) { return existing }
        }
        
        // Stored vectors must come from this provider before new ones join them
        try await prepareIndex(onProgress: onProgress)

        // Parse
        await onProgress("Parsing \(url.lastPathComponent)…")
        let dispatcher = DocumentParserDispatcher(visionLLM: visionLLM)
        let parsed     = try await dispatcher.parse(url: url)
        
        // Chunk
        await onProgress("Chunking \(parsed.title)…")
        var chunks = chunker.chunk(document: parsed, documentID: docID)
        
        // Embed in batches
        await onProgress("Embedding \(chunks.count) chunks…")
        let truncatedBefore = await truncatedInputCount(of: embedder)
        let batchSize = Self.embedBatchSize
        for batchStart in stride(from: 0, to: chunks.count, by: batchSize) {
            let batchEnd   = min(batchStart + batchSize, chunks.count)
            let texts      = chunks[batchStart..<batchEnd].map(\.text)
            var embeddings = try await embedder.embedDocuments(texts)
            for i in 0..<embeddings.count {
                VectorMath.normalize(&embeddings[i])
                chunks[batchStart + i].embedding = embeddings[i]
            }
            await onProgress("Embedding \(parsed.title): \(Self.percent(batchEnd, of: chunks.count))%")
        }
        let truncated = await truncatedInputCount(of: embedder) - truncatedBefore
        
        // Persist
        try await vectorStore.insertDocument(
            id:         docID,
            title:      parsed.title,
            url:        url.absoluteString,
            chunkCount: chunks.count
        )
        try await vectorStore.insertChunks(chunks)
        
        let truncationNote = truncated > 0 ? ", \(truncated) cut at the model's token limit" : ""
        await onProgress("'\(parsed.title)' indexed ✓ (\(chunks.count) chunks\(truncationNote))")
        
        return IndexedDocument(
            id:         docID,
            title:      parsed.title,
            url:        url,
            chunkCount: chunks.count,
            indexedAt:  Date()
        )
    }
    
    // MARK: - Query
    
    /// Ask a question against the entire document library.
    public func ask(
        _ question: String,
        topK: Int = 5,
        maxContextTokens: Int = 2048,
        systemPrompt: String? = nil
    ) async throws -> DocumentAnswer {
        guard let embedder = embeddingProvider, let llm else {
            throw DocumentError.libraryNotReady
        }
        
        try await prepareIndex(onProgress: { _ in })

        // Embed query
        var queryVec = try await embedder.embedQuery(question)
        VectorMath.normalize(&queryVec)
        
        // Hybrid retrieval
        var results = try await vectorStore.search(
            query:          question,
            queryEmbedding: queryVec,
            topK:           topK
        )
        
        // Fallback to pure cosine if FTS returned nothing
        if results.isEmpty {
            results = try await vectorStore.searchCosineOnly(
                queryEmbedding: queryVec,
                topK:           topK
            )
        }
        
        guard !results.isEmpty else {
            return DocumentAnswer(
                text:    "No relevant information found in the indexed documents.",
                sources: []
            )
        }
        
        // Build context — respect token budget
        let context = buildContext(from: results, maxTokens: maxContextTokens)
        
        // Compose prompt
        let sys = systemPrompt ?? """
            You are a helpful assistant. Answer questions based ONLY on the provided document context.
            If the answer is not in the context, say so clearly.
            Always cite the document title and page number when referencing specific information.
            """
        
        let prompt = """
            Document context:
            \(context)
            
            ---
            Question: \(question)
            """
        
        let answer = try await llm.chat(prompt, systemPrompt: sys)
        
        let sources = results.map { r in
            DocumentAnswer.SourceReference(
                documentTitle: r.chunk.documentTitle,
                pageNumber:    r.chunk.pageNumber,
                excerpt:       String(r.chunk.text.prefix(200)),
                score:         r.score
            )
        }
        
        return DocumentAnswer(text: answer, sources: sources)
    }
    
    // MARK: - Index identity

    /// True when the stored vectors came from a different embedding provider (or predate identity
    /// tracking and have another length), so the next ``add(url:onProgress:)`` or
    /// ``ask(_:topK:maxContextTokens:systemPrompt:)`` re-embeds them first. Call
    /// ``reembedAll(onProgress:)`` up front to show progress instead.
    public func indexNeedsReembedding() async throws -> Bool {
        guard let embedder = embeddingProvider else { throw DocumentError.libraryNotReady }
        return try await reconciliation(for: embedder) == .reembed
    }

    /// Recomputes every stored vector from its stored chunk text with the configured provider, in
    /// batches, and records the provider's identity. Documents are not parsed again.
    /// Progress messages read like `"Re-embedding 50/400 chunks: 12%"`.
    public func reembedAll(onProgress: @escaping @MainActor (String) -> Void = { _ in }) async throws {
        guard let embedder = embeddingProvider else { throw DocumentError.libraryNotReady }
        try await runIndexJob(embedder: embedder, forceReembed: true, onProgress: onProgress)
    }

    private func prepareIndex(onProgress: @escaping @MainActor (String) -> Void) async throws {
        guard let embedder = embeddingProvider else { throw DocumentError.libraryNotReady }
        if matchedIdentity == Self.identity(of: embedder) { return }
        try await runIndexJob(embedder: embedder, forceReembed: false, onProgress: onProgress)
    }

    /// Chains index jobs so concurrent add/ask calls wait for one re-embed instead of starting two.
    private func runIndexJob(
        embedder: any EmbeddingProvider,
        forceReembed: Bool,
        onProgress: @escaping @MainActor (String) -> Void
    ) async throws {
        let previous = indexJob
        let job = Task {
            _ = await previous?.result
            try await self.reconcileIndex(embedder: embedder, forceReembed: forceReembed, onProgress: onProgress)
        }
        indexJob = job
        try await job.value
    }

    private func reconcileIndex(
        embedder: any EmbeddingProvider,
        forceReembed: Bool,
        onProgress: @escaping @MainActor (String) -> Void
    ) async throws {
        let configured = Self.identity(of: embedder)
        if !forceReembed {
            if matchedIdentity == configured { return }
            switch try await reconciliation(for: embedder) {
            case .upToDate:
                matchedIdentity = configured
                return
            case .adopt:
                try await vectorStore.setEmbeddingIdentity(configured)
                matchedIdentity = configured
                return
            case .reembed:
                break
            }
        }
        try await reembed(with: embedder, identity: configured, onProgress: onProgress)
    }

    private func reconciliation(for embedder: any EmbeddingProvider) async throws
        -> EmbeddingIndexIdentity.Reconciliation {
        let configured = Self.identity(of: embedder)
        let stored = try await vectorStore.embeddingIdentity()
        if stored == configured { return .upToDate }
        return EmbeddingIndexIdentity.reconciliation(
            stored: stored,
            configured: configured,
            storedVectorLengths: try await vectorStore.storedVectorLengths(),
            storedVectorCount: try await vectorStore.chunkCount()
        )
    }

    private func reembed(
        with embedder: any EmbeddingProvider,
        identity: EmbeddingIndexIdentity,
        onProgress: @escaping @MainActor (String) -> Void
    ) async throws {
        matchedIdentity = nil
        // Matches no provider, so a run interrupted halfway is redone on next use.
        let pending = EmbeddingIndexIdentity(identifier: "pending:\(identity.identifier)", dimensions: identity.dimensions)
        try await vectorStore.setEmbeddingIdentity(pending)
        await vectorStore.clearEmbeddingCache()
        if let auto = embedder as? AutoEmbeddingProvider, !auto.usesDenseModel {
            await refreshCorpus()
        }
        let total = try await vectorStore.chunkCount()
        await onProgress("Re-embedding \(total) chunks…")
        var done = 0
        var cursor: Int64 = 0
        while let page = try await reembedPage(after: cursor, embedder: embedder) {
            cursor = page.lastRowID
            done += page.count
            await onProgress("Re-embedding \(done)/\(total) chunks: \(Self.percent(done, of: total))%")
        }
        try await vectorStore.setEmbeddingIdentity(identity)
        matchedIdentity = identity
    }

    /// Re-embeds the next page of chunks after `rowID`; `nil` when none are left.
    private func reembedPage(
        after rowID: Int64,
        embedder: any EmbeddingProvider
    ) async throws -> (lastRowID: Int64, count: Int)? {
        let page = try await vectorStore.chunkTexts(afterRowID: rowID, limit: Self.embedBatchSize)
        guard let last = page.last else { return nil }
        var vectors = try await embedder.embedDocuments(page.map(\.text))
        guard vectors.count == page.count else {
            throw DocumentError.embeddingFailed("\(vectors.count) vectors for \(page.count) chunks")
        }
        for index in vectors.indices {
            VectorMath.normalize(&vectors[index])
        }
        try await vectorStore.updateEmbeddings(zip(page, vectors).map { (id: $0.0.id, vector: $0.1) })
        return (last.rowID, page.count)
    }

    private static func identity(of embedder: any EmbeddingProvider) -> EmbeddingIndexIdentity {
        EmbeddingIndexIdentity(identifier: embedder.identifier, dimensions: embedder.dimensions)
    }

    private static func percent(_ done: Int, of total: Int) -> Int {
        total > 0 ? done * 100 / total : 100
    }

    private func truncatedInputCount(of embedder: any EmbeddingProvider) async -> Int {
        guard let reporter = embedder as? any TruncationReporting else { return 0 }
        return await reporter.truncatedInputCount()
    }

    // MARK: - Library management
    
    /// List all documents currently indexed in the library.
    public func allDocuments() async throws -> [IndexedDocument] {
        try await vectorStore.allDocuments().map { row in
            IndexedDocument(
                id:         row.id,
                title:      row.title,
                url:        URL(string: row.url) ?? URL(fileURLWithPath: row.url),
                chunkCount: row.chunkCount,
                indexedAt:  row.indexedAt
            )
        }
    }
    
    /// Remove a document and all its chunks from the library.
    public func removeDocument(id: UUID) async throws {
        try await vectorStore.deleteDocument(id: id)
    }
    
    // MARK: - Export
    
    /// Export a single document's chunks to JSONL or JSONL.GZ.
    ///
    /// - Parameters:
    ///   - documentID:        ID of the document to export.
    ///   - destination:       Directory where the file will be written.
    ///   - format:            `.jsonlGz` (default) or `.jsonl`.
    ///   - includeEmbeddings: Include float vectors in each record (larger file).
    /// - Returns: URL of the written file.
    @discardableResult
    public func export(
        documentID:        UUID,
        to destination:    URL,
        format:            ExportFormat = .jsonlGz,
        includeEmbeddings: Bool = false
    ) async throws -> URL {
        let docs = try await allDocuments()
        guard let document = docs.first(where: { $0.id == documentID }) else {
            throw StoreError.conversationNotFound(documentID)
        }
        
        var chunks = try await vectorStore.chunksForDocument(id: documentID)
        
        if includeEmbeddings {
            let chunkIDs = chunks.map(\.id)
            let embeddings = try await vectorStore.embeddings(for: chunkIDs)
            for i in chunks.indices {
                chunks[i].embedding = embeddings[chunks[i].id] ?? []
            }
        }
        
        return try DocumentExporter.export(
            document:          document,
            chunks:            chunks,
            to:                destination,
            format:            format,
            includeEmbeddings: includeEmbeddings
        )
    }
    
    // MARK: - Corpus
    
    /// Re-feeds all stored chunk texts into the embedding provider's corpus.
    /// Required for TFIDFEmbeddingProvider to have accurate IDF weights.
    public func refreshCorpus() async {
        guard let embedder = embeddingProvider as? AutoEmbeddingProvider else { return }
        let texts = (try? await vectorStore.allChunkTexts()) ?? []
        await embedder.updateCorpus(texts: texts)
    }
    
    // MARK: - Helpers
    
    private func deterministicID(for url: URL) -> UUID {
        // UUID v5-style: hash the canonical path
        let path = url.standardizedFileURL.path
        let hash = abs(path.hashValue)
        return UUID(uuid: (
            UInt8((hash >> 56) & 0xFF), UInt8((hash >> 48) & 0xFF),
            UInt8((hash >> 40) & 0xFF), UInt8((hash >> 32) & 0xFF),
            UInt8((hash >> 24) & 0xFF), UInt8((hash >> 16) & 0xFF),
            UInt8((hash >> 8)  & 0xFF), UInt8(hash & 0xFF),
            0x40, 0, 0, 0, 0, 0, 0, 0
        ))
    }
    
    private func buildContext(from results: [VectorStore.SearchResult], maxTokens: Int) -> String {
        var lines:  [String] = []
        var tokens  = 0
        
        for result in results {
            let chunk   = result.chunk
            let pageRef = chunk.pageNumber > 0 ? " (p. \(chunk.pageNumber))" : ""
            let header  = "[\(chunk.documentTitle)\(pageRef)]"
            let entry   = "\(header)\n\(chunk.text)"
            let entryTok = max(1, entry.count / 4)
            
            guard tokens + entryTok <= maxTokens else { break }
            lines.append(entry)
            tokens += entryTok
        }
        
        return lines.joined(separator: "\n\n---\n\n")
    }
}

// MARK: - DocumentChat

/// A stateful, observable chat session grounded in a ``DocumentLibrary``.
///
/// Each question is answered using RAG retrieval, and both the question
/// and answer are persisted to ``ConversationStore`` for history.
///
/// ```swift
/// let chat = DocumentChat(library: library, llm: llm)
/// let answer = try await chat.send("What is the contract amount?")
/// // chat.messages now contains the user question and the grounded answer
/// ```
@MainActor
public final class DocumentChat: ObservableObject {
    
    // MARK: Published
    
    @Published public private(set) var messages: [DocumentChatMessage] = []
    @Published public private(set) var isThinking = false
    @Published public private(set) var progress = ""
    
    // MARK: Private
    
    private let library: DocumentLibrary
    private let llm: AuraLocal
    private let store: ConversationStore
    private var conversationID: UUID?
    
    public init(
        library: DocumentLibrary,
        llm: AuraLocal,
        store: ConversationStore = .shared
    ) {
        self.library = library
        self.llm     = llm
        self.store   = store
    }
    
    // MARK: - Send
    
    /// Send a question grounded in the document library.
    @discardableResult
    public func send(
        _ question: String,
        topK: Int = 5
    ) async throws -> DocumentAnswer {
        messages.append(DocumentChatMessage(role: .user, text: question))
        isThinking = true
        defer { isThinking = false }
        
        let answer = try await library.ask(question, topK: topK)
        
        // Persist to ConversationStore for history
        if conversationID == nil {
            let conv = try await store.createConversation(model: llm.model, title: "Document chat")
            conversationID = conv.id
        }
        if let convID = conversationID {
            try await store.appendTurn(Turn(conversationID: convID, role: .user,      content: question))
            try await store.appendTurn(Turn(conversationID: convID, role: .assistant, content: answer.text))
        }
        
        messages.append(DocumentChatMessage(
            role:    .assistant,
            text:    answer.text,
            sources: answer.sources
        ))
        
        return answer
    }
    
    public func clear() {
        messages = []
    }
}

// MARK: - DocumentChatMessage

/// A single message in a ``DocumentChat`` session — either a user question or a grounded assistant answer.
public struct DocumentChatMessage: Identifiable, Sendable {
    public let id = UUID()
    public enum Role: Sendable { case user, assistant }
    public let role: Role
    public let text: String
    public let sources: [DocumentAnswer.SourceReference]
    
    init(role: Role, text: String, sources: [DocumentAnswer.SourceReference] = []) {
        self.role    = role
        self.text    = text
        self.sources = sources
    }
}
