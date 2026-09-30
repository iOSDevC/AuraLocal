import Foundation

// MARK: - DocumentChunk

/// A text fragment from an indexed document, ready for embedding and retrieval.
///
/// Chunks are created during ``DocumentLibrary/add(url:onProgress:)`` and
/// stored in the vector store alongside their embedding vectors.
public struct DocumentChunk: Identifiable, Sendable {
    public let id: UUID
    public let documentID: UUID
    public let documentTitle: String
    public let pageNumber: Int
    public let text: String
    public let tokenEstimate: Int
    /// Populated after embedding
    public internal(set) var embedding: [Float]

    init(
        id: UUID = UUID(),
        documentID: UUID,
        documentTitle: String,
        pageNumber: Int,
        text: String
    ) {
        self.id            = id
        self.documentID    = documentID
        self.documentTitle = documentTitle
        self.pageNumber    = pageNumber
        self.text          = text
        self.tokenEstimate = max(1, text.count / 4)
        self.embedding     = []
    }
}

// MARK: - DocumentChunker

/// Splits a ParsedDocument into overlapping chunks suitable for embedding.
struct DocumentChunker {
    let targetTokens: Int    // target chunk size
    let overlapTokens: Int   // overlap between consecutive chunks

    init(targetTokens: Int = 512, overlapFraction: Double = 0.1) {
        self.targetTokens  = targetTokens
        self.overlapTokens = max(1, Int(Double(targetTokens) * overlapFraction))
    }

    func chunk(document: ParsedDocument, documentID: UUID) -> [DocumentChunk] {
        var chunks: [DocumentChunk] = []

        for page in document.pages {
            let pageChunks = chunkText(
                page.text,
                documentID:    documentID,
                documentTitle: document.title,
                pageNumber:    page.pageNumber
            )
            chunks.append(contentsOf: pageChunks)
        }

        return chunks
    }

    // MARK: - Private

    private func chunkText(
        _ text: String,
        documentID: UUID,
        documentTitle: String,
        pageNumber: Int
    ) -> [DocumentChunk] {
        // Split into sentences first for cleaner boundaries
        let sentences = splitSentences(text)
        guard !sentences.isEmpty else { return [] }

        var chunks:  [DocumentChunk] = []
        var buffer:  [String]        = []
        var bufTok   = 0

        for sentence in sentences {
            let sTok = max(1, sentence.count / 4)

            if bufTok + sTok > targetTokens, !buffer.isEmpty {
                let chunkText = buffer.joined(separator: " ")
                chunks.append(DocumentChunk(
                    documentID:    documentID,
                    documentTitle: documentTitle,
                    pageNumber:    pageNumber,
                    text:          chunkText
                ))

                // Overlap: keep last N tokens worth of sentences
                var overlapBuf: [String] = []
                var overlapTok = 0
                for s in buffer.reversed() {
                    let t = max(1, s.count / 4)
                    if overlapTok + t > overlapTokens { break }
                    overlapBuf.insert(s, at: 0)
                    overlapTok += t
                }
                buffer = overlapBuf
                bufTok = overlapTok
            }

            buffer.append(sentence)
            bufTok += sTok
        }

        // Flush remaining
        if !buffer.isEmpty {
            chunks.append(DocumentChunk(
                documentID:    documentID,
                documentTitle: documentTitle,
                pageNumber:    pageNumber,
                text:          buffer.joined(separator: " ")
            ))
        }

        return chunks
    }

    private func splitSentences(_ text: String) -> [String] {
        var sentences: [String] = []
        var current = ""

        for char in text {
            current.append(char)
            if [".", "!", "?", "\n"].contains(char),
               current.trimmingCharacters(in: .whitespacesAndNewlines).count > 20 {
                sentences.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
                current = ""
            }
        }
        if !current.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            sentences.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return sentences.filter { !$0.isEmpty }
    }
}

// MARK: - EmbeddingProvider

/// Protocol for converting text into fixed-dimension float vectors.
///
/// Implement this protocol to plug in your own embedding backend
/// (e.g. OpenAI, Cohere, or a local MLX model). The built-in
/// ``TFIDFEmbeddingProvider`` and ``AutoEmbeddingProvider`` work
/// fully offline with no downloads; ``CoreMLEmbeddingProvider`` runs a
/// dense Core ML model such as multilingual-e5-small.
public protocol EmbeddingProvider: Sendable {
    /// Embed a single string. Returns a normalized float vector.
    func embed(_ text: String) async throws -> [Float]
    /// Batch embed — default implementation calls embed() sequentially.
    func embedBatch(_ texts: [String]) async throws -> [[Float]]
    /// Dimensionality of the output vectors.
    var dimensions: Int { get }

    /// Embed a search query. Asymmetric models (e5) embed queries differently from documents.
    /// Defaults to ``embed(_:)``.
    func embedQuery(_ text: String) async throws -> [Float]
    /// Embed document chunks for indexing. Defaults to ``embedBatch(_:)``.
    func embedDocuments(_ texts: [String]) async throws -> [[Float]]
    /// Name of the vector space: vectors from providers with different identifiers are not comparable,
    /// and ``DocumentLibrary`` re-embeds its index when this changes, so it must stay the same across
    /// launches — e.g. `model@revision`. Defaults to the module-qualified type name plus ``dimensions``.
    var identifier: String { get }
}

public extension EmbeddingProvider {
    func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        var results: [[Float]] = []
        for text in texts {
            try await results.append(embed(text))
        }
        return results
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        try await embed(text)
    }

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        try await embedBatch(texts)
    }

    var identifier: String {
        "\(stableTypeName(of: type(of: self)))/\(dimensions)"
    }
}

/// `String(reflecting:)` of a private or local type embeds `(unknown context at $<address>)`, which moves with
/// ASLR on every launch; without it the default identifier stays stable.
func stableTypeName(of type: Any.Type) -> String {
    String(reflecting: type).replacing(/\(unknown context at \$[0-9a-fA-F]+\)\./, with: "")
}

/// Providers that can say how many inputs they truncated, so indexing can report it.
protocol TruncationReporting: Sendable {
    func truncatedInputCount() async -> Int
}

// MARK: - TFIDFEmbeddingProvider

/// Purely local, zero-download sparse embedding using TF-IDF term weighting.
///
/// Produces 4096-dimensional sparse vectors using a DJB2 hash to map tokens
/// to buckets. IDF weights are updated incrementally via ``updateCorpus(texts:)``.
/// Runs entirely on-device with no network requests.
public actor TFIDFEmbeddingProvider: EmbeddingProvider {

    /// Vocabulary size — fixed dimension for all vectors.
    public static let vocabSize = 4096

    public nonisolated let dimensions = TFIDFEmbeddingProvider.vocabSize
    public static let vectorSpaceIdentifier = "aura.tfidf-hash/\(vocabSize)"
    public nonisolated let identifier = TFIDFEmbeddingProvider.vectorSpaceIdentifier

    /// IDF weights built from all indexed documents.
    private var idf: [Int: Float] = [:]
    /// Total number of documents seen (for IDF calculation).
    private var docCount = 0

    public init() {}

    // MARK: - Corpus update

    /// Call this with all chunk texts after indexing a document to update IDF weights.
    public func updateCorpus(texts: [String]) {
        docCount += texts.count
        var dfCounts: [Int: Int] = [:]
        for text in texts {
            let terms = Set(tokenize(text))
            for term in terms {
                dfCounts[term, default: 0] += 1
            }
        }
        for (term, df) in dfCounts {
            // Smooth IDF: log((N+1)/(df+1)) + 1
            idf[term] = log(Float(docCount + 1) / Float(df + 1)) + 1.0
        }
    }

    // MARK: - EmbeddingProvider

    public func embed(_ text: String) async throws -> [Float] {
        let terms = tokenize(text)
        guard !terms.isEmpty else { return [Float](repeating: 0, count: dimensions) }

        var tf: [Int: Float] = [:]
        for t in terms { tf[t, default: 0] += 1 }
        let total = Float(terms.count)

        var vec = [Float](repeating: 0, count: dimensions)
        // Use bitmask instead of modulo — dimensions is 4096 = 2^12, so mask = 4095
        // This is always non-negative regardless of hash sign or Int.min overflow
        let mask = dimensions - 1
        for (term, count) in tf {
            let bucket   = term & mask
            let tfScore  = count / total
            let idfScore = idf[term] ?? 1.0
            vec[bucket] += tfScore * idfScore
        }
        VectorMath.normalize(&vec)
        return vec
    }

    public func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        var result: [[Float]] = []
        result.reserveCapacity(texts.count)
        for text in texts {
            let vec = try await embed(text)
            result.append(vec)
        }
        return result
    }

    // MARK: - Tokenizer

    private func tokenize(_ text: String) -> [Int] {
        let lower = text.lowercased()
        var tokens: [Int] = []
        var word = ""

        for char in lower {
            if char.isLetter || char.isNumber {
                word.append(char)
            } else if !word.isEmpty {
                if word.count >= 2 && !Self.stopwords.contains(word) {
                    tokens.append(stableHash(word))
                }
                word = ""
            }
        }
        if word.count >= 2 && !Self.stopwords.contains(word) {
            tokens.append(stableHash(word))
        }
        return tokens
    }

    /// DJB2 hash — stable across runs, unlike Swift's randomized String.hashValue.
    private func stableHash(_ s: String) -> Int {
        s.utf8.reduce(5381) { acc, byte in ((acc &<< 5) &+ acc) &+ Int(byte) }
    }

    private static let stopwords: Set<String> = [
        "a","an","the","and","or","but","in","on","at","to","for","of","with",
        "is","are","was","were","be","been","being","have","has","had","do","does",
        "did","will","would","could","should","may","might","that","this","these",
        "those","it","its","i","we","you","he","she","they","my","our","your",
        "his","her","their","as","by","from","up","about","into","than","so","if",
        "no","not","also","just","more","can","all","any","both","each","few"
    ]
}

// MARK: - AutoEmbeddingProvider

/// Recommended default embedding provider for ``DocumentLibrary``.
///
/// ``init()`` uses ``TFIDFEmbeddingProvider``: fully offline, nothing to download.
/// ``init(embeddingModelAt:compiledModelsDirectory:)`` opts into a dense Core ML model
/// (multilingual-e5-small) when its bundle is installed and valid, and falls back to TF-IDF
/// otherwise; ``backendName()`` and ``denseModelProblem`` say which one is used and why.
public actor AutoEmbeddingProvider: EmbeddingProvider, TruncationReporting {

    /// Where ``DocsTab`` looks for, and imports, the e5 bundle:
    /// `Application Support/AuraLocal/embeddings/multilingual-e5-small`.
    public static let defaultModelBundleURL: URL = {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return support.appendingPathComponent("AuraLocal/embeddings/multilingual-e5-small", isDirectory: true)
    }()

    public nonisolated let dimensions: Int
    public nonisolated let identifier: String
    /// Why the requested Core ML bundle is not used; `nil` when it is, or when none was requested.
    public nonisolated let denseModelProblem: String?

    private let tfidf = TFIDFEmbeddingProvider()
    private let dense: CoreMLEmbeddingProvider?

    public init() {
        dense = nil
        dimensions = TFIDFEmbeddingProvider.vocabSize
        identifier = TFIDFEmbeddingProvider.vectorSpaceIdentifier
        denseModelProblem = nil
    }

    /// Uses the Core ML embedding bundle at `bundleURL` when it is present and valid (checked on
    /// disk, without loading the model), TF-IDF otherwise. Call ``warmUp()`` before use.
    public init(embeddingModelAt bundleURL: URL?, compiledModelsDirectory: URL? = nil) {
        var problem: String?
        var provider: CoreMLEmbeddingProvider?
        if let bundleURL {
            do {
                provider = try CoreMLEmbeddingProvider(bundleAt: bundleURL,
                                                       compiledModelsDirectory: compiledModelsDirectory)
            } catch {
                problem = error.localizedDescription
            }
        }
        dense = provider
        dimensions = provider?.dimensions ?? TFIDFEmbeddingProvider.vocabSize
        identifier = provider?.identifier ?? TFIDFEmbeddingProvider.vectorSpaceIdentifier
        denseModelProblem = problem
    }

    /// True when a Core ML model, not TF-IDF, produces the vectors.
    public nonisolated var usesDenseModel: Bool { dense != nil }

    public func updateCorpus(texts: [String]) async {
        await tfidf.updateCorpus(texts: texts)
    }

    public func embed(_ text: String) async throws -> [Float] {
        if let dense { return try await dense.embed(text) }
        return try await tfidf.embed(text)
    }

    public func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        if let dense { return try await dense.embedBatch(texts) }
        return try await tfidf.embedBatch(texts)
    }

    public func embedQuery(_ text: String) async throws -> [Float] {
        if let dense { return try await dense.embedQuery(text) }
        return try await tfidf.embed(text)
    }

    /// Compiles and loads the Core ML model and runs it once per sequence bucket; a no-op for
    /// TF-IDF. The first Neural Engine load compiles on-device and can take tens of seconds.
    @discardableResult
    public func warmUp() async throws -> Duration {
        guard let dense else { return .zero }
        return try await dense.warmUp()
    }

    /// Inputs longer than the model's token limit that were truncated so far (0 for TF-IDF).
    public func truncatedInputCount() async -> Int {
        guard let dense else { return 0 }
        return await dense.truncatedInputCount()
    }

    public func backendName() -> String {
        guard let dense else { return "TF-IDF (local, no model required)" }
        return "\(dense.modelName) (Core ML, on-device)"
    }
}

// MARK: - Vector Math

import Accelerate

enum VectorMath {
    /// Cosine similarity between two vectors using Accelerate SIMD.
    /// Returns 0 if either is zero-length.
    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        let n = vDSP_Length(a.count)
        var dot: Float = 0
        var na: Float = 0
        var nb: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, n)
        vDSP_dotpr(a, 1, a, 1, &na,  n)
        vDSP_dotpr(b, 1, b, 1, &nb,  n)
        let denom = sqrt(na) * sqrt(nb)
        return denom > 0 ? dot / denom : 0
    }

    /// L2-normalize a vector in place using Accelerate SIMD.
    static func normalize(_ v: inout [Float]) {
        let n = vDSP_Length(v.count)
        var normSq: Float = 0
        vDSP_dotpr(v, 1, v, 1, &normSq, n)
        let norm = sqrt(normSq)
        guard norm > 0 else { return }
        var divisor = norm
        vDSP_vsdiv(v, 1, &divisor, &v, 1, n)
    }
}
