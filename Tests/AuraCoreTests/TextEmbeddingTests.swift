import XCTest
import CoreML
@testable import AuraCore

final class TextEmbeddingTests: XCTestCase {

    private typealias ToolError = CoreMLTextEmbeddingTool.ToolError

    private static var validManifest: [String: Any] { [
        "schema": "aura.text-embedding/1",
        "model_id": "intfloat/multilingual-e5-small",
        "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "model_file": "MultilingualE5Small.mlpackage",
        "input_name": "input_ids",
        "output_name": "last_hidden_state",
        "buckets": [64, 128, 256, 512],
        "pad_token_id": 1,
        "pooling": "mean",
        "normalize": true,
        "dimensions": 384,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "max_tokens": 512,
        "license": "MIT",
        "parity_min_cosine": ["CPU_ONLY": 0.9999],
    ] }

    private static func manifestData(changing changes: [String: Any] = [:], removing removed: [String] = []) throws
        -> Data {
        var fields = validManifest.merging(changes) { _, new in new }
        removed.forEach { fields[$0] = nil }
        return try JSONSerialization.data(withJSONObject: fields)
    }

    private static func invalidReason(_ data: Data) -> String? {
        do {
            _ = try TextEmbeddingManifest.decode(data)
            return nil
        } catch ToolError.invalidManifest(let reason) {
            return reason
        } catch {
            return "unexpected \(error)"
        }
    }

    private var workDirectory: URL!

    override func setUpWithError() throws {
        workDirectory = try MLTestFixtures.makeWorkDirectory(for: "TextEmbeddingTests")
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: workDirectory)
    }

    // MARK: - Manifest

    func testManifestDecodesSnakeCaseKeysAndIgnoresUnknownOnes() throws {
        let manifest = try TextEmbeddingManifest.decode(try Self.manifestData())

        XCTAssertEqual(manifest.modelID, "intfloat/multilingual-e5-small")
        XCTAssertEqual(manifest.modelFile, "MultilingualE5Small.mlpackage")
        XCTAssertEqual(manifest.inputName, "input_ids")
        XCTAssertEqual(manifest.outputName, "last_hidden_state")
        XCTAssertEqual(manifest.buckets, [64, 128, 256, 512])
        XCTAssertEqual(manifest.padTokenID, 1)
        XCTAssertEqual(manifest.dimensions, 384)
        XCTAssertEqual(manifest.maxTokens, 512)
        XCTAssertTrue(manifest.normalize)
        XCTAssertEqual(manifest.identifier, "intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3")
    }

    func testManifestPrefixDependsOnRole() throws {
        let manifest = try TextEmbeddingManifest.decode(try Self.manifestData())

        XCTAssertEqual(manifest.prefix(for: .query), "query: ")
        XCTAssertEqual(manifest.prefix(for: .passage), "passage: ")
        XCTAssertEqual(manifest.prefix(for: .raw), "")
    }

    func testManifestWithMissingKeyNamesTheKey() throws {
        let reason = Self.invalidReason(try Self.manifestData(removing: ["pad_token_id"]))

        XCTAssertEqual(reason, "missing “pad_token_id”.")
    }

    func testManifestWithWrongTypeIsInvalid() throws {
        let reason = Self.invalidReason(try Self.manifestData(changing: ["buckets": "64,128"]))

        XCTAssertEqual(reason, "“buckets” has the wrong type.")
    }

    func testMalformedManifestJSONIsInvalid() {
        XCTAssertNotNil(Self.invalidReason(Data("{ not json".utf8)))
    }

    func testManifestValidationRejectsEachBadField() throws {
        let cases: [(String, Any, String)] = [
            ("schema", "aura.text-embedding/2", "unsupported schema"),
            ("pooling", "cls", "unsupported pooling"),
            ("buckets", [128, 64], "“buckets”"),
            ("buckets", [64, 64], "“buckets”"),
            ("buckets", [Int](), "“buckets”"),
            ("buckets", [0, 64], "“buckets”"),
            ("max_tokens", 1024, "“max_tokens”"),
            ("max_tokens", 1, "“max_tokens”"),
            ("dimensions", 0, "“dimensions”"),
            ("pad_token_id", -1, "“pad_token_id”"),
            ("model_id", " ", "“model_id” is empty"),
            ("model_file", "../model.mlpackage", "“model_file”"),
            ("model_file", "weights.bin", "“model_file”"),
        ]
        for (key, value, expected) in cases {
            let reason = Self.invalidReason(try Self.manifestData(changing: [key: value]))
            XCTAssertTrue(reason?.contains(expected) == true, "\(key)=\(value): \(reason ?? "accepted")")
        }
    }

    // MARK: - Bundle checks

    func testMissingBundleIsUnavailableWithReason() async {
        let tool = CoreMLTextEmbeddingTool(bundleAt: workDirectory.appendingPathComponent("absent"))

        let availability = await tool.availability()

        XCTAssertTrue(availability.reason?.contains("No embedding model bundle") == true, "\(availability)")
        XCTAssertNil(tool.manifest)
        XCTAssertEqual(tool.dimensions, 0)
        XCTAssertEqual(tool.id, "coreml.text-embedding.absent")
        do {
            _ = try await tool.embed("hola")
            XCTFail("expected an error")
        } catch ToolError.bundleNotFound(let path) {
            XCTAssertTrue(path.hasSuffix("absent"), path)
        } catch {
            XCTFail("\(error)")
        }
    }

    func testBundleWithoutTokenizerNamesTheMissingFile() throws {
        let bundle = try makeBundle(named: "no-tokenizer", tokenizerFiles: [])

        XCTAssertThrowsError(try CoreMLTextEmbeddingTool.validateBundle(at: bundle)) { error in
            guard case ToolError.tokenizerNotFound(let path) = error else {
                return XCTFail("\(error)")
            }
            XCTAssertTrue(path.hasSuffix("tokenizer.json"), path)
        }
    }

    func testBundleWithoutModelFileIsUnavailable() async throws {
        let bundle = try makeBundle(named: "no-model", withModel: false)

        let availability = await CoreMLTextEmbeddingTool(bundleAt: bundle).availability()

        XCTAssertTrue(availability.reason?.contains("no model file") == true, "\(availability)")
    }

    func testEmbeddingWithAnInvalidManifestThrowsItsReason() async throws {
        let bundle = try makeBundle(named: "bad-manifest", changing: ["pooling": "cls"])
        let tool = CoreMLTextEmbeddingTool(bundleAt: bundle)

        do {
            _ = try await tool.embed("hola")
            XCTFail("expected an error")
        } catch ToolError.invalidManifest(let reason) {
            XCTAssertTrue(reason.contains("pooling"), reason)
        }
        let availability = await tool.availability()
        XCTAssertTrue(availability.reason?.contains("pooling") == true, "\(availability)")
    }

    func testCompleteBundleValidatesAndReportsItsIdentity() throws {
        let bundle = try makeBundle(named: "complete")

        let manifest = try CoreMLTextEmbeddingTool.validateBundle(at: bundle)
        let tool = CoreMLTextEmbeddingTool(bundleAt: bundle)

        XCTAssertEqual(tool.identifier, manifest.identifier)
        XCTAssertEqual(tool.dimensions, 384)
    }

    private func makeBundle(
        named name: String,
        changing changes: [String: Any] = [:],
        withModel: Bool = true,
        tokenizerFiles: [String] = CoreMLTextEmbeddingTool.requiredTokenizerFiles
    ) throws -> URL {
        let bundle = workDirectory.appendingPathComponent(name, isDirectory: true)
        let files = FileManager.default
        try files.createDirectory(at: bundle, withIntermediateDirectories: true)
        try Self.manifestData(changing: changes).write(to: bundle.appendingPathComponent(TextEmbeddingManifest.fileName))
        if withModel {
            try files.createDirectory(at: bundle.appendingPathComponent("MultilingualE5Small.mlpackage"),
                                      withIntermediateDirectories: true)
        }
        try tokenizerFiles.forEach { try Data("{}".utf8).write(to: bundle.appendingPathComponent($0)) }
        return bundle
    }

    // MARK: - Pipeline steps

    func testBucketIsTheSmallestThatFits() {
        let buckets = [64, 128, 256, 512]

        XCTAssertEqual(TextEmbeddingPipeline.bucket(forTokenCount: 2, in: buckets), 64)
        XCTAssertEqual(TextEmbeddingPipeline.bucket(forTokenCount: 64, in: buckets), 64)
        XCTAssertEqual(TextEmbeddingPipeline.bucket(forTokenCount: 65, in: buckets), 128)
        XCTAssertEqual(TextEmbeddingPipeline.bucket(forTokenCount: 300, in: buckets), 512)
        XCTAssertEqual(TextEmbeddingPipeline.bucket(forTokenCount: 512, in: buckets), 512)
        XCTAssertNil(TextEmbeddingPipeline.bucket(forTokenCount: 513, in: buckets))
    }

    func testTruncationKeepsTheClosingToken() {
        let ids = [0] + Array(5..<605) + [2]

        let cut = TextEmbeddingPipeline.truncate(ids, maxTokens: 512)

        XCTAssertEqual(cut.count, 512)
        XCTAssertEqual(cut.first, 0)
        XCTAssertEqual(cut.last, 2)
        XCTAssertEqual(Array(cut.dropLast()), Array(ids.prefix(511)))
    }

    func testInputWithinTheLimitIsNotTruncated() {
        let ids = [0, 41, 1294, 2]

        XCTAssertEqual(TextEmbeddingPipeline.truncate(ids, maxTokens: 512), ids)
        XCTAssertEqual(TextEmbeddingPipeline.truncate(ids, maxTokens: 4), ids)
    }

    func testTruncatedTokensReportTheirOriginalLength() {
        let tokens = TextEmbeddingTokens(ids: [0, 7, 2], originalCount: 700)

        XCTAssertTrue(tokens.isTruncated)
        XCTAssertFalse(TextEmbeddingTokens(ids: [0, 2], originalCount: 2).isTruncated)
    }

    func testPaddingFillsTheBucketWithThePadID() {
        let padded = TextEmbeddingPipeline.pad([0, 41, 2], to: 8, padTokenID: 1)

        XCTAssertEqual(padded, [0, 41, 2, 1, 1, 1, 1, 1])
    }

    func testInputArrayHoldsTheIDsInOrder() throws {
        let array = try TextEmbeddingPipeline.makeInputArray([0, 41, 2, 1])

        XCTAssertEqual(array.shape, [1, 4])
        XCTAssertEqual(array.dataType, .int32)
        XCTAssertEqual(MLShapedArray<Int32>(converting: array).scalars, [0, 41, 2, 1])
    }

    func testMeanPoolingIgnoresPaddingAndNormalizes() throws {
        let states = MLMultiArray(MLShapedArray<Float>(
            scalars: [1, 2, 2,
                      3, 2, 0,
                      2, 2, 1,
                      100, 100, 100],
            shape: [1, 4, 3]))

        let vector = try TextEmbeddingPipeline.meanPool(states, tokenIDs: [0, 7, 2, 1], padTokenID: 1, normalize: true)

        assertVectorsEqual(vector, [2 / 3, 2 / 3, 1 / 3], accuracy: 1e-6)
    }

    func testMeanPoolingWithoutNormalizationReturnsTheMean() throws {
        let states = MLMultiArray(MLShapedArray<Double>(scalars: [1, 4, 3, 8, 50, 50], shape: [3, 2]))

        let vector = try TextEmbeddingPipeline.meanPool(states, tokenIDs: [0, 2, 1], padTokenID: 1, normalize: false)

        assertVectorsEqual(vector, [2, 6], accuracy: 1e-6)
    }

    func testMeanPoolingHonoursPaddedRowStrides() throws {
        let values: [Float] = [3, 0, 4, -1, 99, 99, 99, -1]
        let storage = UnsafeMutablePointer<Float>.allocate(capacity: values.count)
        storage.initialize(from: values, count: values.count)
        let states = try MLMultiArray(dataPointer: storage, shape: [1, 2, 3], dataType: .float32,
                                      strides: [8, 4, 1], deallocator: { $0.deallocate() })

        let vector = try TextEmbeddingPipeline.meanPool(states, tokenIDs: [0, 1], padTokenID: 1, normalize: true)

        assertVectorsEqual(vector, [0.6, 0, 0.8], accuracy: 1e-6)
    }

    func testMeanPoolingRejectsABatchOfMoreThanOne() {
        let states = MLMultiArray(MLShapedArray<Float>(repeating: 0, shape: [2, 4, 3]))

        XCTAssertThrowsError(try TextEmbeddingPipeline.meanPool(states, tokenIDs: [0, 2, 1, 1], padTokenID: 1,
                                                                 normalize: true))
    }

    func testPreprocessingComposesToNFCAndKeepsZeroWidthCharacters() {
        let decomposed = "pingu\u{0308}ino cafe\u{0301}"
        let zeroWidth = "a\u{200B}b\u{200C}c\u{200D}d\u{2060}e\u{FEFF}f"

        XCTAssertEqual(Array(TextEmbeddingPipeline.preprocess(decomposed).unicodeScalars),
                       Array("ping\u{00FC}ino caf\u{00E9}".unicodeScalars))
        XCTAssertEqual(Array(TextEmbeddingPipeline.preprocess(zeroWidth).unicodeScalars), Array(zeroWidth.unicodeScalars))
        XCTAssertEqual(TextEmbeddingPipeline.preprocess("query: ¿Qué tal?"), "query: ¿Qué tal?")
    }

    // MARK: - Index identity

    private struct ReconciliationCase {
        let stored: EmbeddingIndexIdentity?
        let configured: EmbeddingIndexIdentity
        let lengths: Set<Int>
        let count: Int
        let expected: EmbeddingIndexIdentity.Reconciliation
    }

    func testIndexReconciliationAdoptsOnlyEmptyOrPreTrackingTFIDFIndexes() {
        let e5 = EmbeddingIndexIdentity(identifier: "intfloat/multilingual-e5-small@614241f", dimensions: 384)
        let miniLM = EmbeddingIndexIdentity(identifier: "someone/MiniLM-L12", dimensions: 384)
        let tfidf = EmbeddingIndexIdentity(identifier: EmbeddingIndexIdentity.preTrackingIdentifier, dimensions: 4096)
        let cases = [
            ReconciliationCase(stored: e5, configured: e5, lengths: [384], count: 10, expected: .upToDate),
            ReconciliationCase(stored: tfidf, configured: e5, lengths: [4096], count: 10, expected: .reembed),
            ReconciliationCase(stored: e5, configured: tfidf, lengths: [384], count: 10, expected: .reembed),
            ReconciliationCase(stored: miniLM, configured: e5, lengths: [384], count: 10, expected: .reembed),
            ReconciliationCase(stored: nil, configured: tfidf, lengths: [4096], count: 10, expected: .adopt),
            ReconciliationCase(stored: nil, configured: e5, lengths: [384], count: 10, expected: .reembed),
            ReconciliationCase(stored: nil, configured: e5, lengths: [4096], count: 10, expected: .reembed),
            ReconciliationCase(stored: nil, configured: tfidf, lengths: [4096, 0], count: 10, expected: .reembed),
            ReconciliationCase(stored: nil, configured: e5, lengths: [], count: 0, expected: .adopt),
            ReconciliationCase(stored: tfidf, configured: e5, lengths: [], count: 0, expected: .adopt),
        ]
        for item in cases {
            let action = EmbeddingIndexIdentity.reconciliation(stored: item.stored, configured: item.configured,
                                                               storedVectorLengths: item.lengths,
                                                               storedVectorCount: item.count)
            XCTAssertEqual(action, item.expected, "\(item.stored as Any) → \(item.configured), \(item.lengths)")
        }
    }

    // MARK: - Live model (needs the bundle; see scripts/embeddings)

    private struct ReferenceFixtures: Decodable {
        let sentences: [String]
        let ref: [[Float]]
        let ids: [String: [[Int32]]]
    }

    // Opt-in: the first run compiles ~225 MB for the Neural Engine and takes about half a minute.
    private static let bundleURL = ProcessInfo.processInfo.environment["AURA_E5_BUNDLE"]
        .map { URL(fileURLWithPath: $0, isDirectory: true) }

    private static let fixturesURL = ProcessInfo.processInfo.environment["AURA_E5_FIXTURES"]
        .map { URL(fileURLWithPath: $0) }

    /// A stable path lets the Neural Engine reuse its compiled model while AURA_E5_KEEP_COMPILED is set.
    private static let compiledModels = FileManager.default.temporaryDirectory
        .appendingPathComponent("AuraLocalTests/CompiledModels", isDirectory: true)

    private static let liveTool = bundleURL.map {
        CoreMLTextEmbeddingTool(bundleAt: $0, compiledModelsDirectory: compiledModels)
    }

    override static func tearDown() {
        if bundleURL != nil, ProcessInfo.processInfo.environment["AURA_E5_KEEP_COMPILED"] == nil {
            try? FileManager.default.removeItem(at: compiledModels)
        }
        super.tearDown()
    }

    private func requireLiveTool() async throws -> CoreMLTextEmbeddingTool {
        guard let tool = Self.liveTool else {
            throw XCTSkip("Live e5 tests are opt-in: set AURA_E5_BUNDLE to the bundle folder")
        }
        let availability = await tool.availability()
        guard availability.isAvailable else {
            throw XCTSkip("e5 bundle unavailable (\(availability.reason ?? "?")); check AURA_E5_BUNDLE")
        }
        return tool
    }

    private func requireFixtures() throws -> ReferenceFixtures {
        guard let path = Self.fixturesURL?.path, let data = FileManager.default.contents(atPath: path) else {
            throw XCTSkip("No reference fixtures; set AURA_E5_FIXTURES")
        }
        return try JSONDecoder().decode(ReferenceFixtures.self, from: data)
    }

    private static func cosine(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(0) { $0 + $1.0 * $1.1 }
    }

    func testLiveTokenIDsMatchThePythonTokenizer() async throws {
        let tool = try await requireLiveTool()
        let fixtures = try requireFixtures()
        let padded = try XCTUnwrap(fixtures.ids["512"])
        let reference = padded.map { row in row.prefix { $0 != 1 }.map { Int($0) } }
        XCTAssertEqual(reference.count, fixtures.sentences.count)

        for (sentence, expected) in zip(fixtures.sentences, reference) {
            let tokens = try await tool.tokenize(sentence, role: .raw)
            XCTAssertEqual(tokens.ids, expected, sentence)
        }
    }

    /// Expected ids come from Python `tokenizers` 0.23.2 on the bundle's tokenizer.json, which maps ZWSP, ZWNJ,
    /// ZWJ and BOM to a space and keeps U+2060.
    func testLiveZeroWidthCharactersTokenizeLikePython() async throws {
        let tool = try await requireLiveTool()
        let expected: [(String, [Int])] = [
            ("zero\u{200B}width", [0, 45234, 6, 146984, 2]),
            ("\u{0645}\u{06CC}\u{200C}\u{062E}\u{0648}\u{0627}\u{0647}\u{0645}", [0, 383, 140113, 2]),
            ("a\u{200D}b", [0, 10, 876, 2]),
            ("a\u{2060}b", [0, 10, 243465, 275, 2]),
            ("\u{FEFF}Invoice number", [0, 360, 965, 2980, 14012, 2]),
        ]
        for (text, ids) in expected {
            let tokens = try await tool.tokenize(text, role: .raw)
            XCTAssertEqual(tokens.ids, ids, Self.scalarDump(text))
        }
    }

    private static func scalarDump(_ text: String) -> String {
        text.unicodeScalars.map { String($0.value, radix: 16) }.joined(separator: " ")
    }

    func testLiveEmbeddingsMatchSentenceTransformers() async throws {
        let tool = try await requireLiveTool()
        let fixtures = try requireFixtures()

        let embeddings = try await tool.embed(fixtures.sentences, role: .raw)

        let cosines = zip(embeddings, fixtures.ref).map { Self.cosine($0.vector, $1) }
        XCTAssertEqual(cosines.count, fixtures.sentences.count)
        XCTAssertGreaterThanOrEqual(cosines.min() ?? 0, 0.999, "\(cosines)")
        XCTAssertTrue(embeddings.allSatisfy { !$0.isTruncated && $0.vector.count == 384 })
    }

    func testLiveSpanishQueryRanksTheSpanishPassageAboveTheEnglishOne() async throws {
        let tool = try await requireLiveTool()

        let query = try await tool.embed("¿Cómo conecto el iPhone al Mac por USB para calcular embeddings?", role: .query)
        let passages = try await tool.embed([
            "Configura el iPhone 12 como nodo ligero de embeddings conectado por USB al Mac.",
            "As a general guideline, the CDC's average requirement of protein for women is 46 grams per day.",
        ], role: .passage)

        let spanish = Self.cosine(query.vector, passages[0].vector)
        let english = Self.cosine(query.vector, passages[1].vector)
        XCTAssertGreaterThan(spanish, english)
    }

    func testLiveLongInputIsTruncatedToTheTokenLimitAndCounted() async throws {
        let tool = try await requireLiveTool()
        let long = Array(repeating: "La cláusula del contrato fija el pago a treinta días.", count: 80)
            .joined(separator: " ")
        let before = await tool.truncatedInputCount()

        let tokens = try await tool.tokenize(long)
        let embedding = try await tool.embed(long)

        XCTAssertGreaterThan(tokens.originalCount, 512)
        XCTAssertEqual(tokens.ids.count, 512)
        XCTAssertEqual(tokens.ids.last, 2)
        XCTAssertTrue(embedding.isTruncated)
        XCTAssertEqual(embedding.tokenCount, 512)
        XCTAssertEqual(embedding.vector.reduce(0) { $0 + $1 * $1 }, 1, accuracy: 1e-4)
        let after = await tool.truncatedInputCount()
        XCTAssertEqual(after - before, 1)
    }

    func testLiveWarmUpLoadsTheModel() async throws {
        let tool = try await requireLiveTool()

        let elapsed = try await tool.warmUp()

        XCTAssertGreaterThan(elapsed, .zero)
        let availability = await tool.availability()
        XCTAssertTrue(availability.isAvailable)
    }
}

private func assertVectorsEqual(
    _ actual: [Float],
    _ expected: [Float],
    accuracy: Float,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    XCTAssertEqual(actual.count, expected.count, "\(actual)", file: file, line: line)
    let worst = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
    XCTAssertLessThanOrEqual(worst, accuracy, "\(actual) vs \(expected)", file: file, line: line)
}
