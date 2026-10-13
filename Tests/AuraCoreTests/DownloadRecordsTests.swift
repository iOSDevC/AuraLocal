import XCTest
@testable import AuraCore

final class DownloadRecordsTests: XCTestCase {
    private var suiteName = ""
    private var defaults: UserDefaults!
    private var records: DownloadRecords!
    private var registeredIDs: [String] = []
    private var tempRoot: URL?

    override func setUpWithError() throws {
        suiteName = "DownloadRecordsTests-\(UUID().uuidString)"
        defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        records = DownloadRecords(defaults: defaults)
    }

    override func tearDown() {
        registeredIDs.forEach { ModelRegistry.shared.unregister(id: $0) }
        defaults.removePersistentDomain(forName: suiteName)
        if let tempRoot { try? FileManager.default.removeItem(at: tempRoot) }
        super.tearDown()
    }

    // MARK: Fixtures

    private static let stubSession: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [DownloadStubProtocol.self]
        return URLSession(configuration: configuration)
    }()

    /// A unique sized model, built the way search results are.
    private func makeModel(_ suffix: String, format: ModelFormat = .gguf, category: Model.Category = .text,
                           repo: String? = nil, file: String? = nil) throws -> Model {
        let tag = UUID().uuidString.prefix(8).lowercased()
        let repoID = repo ?? "test-org/records-\(suffix)-\(tag)"
        let entry = CatalogEntry(
            id: "test_records_\(suffix)_\(tag)", repoID: repoID, displayName: "Test \(suffix)",
            modelCategory: category, weightFormat: format, approximateSizeMB: 412, isUncensored: false,
            ggufFilename: format == .gguf ? (file ?? "model-Q4_K_M.gguf") : nil, numLayers: 24,
            kvHeads: format == .gguf ? 2 : 0, headDim: format == .gguf ? 64 : 0, maxContextLength: 4096)
        let model = try XCTUnwrap(entry.decodedModel())
        registeredIDs.append(model.id)
        return model
    }

    private func registered(_ model: Model) -> Bool {
        ModelRegistry.shared.model(id: model.id) != nil
    }

    // MARK: Records

    func testRememberRoundTripsSizedModel() throws {
        let model = try makeModel("roundtrip")
        records.remember(model)

        let stored = try XCTUnwrap(records.model(repoID: model.repoID, ggufFilename: model.ggufFilename))
        XCTAssertEqual(stored.id, model.id)
        XCTAssertEqual(stored.repoID, model.repoID)
        XCTAssertEqual(stored.format, .gguf)
        XCTAssertEqual(stored.ggufFilename, "model-Q4_K_M.gguf")
        XCTAssertEqual(stored.approximateSizeMB, 412)
        XCTAssertEqual(stored.numLayers, 24)
        XCTAssertEqual(stored.kvHeads, 2)
        XCTAssertEqual(stored.headDim, 64)
        XCTAssertEqual(stored.maxContextLength, 4096)
        XCTAssertEqual(stored.category, .text)
        XCTAssertTrue(records.contains(model))
        XCTAssertEqual(records.models.map(\.id), [model.id])
    }

    func testKeyIsFileIdentity() throws {
        let q4 = try makeModel("quant", file: "m-Q4_K_M.gguf")
        let q8 = try makeModel("quant", repo: q4.repoID, file: "m-Q8_0.gguf")
        records.remember(q4)
        records.remember(q8)
        XCTAssertEqual(Set(records.models.map(\.id)), [q4.id, q8.id])

        let sameFile = try makeModel("renamed", repo: q4.repoID, file: "m-Q4_K_M.gguf")
        XCTAssertEqual(DownloadRecords.key(for: sameFile), DownloadRecords.key(for: q4))
        records.remember(sameFile)

        XCTAssertEqual(Set(records.models.map(\.id)), [sameFile.id, q8.id])
        XCTAssertEqual(records.model(repoID: q4.repoID, ggufFilename: "m-Q4_K_M.gguf")?.id, sameFile.id)
    }

    func testLegacyKeyIsCopiedOnce() throws {
        let old = try makeModel("legacy")
        let data = try JSONEncoder().encode(old)
        defaults.set([DownloadRecords.key(for: old): data], forKey: DownloadRecords.legacyKey)

        XCTAssertEqual(records.models.map(\.id), [old.id])
        XCTAssertNotNil(defaults.dictionary(forKey: DownloadRecords.defaultsKey))

        let later = try makeModel("legacy-later")
        let laterData = try JSONEncoder().encode(later)
        defaults.set([DownloadRecords.key(for: old): data, DownloadRecords.key(for: later): laterData],
                     forKey: DownloadRecords.legacyKey)
        XCTAssertEqual(records.models.map(\.id), [old.id])

        records.forget(old)
        XCTAssertTrue(records.models.isEmpty)
    }

    func testConcurrentRemembersKeepEveryRecord() throws {
        let models = try (0..<64).map { try makeModel("concurrent\($0)") }
        let records = try XCTUnwrap(records)
        DispatchQueue.concurrentPerform(iterations: models.count) { records.remember(models[$0]) }
        XCTAssertEqual(Set(records.models.map(\.id)), Set(models.map(\.id)))
    }

    // MARK: Registry

    func testRegisterDownloadedAddsOnDiskModelsToPickers() throws {
        let text = try makeModel("text")
        let vision = try makeModel("vision", format: .mlx, category: .vision)
        let missing = try makeModel("missing")
        for model in [text, vision, missing] {
            records.remember(model)
        }

        let listed = records.registerDownloaded(isOnDisk: { $0.id != missing.id }, bytesOnDisk: { _ in 1 })

        XCTAssertEqual(Set(listed.map(\.id)), [text.id, vision.id])
        XCTAssertTrue(Model.textModels.contains { $0.id == text.id })
        XCTAssertTrue(Model.ggufModels.contains { $0.id == text.id })
        XCTAssertTrue(Model.visionModels.contains { $0.id == vision.id })
        XCTAssertFalse(registered(missing))
    }

    func testRegisterDownloadedPrunesRecordsWithNothingOnDisk() throws {
        let purged = try makeModel("purged")
        let partial = try makeModel("partial")
        records.remember(purged)
        records.remember(partial)

        records.registerDownloaded(isOnDisk: { _ in false }, bytesOnDisk: { $0.id == partial.id ? 2048 : 0 })

        XCTAssertEqual(records.models.map(\.id), [partial.id])
        XCTAssertFalse(registered(purged))
        XCTAssertFalse(registered(partial))
    }

    func testRegisterSkipsDuplicateOfCatalogEntry() throws {
        let catalog = try XCTUnwrap(ModelRegistry.shared.models.first { $0.format == .gguf && $0.category == .text })
        let duplicate = try makeModel("duplicate", repo: catalog.repoID, file: catalog.ggufFilename)
        records.remember(duplicate)

        let listed = records.registerDownloaded(isOnDisk: { _ in true }, bytesOnDisk: { _ in 1 })

        XCTAssertTrue(listed.isEmpty)
        XCTAssertFalse(registered(duplicate))
        let sameFile = Model.ggufModels.filter {
            $0.repoID == catalog.repoID && $0.ggufFilename == catalog.ggufFilename
        }
        XCTAssertEqual(sameFile.map(\.id), [catalog.id])
        XCTAssertFalse(records.registerIfDownloaded(duplicate, isOnDisk: { _ in true }))
    }

    func testRemoveForgetsAndUnregisters() throws {
        let model = try makeModel("remove")
        records.remember(model)
        XCTAssertTrue(records.registerIfDownloaded(model, isOnDisk: { _ in true }))
        XCTAssertTrue(Model.textModels.contains { $0.id == model.id })

        records.remove(model)

        XCTAssertFalse(Model.textModels.contains { $0.id == model.id })
        XCTAssertFalse(ModelRegistry.shared.models.contains { $0.id == model.id })
        XCTAssertNil(records.model(repoID: model.repoID, ggufFilename: model.ggufFilename))
    }

    func testUnregisterKeepsCatalogEntries() throws {
        let catalog = try XCTUnwrap(ModelRegistry.shared.models.first)

        XCTAssertFalse(ModelRegistry.shared.unregister(id: catalog.id))

        XCTAssertNotNil(ModelRegistry.shared.model(id: catalog.id))
        XCTAssertTrue(ModelRegistry.shared.models.contains { $0.id == catalog.id })
    }

    /// Offline: a stubbed Hugging Face serves the file into a temporary Caches root.
    func testFinishedStubDownloadIsRegistered() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("DownloadRecordsTests-\(UUID().uuidString)", isDirectory: true)
        tempRoot = root
        let model = try makeModel("stub", file: "tiny-Q4_0.gguf")
        let body = Data(repeating: 7, count: 4096)
        DownloadStubs.shared.serve("\(model.repoID)/resolve/main/tiny-Q4_0.gguf", body: body)
        let transfer = HubFileTransfer(session: Self.stubSession, auth: NoDownloadAuth(), freeSpace: { _ in nil },
                                       retryDelay: { _ in .zero }, progressInterval: .milliseconds(10))
        records.remember(model)

        let file = try await ModelDownloads.download(model, transfer: transfer, root: root) { _, _ in }

        XCTAssertEqual(try Data(contentsOf: file), body)
        let onDisk: (Model) -> Bool = { candidate in
            guard let target = try? ModelDownloads.ggufTarget(for: candidate, root: root) else { return false }
            return FileManager.default.fileExists(atPath: target.file.path)
        }
        XCTAssertTrue(records.registerIfDownloaded(model, isOnDisk: onDisk))
        XCTAssertTrue(Model.textModels.contains { $0.id == model.id })
    }
}
