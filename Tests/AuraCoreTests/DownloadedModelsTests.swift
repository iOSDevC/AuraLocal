import XCTest
@testable import AuraCore

/// `DownloadedModels.scan` — enumerate the `.gguf` models already on disk under the model cache root.
/// Pure + deterministic: the scan takes an injected root (a temp dir), so nothing touches the real caches.
final class DownloadedModelsTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("DownloadedModelsTests-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        // Clean up the whole temp tree (parent of `models`).
        try? FileManager.default.removeItem(at: root.deletingLastPathComponent())
    }

    /// A file at `<root>/org/repo/a.gguf` is found with the right repoID/filename/size; a stray non-gguf and
    /// a URLSession-style temp are ignored.
    func testFindsOneGGUFAndDerivesRepoID() throws {
        let repoDir = root.appendingPathComponent("org/repo", isDirectory: true)
        try FileManager.default.createDirectory(at: repoDir, withIntermediateDirectories: true)

        let payload = Data(count: 4096)
        let gguf = repoDir.appendingPathComponent("a.gguf")
        try payload.write(to: gguf)
        // Stray non-gguf + a partial-download temp: both must be ignored.
        try Data("not a model".utf8).write(to: repoDir.appendingPathComponent("README.md"))
        try Data(count: 128).write(to: repoDir.appendingPathComponent("CFNetworkDownload_abc.tmp"))

        let found = DownloadedModels.scan(root: root, fileManager: .default)

        XCTAssertEqual(found.count, 1)
        let model = try XCTUnwrap(found.first)
        XCTAssertEqual(model.repoID, "org/repo")
        XCTAssertEqual(model.filename, "a.gguf")
        XCTAssertEqual(model.fileURL.lastPathComponent, "a.gguf")
        XCTAssertEqual(model.sizeBytes, 4096)
        XCTAssertEqual(model.id, model.fileURL.path)
        XCTAssertFalse(model.sizeText.isEmpty)
    }

    /// An empty models root → no results (and no crash).
    func testEmptyRootReturnsEmpty() {
        XCTAssertTrue(DownloadedModels.scan(root: root, fileManager: .default).isEmpty)
    }

    /// A missing root → `[]` rather than throwing.
    func testMissingRootReturnsEmpty() {
        let missing = root.appendingPathComponent("does-not-exist", isDirectory: true)
        XCTAssertTrue(DownloadedModels.scan(root: missing, fileManager: .default).isEmpty)
    }

    /// A flat `.gguf` directly under the root (no repo folder) → empty repoID, filename still populated.
    func testFlatFileHasEmptyRepoID() throws {
        let flat = root.appendingPathComponent("loose.gguf")
        try Data(count: 10).write(to: flat)

        let found = DownloadedModels.scan(root: root, fileManager: .default)
        XCTAssertEqual(found.count, 1)
        XCTAssertEqual(found.first?.repoID, "")
        XCTAssertEqual(found.first?.filename, "loose.gguf")
        XCTAssertEqual(found.first?.displayName, "loose.gguf")
    }

    /// Nested subfolders under a repo (e.g. `org/repo/MTP/x.gguf`) fold into the repoID path.
    func testSubfolderFoldsIntoRepoID() throws {
        let deep = root.appendingPathComponent("org/repo/MTP", isDirectory: true)
        try FileManager.default.createDirectory(at: deep, withIntermediateDirectories: true)
        try Data(count: 1).write(to: deep.appendingPathComponent("x.Q4_K_M.gguf"))

        let found = DownloadedModels.scan(root: root, fileManager: .default)
        XCTAssertEqual(found.first?.repoID, "org/repo/MTP")
        XCTAssertEqual(found.first?.filename, "x.Q4_K_M.gguf")
        // Quant token parsed for the display label.
        XCTAssertTrue(found.first?.displayName.contains("Q4_K_M") == true)
    }

    /// Results come back sorted by displayName (case-insensitive).
    func testSortedByDisplayName() throws {
        for name in ["zeta/z", "alpha/a"] {
            let dir = root.appendingPathComponent(name, isDirectory: true)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try Data(count: 1).write(to: dir.appendingPathComponent("model.gguf"))
        }
        let names = DownloadedModels.scan(root: root, fileManager: .default).map(\.displayName)
        XCTAssertEqual(names, names.sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending })
    }
}
