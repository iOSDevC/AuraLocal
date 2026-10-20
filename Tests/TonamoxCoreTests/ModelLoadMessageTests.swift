import XCTest
@testable import TonamoxCore

final class ModelLoadMessageTests: XCTestCase {
    func testOnlyDownloadProgressCountsAsDownloading() {
        XCTAssertTrue(ModelManager.isDownloadMessage("Downloading Qwen3 1.7B: 42%"))
        XCTAssertFalse(ModelManager.isDownloadMessage("Qwen3 1.7B ready"))
        XCTAssertFalse(ModelManager.isDownloadMessage("Loading Llama 3.1 8B (GGUF)..."))
        XCTAssertFalse(ModelManager.isDownloadMessage("Preparing Mistral 7B (streaming mode)..."))
    }
}
