import XCTest
@testable import AuraCore

#if !targetEnvironment(simulator)

/// State-guard invariants for ``GGUFModelDownloader``'s pause/resume/cancel controls. These lock the guards
/// that keep the control API safe to call in any order without a live transfer (the real network flow — a
/// paused download resuming from where it stopped — is verified manually in the UI against a live HF download,
/// following the same rationale as the MITM URLSession test: ATS + the fixed cache directory make an automated
/// end-to-end download test unreliable on the test host).
@MainActor
final class GGUFDownloaderStateTests: XCTestCase {

    func testFreshStateIsIdle() {
        let downloader = GGUFModelDownloader()
        XCTAssertFalse(downloader.isDownloading)
        XCTAssertFalse(downloader.isPaused)
        XCTAssertEqual(downloader.progress, 0)
        XCTAssertNil(downloader.error)
    }

    func testPauseIsNoOpWhenNotDownloading() {
        let downloader = GGUFModelDownloader()
        downloader.pause()   // nothing in flight → must not flip into a paused-but-nothing state
        XCTAssertFalse(downloader.isPaused)
        XCTAssertFalse(downloader.isDownloading)
    }

    func testResumeIsNoOpWhenNotPaused() {
        let downloader = GGUFModelDownloader()
        downloader.resume()  // not paused → must not start a phantom download
        XCTAssertFalse(downloader.isDownloading)
        XCTAssertFalse(downloader.isPaused)
    }

    func testCancelWhenIdleIsSafe() {
        let downloader = GGUFModelDownloader()
        downloader.cancel()  // no continuation to resume → must not crash and must stay clean
        XCTAssertFalse(downloader.isDownloading)
        XCTAssertFalse(downloader.isPaused)
    }
}

#endif
