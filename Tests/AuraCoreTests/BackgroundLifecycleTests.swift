import XCTest
@testable import AuraCore

/// Emits one token every few milliseconds until cancelled, then returns what it produced, as llama.cpp does.
@MainActor
final class SlowFakeBackend: InferenceBackend {
    private(set) var isLoaded = false
    private(set) var generations = 0
    private(set) var producedTokens = 0
    private(set) var isGenerating = false
    private(set) var lastGenerationStats: GenerationStats?
    private(set) var lastContextOverflow: ContextWindowOverflow?
    /// Reported as ``lastContextOverflow`` by every generation that runs.
    var overflowToReport: ContextWindowOverflow?

    func load(onProgress: @escaping @MainActor (String) -> Void) async throws {
        isLoaded = true
    }

    func generate(prompt: String, systemPrompt: String?, maxTokens: Int,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String {
        generations += 1
        isGenerating = true
        defer { isGenerating = false }
        var text = ""
        while producedTokens < maxTokens, !Task.isCancelled {
            text += "t\(producedTokens) "
            producedTokens += 1
            onToken(text)
            try? await Task.sleep(for: .milliseconds(2))
        }
        lastGenerationStats = GenerationStats(backend: .llamaCpp, source: .measured, promptTokens: 1,
                                              generatedTokens: producedTokens, promptTime: 0, timeToFirstToken: 0,
                                              decodeTime: 0)
        lastContextOverflow = overflowToReport
        return text
    }

    func generate(prompt: String, systemPrompt: String?, maxTokens: Int, constraint: GenerationConstraint,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String {
        try await generate(prompt: prompt, systemPrompt: systemPrompt, maxTokens: maxTokens, onToken: onToken)
    }

    func generate(prompt: String, image: PlatformImage?, maxTokens: Int,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String {
        try await generate(prompt: prompt, systemPrompt: nil, maxTokens: maxTokens, onToken: onToken)
    }

    func generate(messages: [[String: String]], maxTokens: Int,
                  onToken: @escaping @MainActor (String) -> Void) async throws -> String {
        try await generate(prompt: "", systemPrompt: nil, maxTokens: maxTokens, onToken: onToken)
    }

    func unload() {
        isLoaded = false
    }
}

struct WaitTimedOut: Error {}

@MainActor
func waitUntil(timeout: Duration = .seconds(5), _ condition: () -> Bool) async throws {
    let deadline = ContinuousClock.now + timeout
    while !condition(), ContinuousClock.now < deadline {
        try await Task.sleep(for: .milliseconds(2))
    }
    if !condition() { throw WaitTimedOut() }
}

/// Backgrounding is simulated with a private `NotificationCenter`, so it runs on macOS where UIKit never posts.
@MainActor
final class BackgroundLifecycleTests: XCTestCase {

    private let background = Notification.Name("test.didEnterBackground")
    private let foreground = Notification.Name("test.willEnterForeground")
    private let notifications = NotificationCenter()
    private let model = Model.dolphin3_qwen25_1_5b_gguf

    private func makeLifecycle() -> BackgroundLifecycle {
        BackgroundLifecycle(notificationCenter: notifications, didEnterBackground: background,
                            willEnterForeground: foreground)
    }

    func testBackgroundNotificationEndsAnInFlightGeneration() async throws {
        let lifecycle = makeLifecycle()
        let backend = SlowFakeBackend()
        let engine = AuraEngine(model: model, backend: backend, lifecycle: lifecycle)
        try await engine.load { _ in }
        let pending = Task { @MainActor in
            try await engine.generate(prompt: "x", maxTokens: 100_000) { _ in }
        }
        try await waitUntil { backend.producedTokens >= 3 }

        notifications.post(name: background, object: nil)

        let result = await pending.result
        XCTAssertThrowsError(try result.get()) { error in
            guard let stop = error as? GenerationInterrupted else { return XCTFail("\(error)") }
            XCTAssertEqual(stop.reason, .appBackgrounded)
            XCTAssertTrue(stop.partialText.hasPrefix("t0 t1 t2 "), stop.partialText)
        }
        XCTAssertFalse(backend.isGenerating, "the call must not return while the decode still runs")
        XCTAssertLessThan(backend.producedTokens, 100_000)
        XCTAssertTrue(lifecycle.isPaused)
        XCTAssertEqual(lifecycle.runningGenerations, 0)
    }

    func testInterruptedGenerationLeavesNoStats() async throws {
        let lifecycle = makeLifecycle()
        let backend = SlowFakeBackend()
        let engine = AuraEngine(model: model, backend: backend, lifecycle: lifecycle)
        try await engine.load { _ in }
        _ = try await engine.generate(prompt: "x", maxTokens: 2) { _ in }
        XCTAssertNotNil(engine.lastGenerationStats)

        let pending = Task { @MainActor in
            try await engine.generate(prompt: "x", maxTokens: 100_000) { _ in }
        }
        try await waitUntil { backend.producedTokens >= 5 }
        notifications.post(name: background, object: nil)

        let result = await pending.result
        XCTAssertThrowsError(try result.get()) { XCTAssertTrue($0 is GenerationInterrupted, "\($0)") }
        XCTAssertNotNil(backend.lastGenerationStats, "the backend returned normally after the cancel")
        XCTAssertNil(engine.lastGenerationStats, "an interrupted call must not report the cut-off decode as finished")
    }

    func testRefusedGenerationLeavesNoContextOverflow() async throws {
        let lifecycle = makeLifecycle()
        let backend = SlowFakeBackend()
        backend.overflowToReport = ContextWindowOverflow.detect(promptTokens: 1900, generatedTokens: 300,
                                                                windowSize: 2048)
        let engine = AuraEngine(model: model, backend: backend, lifecycle: lifecycle)
        try await engine.load { _ in }
        _ = try await engine.generate(prompt: "x", maxTokens: 2) { _ in }
        XCTAssertNotNil(engine.lastContextOverflow)

        notifications.post(name: background, object: nil)
        _ = try? await engine.generate(prompt: "x", maxTokens: 4) { _ in }

        XCTAssertNotNil(backend.lastContextOverflow, "the backend still holds the previous call's value")
        XCTAssertNil(engine.lastContextOverflow)
    }

    func testGenerationIsRefusedInTheBackgroundAndAllowedAfterReturning() async throws {
        let lifecycle = makeLifecycle()
        let backend = SlowFakeBackend()
        let engine = AuraEngine(model: model, backend: backend, lifecycle: lifecycle)
        try await engine.load { _ in }
        notifications.post(name: background, object: nil)

        do {
            _ = try await engine.generate(prompt: "x", maxTokens: 4) { _ in }
            XCTFail("expected GenerationInterrupted")
        } catch let stop as GenerationInterrupted {
            XCTAssertEqual(stop, GenerationInterrupted(reason: .appBackgrounded, partialText: ""))
        }
        XCTAssertEqual(backend.generations, 0)

        notifications.post(name: foreground, object: nil)
        let reply = try await engine.generate(prompt: "x", maxTokens: 2) { _ in }
        XCTAssertEqual(reply, "t0 t1 ")
        XCTAssertFalse(lifecycle.isPaused)
    }

    func testObserversAreInstalledWhenAModelLoads() async throws {
        let lifecycle = makeLifecycle()
        let engine = AuraEngine(model: model, backend: SlowFakeBackend(), lifecycle: lifecycle)

        notifications.post(name: background, object: nil)
        XCTAssertFalse(lifecycle.isPaused, "nothing observes before a model loads")

        try await engine.load { _ in }
        notifications.post(name: background, object: nil)
        XCTAssertTrue(lifecycle.isPaused)
    }

    func testCallerCancellationIsNotReportedAsAnInterruption() async throws {
        let lifecycle = makeLifecycle()
        let backend = SlowFakeBackend()
        let engine = AuraEngine(model: model, backend: backend, lifecycle: lifecycle)
        let pending = Task { @MainActor in
            try await engine.generate(prompt: "x", maxTokens: 100_000) { _ in }
        }
        try await waitUntil { backend.producedTokens >= 2 }

        pending.cancel()

        let reply = try await pending.value
        XCTAssertTrue(reply.hasPrefix("t0 t1 "), reply)
        XCTAssertFalse(backend.isGenerating)
        XCTAssertEqual(lifecycle.runningGenerations, 0)
    }

    func testBackgroundingAfterAGenerationFinishedLeavesItsResult() async throws {
        let lifecycle = makeLifecycle()
        let engine = AuraEngine(model: model, backend: SlowFakeBackend(), lifecycle: lifecycle)
        try await engine.load { _ in }

        let reply = try await engine.generate(prompt: "x", maxTokens: 3) { _ in }
        notifications.post(name: background, object: nil)

        XCTAssertEqual(reply, "t0 t1 t2 ")
        XCTAssertEqual(lifecycle.runningGenerations, 0)
    }
}
