import Foundation

// MARK: - SupervisedRun

/// State of one generation shared by its token callback, the background lifecycle and the device guard.
@MainActor
final class SupervisedRun {
    var latestText = ""
    var guardState: DeviceConditionsCheck?
    var generation: Task<String, Error>?
    private(set) var reason: GenerationInterrupted.Trigger?
    /// MLX delivers tokens through main-actor hops that can land after its call returned; those must not
    /// relabel a finished generation as interrupted.
    var isFinished = false

    func interrupt(_ trigger: GenerationInterrupted.Trigger) {
        guard reason == nil, !isFinished else { return }
        reason = trigger
        generation?.cancel()
    }
}

// MARK: - Supervision

extension TonamoxEngine {

    /// Runs `body` in a task that backgrounding and the device guard can cancel, and reports their stop as
    /// ``GenerationInterrupted``. It always awaits `body`, so a returned call leaves no decode running.
    func supervised(
        onToken: @escaping @MainActor (String) -> Void,
        _ body: @escaping @MainActor (_ onToken: @escaping @MainActor (String) -> Void) async throws -> String
    ) async throws -> String {
        appLifecycle.startObserving()
        if appLifecycle.isPaused {
            throw GenerationInterrupted(reason: .appBackgrounded, partialText: "")
        }
        let run = SupervisedRun()
        let policy = deviceGuard
        if let policy {
            var state = DeviceConditionsCheck(policy: policy)
            if let blocked = state.evaluate(policy.currentReading()) {
                throw GenerationInterrupted(reason: .deviceConditions(blocked), partialText: "")
            }
            run.guardState = state
        }

        let generation = Task { @MainActor in
            defer { run.isFinished = true }
            return try await body { partial in
                guard !run.isFinished else { return }
                run.latestText = partial
                onToken(partial)
                if let policy, let blocked = run.guardState?.tick({ policy.currentReading() }) {
                    run.interrupt(.deviceConditions(blocked))
                }
            }
        }
        run.generation = generation
        let registration = appLifecycle.register { run.interrupt($0) }
        defer { appLifecycle.unregister(registration) }

        let result = await withTaskCancellationHandler {
            await generation.result
        } onCancel: {
            generation.cancel()
        }
        if let reason = run.reason {
            let returned = (try? result.get()) ?? ""
            let partial = returned.count >= run.latestText.count ? returned : run.latestText
            throw GenerationInterrupted(reason: reason, partialText: partial)
        }
        return try result.get()
    }
}
