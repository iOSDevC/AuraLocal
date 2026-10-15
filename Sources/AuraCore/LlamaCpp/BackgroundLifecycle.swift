import Foundation
#if os(iOS) || os(tvOS)
import UIKit
#endif

// MARK: - BackgroundLifecycle

/// Stops on-device generation when an iOS app moves to the background.
///
/// When the app enters the background, every running local generation (MLX, llama.cpp, layer streaming, and
/// `CoreMLDiffusionEngine` in AuraImageGen) is cancelled and its call throws ``GenerationInterrupted`` with
/// ``GenerationInterrupted/Trigger/appBackgrounded`` and the text produced so far. Until the app returns to the
/// foreground, new generations throw the same error at once. Optionally it also evicts models to shrink the
/// footprint iOS judges a backgrounded app by.
///
/// The observers are installed when a model loads or an image generation starts, so apps need no setup. Before
/// that the state is not tracked: an app already in the background when it first loads a model reads as foreground.
/// It observes UIKit on iOS and tvOS only; on macOS and visionOS it never pauses.
@MainActor
public final class BackgroundLifecycle {

    public static let shared = BackgroundLifecycle()

    /// `true` while the app is in the background. AuraLocal refuses to start a local generation while it is.
    @Published public private(set) var isPaused = false

    /// Evict every model except the most recently used one on entering the background.
    public var aggressiveMemorySaving = false {
        didSet { startObserving() }
    }

    private let notifications: NotificationCenter
    private let didEnterBackground: Notification.Name?
    private let willEnterForeground: Notification.Name?
    private var isObserving = false
    private var interruptions: [UUID: @MainActor (GenerationInterrupted.Trigger) -> Void] = [:]

    private convenience init() {
        #if os(iOS) || os(tvOS)
        self.init(notificationCenter: .default,
                  didEnterBackground: UIApplication.didEnterBackgroundNotification,
                  willEnterForeground: UIApplication.willEnterForegroundNotification)
        #else
        self.init(notificationCenter: .default, didEnterBackground: nil, willEnterForeground: nil)
        #endif
    }

    /// A lifecycle driven by arbitrary notifications, so tests can simulate backgrounding on any platform.
    package init(notificationCenter: NotificationCenter, didEnterBackground: Notification.Name?,
                 willEnterForeground: Notification.Name?) {
        self.notifications = notificationCenter
        self.didEnterBackground = didEnterBackground
        self.willEnterForeground = willEnterForeground
    }

    /// Installs the observers once. Called by the library when a model loads or an image generation starts.
    package func startObserving() {
        guard !isObserving else { return }
        isObserving = true
        if let didEnterBackground {
            notifications.addObserver(forName: didEnterBackground, object: nil, queue: nil) { [weak self] _ in
                let lifecycle = self
                Self.onMainActor { lifecycle?.enterBackground() }
            }
        }
        if let willEnterForeground {
            notifications.addObserver(forName: willEnterForeground, object: nil, queue: nil) { [weak self] _ in
                let lifecycle = self
                Self.onMainActor { lifecycle?.enterForeground() }
            }
        }
    }

    /// Registers a running generation; `interrupt` is called if the app backgrounds before ``unregister(_:)``.
    package func register(_ interrupt: @escaping @MainActor (GenerationInterrupted.Trigger) -> Void) -> UUID {
        let id = UUID()
        interruptions[id] = interrupt
        return id
    }

    package func unregister(_ id: UUID) {
        interruptions[id] = nil
    }

    /// Generations registered right now.
    package var runningGenerations: Int { interruptions.count }

    private func enterBackground() {
        isPaused = true
        // Cancel before evicting, so no generation is mid-token on a model being unloaded.
        for interrupt in interruptions.values {
            interrupt(.appBackgrounded)
        }
        if aggressiveMemorySaving {
            ModelManager.shared.evictAllButMostRecent()
        }
    }

    private func enterForeground() {
        isPaused = false
    }

    /// UIKit posts these on the main thread; run synchronously there, because the app may be suspended soon after.
    nonisolated private static func onMainActor(_ work: @escaping @MainActor @Sendable () -> Void) {
        if Thread.isMainThread {
            MainActor.assumeIsolated(work)
        } else {
            Task { @MainActor in work() }
        }
    }
}
