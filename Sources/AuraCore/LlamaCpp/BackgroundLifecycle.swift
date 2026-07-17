import Foundation
#if os(iOS) || os(tvOS)
import UIKit
#endif

// MARK: - BackgroundLifecycle

/// Manages inference lifecycle when the app transitions to/from background on iOS.
///
/// On iOS, apps in background are subject to stricter memory limits and may be
/// suspended or terminated. This handler:
/// - Pauses active generation when entering background
/// - Optionally evicts models to reduce memory footprint
/// - Restores state when returning to foreground
///
/// On macOS this is a no-op since apps are not suspended.
@MainActor
public final class BackgroundLifecycle {

    public static let shared = BackgroundLifecycle()

    /// Whether inference should be paused (app is in background).
    @Published public private(set) var isPaused = false

    /// Whether aggressive memory saving is enabled (evict models on background).
    public var aggressiveMemorySaving = false

    private init() {
        #if os(iOS) || os(tvOS)
        observeAppLifecycle()
        #endif
    }

    #if os(iOS) || os(tvOS)
    private func observeAppLifecycle() {
        NotificationCenter.default.addObserver(
            forName: UIApplication.didEnterBackgroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleDidEnterBackground()
            }
        }

        NotificationCenter.default.addObserver(
            forName: UIApplication.willEnterForegroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleWillEnterForeground()
            }
        }
    }
    #endif

    private func handleDidEnterBackground() {
        isPaused = true

        if aggressiveMemorySaving {
            // Free RAM while backgrounded, keeping only the active model warm.
            ModelManager.shared.evictAllButMostRecent()
        }
    }

    private func handleWillEnterForeground() {
        isPaused = false
    }
}
