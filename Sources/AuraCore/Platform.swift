import SwiftUI

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - Platform

/// A cross-platform image type — `UIImage` on iOS/tvOS/visionOS, `NSImage` on macOS.
///
/// All AuraLocal vision APIs accept and return `PlatformImage` so that
/// callers don't need platform-conditional code.
#if canImport(UIKit)
public typealias PlatformImage = UIImage
#elseif canImport(AppKit)
public typealias PlatformImage = NSImage
#endif

// MARK: - SwiftUI extensions

public extension SwiftUI.Image {
    /// Create a SwiftUI `Image` from a ``PlatformImage``.
    init(platformImage: PlatformImage) {
#if canImport(UIKit)
        self.init(uiImage: platformImage)
#elseif canImport(AppKit)
        self.init(nsImage: platformImage)
#endif
    }
}
