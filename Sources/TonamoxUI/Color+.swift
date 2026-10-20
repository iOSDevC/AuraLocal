import SwiftUI

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

extension SwiftUI.Color {
    static var outputBackground: Color {
#if canImport(UIKit)
        return Color(.systemGroupedBackground)
#elseif canImport(AppKit)
        return Color(nsColor: .windowBackgroundColor)
#else
        return Color.secondary.opacity(0.1)
#endif
    }
    
    /// UIColor.systemGroupedBackground / NSColor.windowBackgroundColor
    static var groupedBackground: Color {
#if os(iOS)
        Color(.systemGroupedBackground)
#else
        Color(nsColor: .windowBackgroundColor)
#endif
    }
    /// UIColor.secondarySystemGroupedBackground / NSColor.controlBackgroundColor
    static var secondaryGroupedBackground: Color {
#if os(iOS)
        Color(.secondarySystemGroupedBackground)
#else
        Color(nsColor: .controlBackgroundColor)
#endif
    }
    /// UIColor.tertiarySystemGroupedBackground / NSColor.textBackgroundColor
    static var tertiaryGroupedBackground: Color {
#if os(iOS)
        Color(.tertiarySystemGroupedBackground)
#else
        Color(nsColor: .textBackgroundColor)
#endif
    }
}
