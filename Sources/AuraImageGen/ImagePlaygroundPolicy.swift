import Foundation
#if canImport(ImagePlayground)
import ImagePlayground
#endif

/// Where Apple says an Image Playground generation runs. No API reports it per style, so this follows Apple's
/// statements for the OS version.
public enum ImageGenProcessing: Sendable {
    case onDevice
    case privateCloudCompute

    /// Whether the prompt and anything added in Apple's sheet leave the device.
    public var leavesDevice: Bool {
        self == .privateCloudCompute
    }
}

#if canImport(ImagePlayground)
/// Egress-safe settings for presenting Apple's Image Playground sheet (`imagePlaygroundSheet`). AuraLocal does not
/// generate through Image Playground itself: the sheet is Apple's UI and runs Apple's pipeline.
@available(iOS 18.4, macOS 15.4, visionOS 2.4, *)
public enum ImagePlaygroundPolicy {
    /// Pass explicitly to `imagePlaygroundGenerationStyle(_:in:)`: the API's default (`.all`) includes
    /// `.externalProvider` (ChatGPT, a third-party cloud).
    public static let allowedStyles: [ImagePlaygroundStyle] = [.animation, .illustration, .sketch]
    public static let defaultStyle: ImagePlaygroundStyle = .animation

    /// On device before OS 27, Private Cloud Compute from 27, per Apple. Ask for consent when it leaves the device.
    public static var processing: ImageGenProcessing {
        processing(forMajorVersion: ProcessInfo.processInfo.operatingSystemVersion.majorVersion)
    }

    static func processing(forMajorVersion major: Int) -> ImageGenProcessing {
        major >= 27 ? .privateCloudCompute : .onDevice
    }

    /// `personalization = .disabled`, the most restrictive setting.
    @available(iOS 26.4, macOS 26.4, visionOS 26.4, *)
    public static var options: ImagePlaygroundOptions {
        var settings = ImagePlaygroundOptions()
        settings.personalization = .disabled
        return settings
    }
}
#endif
