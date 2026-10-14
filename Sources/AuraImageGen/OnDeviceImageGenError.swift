import Foundation

/// Why on-device image generation (``CoreMLDiffusionEngine``) or the Image Playground path failed. A separate enum
/// from ``ImageGenError`` so existing exhaustive switches over it keep compiling. Cancellation throws
/// `CancellationError`, not a case here.
public enum OnDeviceImageGenError: LocalizedError, Equatable, Sendable {
    /// No Core ML Stable Diffusion on this platform (visionOS).
    case unsupportedPlatform
    /// The Simulator has no Neural Engine, and on the CPU this model only produces black images.
    case simulatorUnsupported
    /// The model folder is missing or incomplete (iOS may have purged Caches).
    case modelNotDownloaded(sizeBytes: Int64)
    /// Free memory is below the estimated peak.
    case insufficientMemory(requiredBytes: Int64, availableBytes: Int64)
    /// A request field is out of range; nothing is clamped.
    case invalidRequest(field: String, reason: String)
    /// Another generation is running on this engine.
    case busy
    /// Core ML could not load the model.
    case modelLoadFailed(String)
    /// The pipeline failed during generation.
    case generationFailed(String)
    /// Every image was replaced by the safety checker.
    case filteredBySafetyChecker
    /// `supportsImagePlayground` is false. Apple exposes only that Bool, so the reason is unknown.
    case imagePlaygroundUnavailable
    /// The user declined sending the prompt to Apple's servers.
    case consentDeclined

    public var errorDescription: String? {
        switch self {
        case .unsupportedPlatform:
            "On-device image generation isn't available on this platform."
        case .simulatorUnsupported:
            "Core ML Stable Diffusion can't produce images in the Simulator."
        case .modelNotDownloaded(let sizeBytes):
            "Download the image model (about \(Self.fileSize(sizeBytes))) first."
        case .insufficientMemory(let required, let available):
            "Not enough free memory: needs about \(Self.memorySize(required)), \(Self.memorySize(available)) available."
        case .invalidRequest(let field, let reason):
            "\(field): \(reason)"
        case .busy:
            "An image is already being generated."
        case .modelLoadFailed(let detail):
            "The image model could not be loaded: \(detail)"
        case .generationFailed(let detail):
            "Image generation failed: \(detail)"
        case .filteredBySafetyChecker:
            "The safety checker blocked every image."
        case .imagePlaygroundUnavailable:
            "Apple Image Playground isn't available. It needs a supported device with Apple Intelligence turned on, "
                + "image generation enabled, and a supported language and region. It is not available on shared "
                + "devices or in the Simulator on this Mac."
        case .consentDeclined:
            "Nothing was sent. Image Playground needs your OK because it runs on Apple's servers."
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .unsupportedPlatform:
            "Use an iPhone, iPad or Mac."
        case .simulatorUnsupported:
            "Run the app on an iPhone, iPad or Mac."
        case .modelNotDownloaded:
            "Download it once; it stays on this device until you delete it or iOS clears its caches."
        case .insufficientMemory:
            "Unload the chat model or close other apps."
        case .invalidRequest(let field, _):
            "Change \(field) and try again."
        case .busy:
            "Wait for it to finish or cancel it."
        case .modelLoadFailed:
            "Delete and download it again."
        case .generationFailed:
            "Try again, switch compute units, or use the low-memory mode."
        case .filteredBySafetyChecker:
            "Try a different prompt."
        case .imagePlaygroundUnavailable:
            "Use the on-device engine instead."
        case .consentDeclined:
            "Use the on-device engine to keep everything on this device."
        }
    }

    private static func fileSize(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    private static func memorySize(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .memory)
    }
}
