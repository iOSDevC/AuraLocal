import Foundation

// MARK: - SamplingSettings

/// Temperature and top-p, fixed for an instance's lifetime: llama.cpp builds its sampler chain with the context and
/// the pinned LocalLLMClient keeps it private. MLX could change them per call; both keep one setting per instance.
struct SamplingSettings: Hashable, Sendable {
    let temperature: Float
    let topP: Float

    /// `nil` takes the backend default; an out-of-range value throws ``ModelConfigurationError``.
    static func resolve(temperature: Float?, topP: Float?, for model: Model) throws -> SamplingSettings {
        if let temperature, !(temperature.isFinite && temperature >= 0) {
            throw ModelConfigurationError.invalidTemperature(temperature)
        }
        if let topP, !(topP.isFinite && topP > 0 && topP <= 1) {
            throw ModelConfigurationError.invalidTopP(topP)
        }
        let fallback = defaults(for: model)
        return SamplingSettings(temperature: temperature ?? fallback.temperature, topP: topP ?? fallback.topP)
    }

    /// Backend defaults: GGUF 0.7 / top-p 0.95; MLX 0.7 text, 0.1 vision, 0 OCR, with top-p off (1).
    static func defaults(for model: Model) -> SamplingSettings {
        switch model.format {
        case .gguf:
            return SamplingSettings(temperature: 0.7, topP: 0.95)
        case .mlx:
            switch model.purpose {
            case .text:              return SamplingSettings(temperature: 0.7, topP: 1)
            case .vision:            return SamplingSettings(temperature: 0.1, topP: 1)
            case .visionSpecialized: return SamplingSettings(temperature: 0, topP: 1)
            }
        }
    }
}
