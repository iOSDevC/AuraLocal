import Foundation

// MARK: - Local provider discovery

@MainActor
public extension AuraLocal {

    /// Discover local Ollama / llama.cpp `llama-server` providers and the models
    /// they expose. Never throws — an unreachable provider comes back with
    /// `reachable == false` and no models.
    ///
    /// ```swift
    /// let providers = await AuraLocal.detectLocalProviders()
    /// for p in providers where p.isAvailable {
    ///     print(p.kind, p.models.map(\.name))
    /// }
    /// ```
    static func detectLocalProviders(
        endpoints: [LocalProviderEndpoint] = LocalProviderEndpoint.defaults,
        timeout: TimeInterval = 2.0
    ) async -> [LocalProviderStatus] {
        await LocalProviderDetector.detectAll(endpoints: endpoints, timeout: timeout)
    }

    /// Only the providers that are reachable and expose at least one model.
    static func availableLocalProviders(
        endpoints: [LocalProviderEndpoint] = LocalProviderEndpoint.defaults,
        timeout: TimeInterval = 2.0
    ) async -> [LocalProviderStatus] {
        await detectLocalProviders(endpoints: endpoints, timeout: timeout).filter(\.isAvailable)
    }
}
