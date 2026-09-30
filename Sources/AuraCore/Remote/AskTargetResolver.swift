import Foundation

/// Chooses the ``RemoteTarget`` for a one-shot ask (the `aura ask` CLI) from
/// explicit inputs only: no network and no Keychain access of its own.
///
/// `.auto` and `.local` never pick a cloud provider. A prompt leaves the machine
/// only when the caller names `.openAI` / `.anthropic` or passes a base URL.
public enum AskTargetResolver {

    /// The provider a caller asks for.
    public enum Choice: String, Sendable, CaseIterable {
        /// A running llama-server, else Ollama. Same as `.local`; the default.
        case auto
        /// A running llama-server, else Ollama.
        case local
        /// Hosted OpenAI; key from `OPENAI_API_KEY`, else the Keychain `cloud.openai`.
        case openAI = "openai"
        /// Anthropic; key from `ANTHROPIC_API_KEY`, else the Keychain `cloud.anthropic`.
        case anthropic
    }

    /// Why no target could be chosen. `errorDescription` says what to do next.
    public enum Failure: Error, Equatable, LocalizedError {
        /// No llama-server or Ollama with an on-machine model is running.
        case noLocalProvider
        /// No running local provider serves `model` on its machine; `available` lists what they do serve.
        case modelNotServedLocally(model: String, available: [String])
        /// A named cloud provider has no key in the environment or the Keychain.
        case missingAPIKey(provider: Choice, environmentVariable: String, keychainAccount: String)
        /// A base URL was combined with a cloud provider, or `.local` with a public host.
        case conflictingOptions(String)
        /// A base URL was given without a model id.
        case modelRequired
        /// The base URL is not an absolute `http`/`https` URL with a host.
        case invalidBaseURL(String)

        /// `true` when the options themselves are wrong (the CLI exits 2, not 1).
        public var isUsageError: Bool {
            switch self {
            case .conflictingOptions, .modelRequired, .invalidBaseURL: true
            case .noLocalProvider, .modelNotServedLocally, .missingAPIKey: false
            }
        }

        public var errorDescription: String? {
            switch self {
            case .noLocalProvider:
                """
                No local model server with an on-machine model is running (Ollama cloud models don't count). Either:
                  - start llama-server (http://127.0.0.1:8080/v1) or Ollama (http://localhost:11434) with a model loaded,
                  - pass --provider openai or --provider anthropic to send the prompt to that cloud API, or
                  - pass --base-url <url> --model <id> for another OpenAI-compatible server.
                """
            case .modelNotServedLocally(let model, let available):
                "No running local provider serves \"\(model)\" on this machine. Local models: "
                    + (available.isEmpty ? "none" : available.joined(separator: ", ")) + "."
            case .missingAPIKey(let provider, let variable, let account):
                "No API key for \(provider.rawValue). Set \(variable), or store the key in the Keychain "
                    + "(service dev.auralocal.remote, account \(account))."
            case .conflictingOptions(let reason):
                reason
            case .modelRequired:
                "--base-url needs --model <id> (llama-server accepts any id; other servers need a real one)."
            case .invalidBaseURL(let value):
                "Invalid --base-url \"\(value)\": expected an http(s) URL such as http://192.168.1.20:8080/v1."
            }
        }
    }

    /// Resolve the target for `choice`.
    ///
    /// - Parameters:
    ///   - choice: The requested provider.
    ///   - model: A model id; `nil` picks the provider's default (local: the model
    ///     with the largest context window).
    ///   - baseURL: An OpenAI-compatible base URL (`chat/completions` is appended).
    ///     Allowed with `.auto` / `.local` only; its key comes from `AURA_API_KEY`.
    ///   - environment: Process environment (API-key variables).
    ///   - readKey: Keychain reader, called only for a named cloud provider.
    ///   - localProviders: Probed local providers (``LocalProviderDetector/detectAll(endpoints:timeout:)``).
    ///     Ignored when `baseURL` is set.
    public static func resolve(
        _ choice: Choice,
        model: String? = nil,
        baseURL: String? = nil,
        environment: [String: String],
        readKey: (String) -> String?,
        localProviders: [LocalProviderStatus]
    ) throws(Failure) -> RemoteTarget {
        let model = nonEmpty(model)
        if let baseURL {
            return try customTarget(choice, baseURL: baseURL, model: model, environment: environment)
        }
        switch choice {
        case .auto, .local:
            return try localTarget(model: model, localProviders: localProviders)
        case .openAI:
            let key = try apiKey(for: choice, variable: "OPENAI_API_KEY", account: CloudAccount.openAI,
                                 environment: environment, readKey: readKey)
            return .openAI(apiKey: key, model: model ?? RemoteTarget.defaultOpenAIModel)
        case .anthropic:
            let key = try apiKey(for: choice, variable: "ANTHROPIC_API_KEY", account: CloudAccount.anthropic,
                                 environment: environment, readKey: readKey)
            return .anthropic(apiKey: key, model: model ?? RemoteTarget.defaultAnthropicModel)
        }
    }

    // MARK: - Local

    private static func localTarget(
        model: String?, localProviders: [LocalProviderStatus]
    ) throws(Failure) -> RemoteTarget {
        guard let model else {
            guard let target = HybridEscalator.bestLocalTarget(from: localProviders) else {
                throw .noLocalProvider
            }
            return target
        }
        let available = localProviders.filter { $0.isAvailable && $0.models.contains(where: \.runsLocally) }
        guard !available.isEmpty else { throw .noLocalProvider }
        let ordered = available.filter { $0.kind == .llamaServer } + available.filter { $0.kind != .llamaServer }
        let matches = ordered.compactMap { status in
            status.models.first { $0.runsLocally && serves($0.name, requested: model) }
                .map { (status: status, served: $0) }
        }
        guard let match = matches.first else {
            let local = ordered.flatMap { $0.models.filter(\.runsLocally).map(\.name) }
            throw .modelNotServedLocally(model: model, available: local)
        }
        return RemoteTarget(
            provider: OpenAICompatibleProvider.from(match.status),
            modelID: match.served.name,
            contextLength: match.served.contextLength,
            origin: .localNetwork(match.status.kind))
    }

    /// Ollama resolves a bare name to its `:latest` tag.
    private static func serves(_ name: String, requested: String) -> Bool {
        name == requested || name == requested + ":latest"
    }

    // MARK: - Cloud

    private static func apiKey(
        for choice: Choice, variable: String, account: String,
        environment: [String: String], readKey: (String) -> String?
    ) throws(Failure) -> String {
        guard let key = nonEmpty(environment[variable]) ?? nonEmpty(readKey(account)) else {
            throw .missingAPIKey(provider: choice, environmentVariable: variable, keychainAccount: account)
        }
        return key
    }

    // MARK: - Custom base URL

    private static func customTarget(
        _ choice: Choice, baseURL: String, model: String?, environment: [String: String]
    ) throws(Failure) -> RemoteTarget {
        if choice == .openAI || choice == .anthropic {
            throw .conflictingOptions(
                "--base-url and --provider \(choice.rawValue) are mutually exclusive: "
                    + "--base-url already names the server.")
        }
        guard let url = URL(string: baseURL),
              let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https",
              let host = url.host(), !host.isEmpty else {
            throw .invalidBaseURL(baseURL)
        }
        guard let model else { throw .modelRequired }

        let isLocal = isLocalNetworkHost(host)
        if choice == .local && !isLocal {
            throw .conflictingOptions(
                "--provider local needs a loopback or private-network --base-url; \(host) is public. "
                    + "Drop --provider local to send to it.")
        }
        let bracketed = host.contains(":") && !host.hasPrefix("[") ? "[\(host)]" : host
        let authority = url.port.map { "\(bracketed):\($0)" } ?? bracketed
        let provider = OpenAICompatibleProvider(
            id: "custom.\(authority)",
            displayName: "\(authority) (OpenAI-compatible)",
            baseURL: url,
            apiKey: nonEmpty(environment["AURA_API_KEY"]),
            retentionNote: isLocal
                ? "Sent to your own server at \(authority) on the local network."
                : "Sent to \(authority). See that service's data-retention policy.")
        // A custom server speaks the llama-server (OpenAI /v1) dialect; the kind only labels the origin.
        return RemoteTarget(
            provider: provider, modelID: model, contextLength: nil,
            origin: isLocal ? .localNetwork(.llamaServer) : .cloud)
    }

    /// `true` for loopback, private (RFC 1918 / ULA), link-local and mDNS hosts.
    /// Anything else, including CGNAT and single-label names, counts as public.
    static func isLocalNetworkHost(_ rawHost: String) -> Bool {
        var host = rawHost.lowercased()
        if host.hasPrefix("[") && host.hasSuffix("]") { host = String(host.dropFirst().dropLast()) }
        if let zone = host.firstIndex(of: "%") { host = String(host[..<zone]) }
        if host.hasSuffix(".") { host = String(host.dropLast()) }   // FQDN form: "localhost."
        if host == "localhost" || host.hasSuffix(".localhost")
            || host.hasSuffix(".local") || host.hasSuffix(".home.arpa") {
            return true
        }
        var v4 = in_addr()
        if inet_pton(AF_INET, host, &v4) == 1 {
            return isPrivateIPv4(withUnsafeBytes(of: &v4) { Array($0) })
        }
        var v6 = in6_addr()
        if inet_pton(AF_INET6, host, &v6) == 1 {
            let bytes = withUnsafeBytes(of: &v6) { Array($0) }
            if bytes[0..<15].allSatisfy({ $0 == 0 }) && bytes[15] == 1 { return true }   // ::1
            if bytes[0] & 0xFE == 0xFC { return true }                                   // fc00::/7
            if bytes[0] == 0xFE && bytes[1] & 0xC0 == 0x80 { return true }               // fe80::/10
            if bytes[0..<10].allSatisfy({ $0 == 0 }) && bytes[10] == 0xFF && bytes[11] == 0xFF {
                return isPrivateIPv4(Array(bytes[12..<16]))                              // ::ffff:a.b.c.d
            }
        }
        return false
    }

    /// Network-order IPv4 bytes.
    private static func isPrivateIPv4(_ b: [UInt8]) -> Bool {
        b[0] == 127 || b[0] == 10 || b[0] == 0
            || (b[0] == 172 && (16...31).contains(b[1]))
            || (b[0] == 192 && b[1] == 168)
            || (b[0] == 169 && b[1] == 254)
    }

    private static func nonEmpty(_ value: String?) -> String? {
        guard let value = value?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty else {
            return nil
        }
        return value
    }
}
