import Foundation

/// A ``RemoteLLMProvider`` speaking the OpenAI `/chat/completions` SSE dialect.
///
/// One conformer covers three targets: hosted OpenAI, the user's own
/// `llama-server` (`http://127.0.0.1:8080/v1`), and Ollama's `/v1` compat
/// endpoint. `baseURL` must include the `/v1` segment; the API key is optional
/// (local servers ignore it).
public struct OpenAICompatibleProvider: RemoteLLMProvider {
    public let id: String
    public let displayName: String
    public let retentionNote: String
    private let baseURL: URL
    private let apiKey: String?

    public init(
        id: String,
        displayName: String,
        baseURL: URL,
        apiKey: String? = nil,
        retentionNote: String = "Sent to an OpenAI-compatible endpoint."
    ) {
        self.id = id
        self.displayName = displayName
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.retentionNote = retentionNote
    }

    /// Build a provider from a discovered local llama-server / Ollama endpoint.
    public static func from(_ status: LocalProviderStatus) -> OpenAICompatibleProvider {
        OpenAICompatibleProvider(
            id: "local.\(status.kind.rawValue)",
            displayName: status.kind == .ollama ? "Ollama (local)" : "llama-server (local)",
            baseURL: status.baseURL,
            apiKey: nil,
            retentionNote: "Runs on your own machine — nothing leaves your device.")
    }

    public func stream(_ request: RemoteRequest) -> AsyncThrowingStream<RemoteEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let urlRequest = try makeRequest(request)
                    let session = SSELineStream.makeSession()
                    defer { session.invalidateAndCancel() }
                    for try await line in SSELineStream.lines(for: urlRequest, session: session) {
                        if Task.isCancelled { break }
                        for event in Self.parse(line: line) {
                            continuation.yield(event)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Request

    private func makeRequest(_ req: RemoteRequest) throws -> URLRequest {
        var messages = req.messages
        if let system = req.system {
            messages.insert(["role": "system", "content": system], at: 0)
        }
        let body: [String: Any] = [
            "model": req.model,
            "messages": messages,
            "max_tokens": req.maxTokens,
            "temperature": req.temperature,
            "stream": true,
            "stream_options": ["include_usage": true],
        ]
        var r = URLRequest(url: baseURL.appending(path: "chat/completions"))
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        r.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        if let apiKey, !apiKey.isEmpty {
            r.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        r.httpBody = try JSONSerialization.data(withJSONObject: body)
        return r
    }

    // MARK: - SSE parsing (OpenAI dialect)

    private static func parse(line: String) -> [RemoteEvent] {
        guard line.hasPrefix("data:") else { return [] }
        let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
        guard !payload.isEmpty, payload != "[DONE]" else { return [] }
        guard let data = payload.data(using: .utf8),
              let chunk = try? JSONDecoder().decode(Chunk.self, from: data) else { return [] }
        var events: [RemoteEvent] = []
        if let content = chunk.choices?.first?.delta.content, !content.isEmpty {
            events.append(.token(content))
        }
        if let usage = chunk.usage {
            events.append(.usage(TokenUsage(
                inputTokens: usage.prompt_tokens ?? 0,
                outputTokens: usage.completion_tokens ?? 0)))
        }
        return events
    }

    private struct Chunk: Decodable {
        struct Choice: Decodable {
            struct Delta: Decodable { let content: String? }
            let delta: Delta
        }
        struct Usage: Decodable {
            let prompt_tokens: Int?
            let completion_tokens: Int?
        }
        let choices: [Choice]?
        let usage: Usage?
    }
}
