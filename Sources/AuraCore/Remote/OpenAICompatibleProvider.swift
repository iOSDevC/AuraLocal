import Foundation

/// A ``RemoteLLMProvider`` speaking the OpenAI `/chat/completions` dialect.
///
/// One conformer covers hosted OpenAI, the user's own `llama-server`
/// (`http://127.0.0.1:8080/v1`), Ollama's `/v1` compat endpoint, and GitHub
/// Models (`https://models.github.ai/inference`). `baseURL` is the segment the
/// `chat/completions` path is appended to; the API key is optional (local
/// servers ignore it). `streaming` picks SSE deltas (default) or a single
/// non-streamed JSON response — the robust path for endpoints whose SSE/usage
/// support is unverified.
public struct OpenAICompatibleProvider: RemoteLLMProvider {
    public let id: String
    public let displayName: String
    public let retentionNote: String
    private let baseURL: URL
    private let apiKey: String?
    private let streaming: Bool

    public init(
        id: String,
        displayName: String,
        baseURL: URL,
        apiKey: String? = nil,
        streaming: Bool = true,
        retentionNote: String = "Sent to an OpenAI-compatible endpoint."
    ) {
        self.id = id
        self.displayName = displayName
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.streaming = streaming
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

    /// Build a provider for **GitHub Models** — GitHub's official, OpenAI-compatible
    /// inference API, powered by the user's GitHub/Copilot account. Auth is a
    /// fine-grained PAT with the `models:read` permission. Uses the non-streaming
    /// path so token `usage` is always returned for the cost ledger, regardless of
    /// the endpoint's SSE behavior.
    public static func gitHubModels(apiKey: String) -> OpenAICompatibleProvider {
        OpenAICompatibleProvider(
            id: "cloud.github-models",
            displayName: "GitHub Models",
            baseURL: URL(string: "https://models.github.ai/inference")!,
            apiKey: apiKey,
            streaming: false,
            retentionNote: "Sent to GitHub Models (models.github.ai) with your GitHub token. See GitHub's data-retention terms.")
    }

    public func stream(_ request: RemoteRequest) -> AsyncThrowingStream<RemoteEvent, Error> {
        streaming ? streamSSE(request) : sendOnce(request)
    }

    /// SSE path: stream `data:` deltas via the shared line reader.
    private func streamSSE(_ request: RemoteRequest) -> AsyncThrowingStream<RemoteEvent, Error> {
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

    /// Non-streaming path: one POST, decode the full completion, yield the whole
    /// message plus exact usage. Robust for endpoints whose SSE/usage support is
    /// unverified (e.g. GitHub Models).
    private func sendOnce(_ request: RemoteRequest) -> AsyncThrowingStream<RemoteEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let urlRequest = try makeRequest(request)
                    let (data, response) = try await URLSession.shared.data(for: urlRequest)
                    guard let http = response as? HTTPURLResponse else {
                        throw AuraError.remoteFailed(-1, "No HTTP response")
                    }
                    guard (200..<300).contains(http.statusCode) else {
                        let body = String(data: data, encoding: .utf8) ?? ""
                        throw AuraError.remoteFailed(http.statusCode, String(body.prefix(600)))
                    }
                    let completion = try JSONDecoder().decode(Completion.self, from: data)
                    if let content = completion.choices?.first?.message.content, !content.isEmpty {
                        continuation.yield(.token(content))
                    }
                    if let usage = completion.usage {
                        continuation.yield(.usage(TokenUsage(
                            inputTokens: usage.prompt_tokens ?? 0,
                            outputTokens: usage.completion_tokens ?? 0)))
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
        var body: [String: Any] = [
            "model": req.model,
            "messages": messages,
            "max_tokens": req.maxTokens,
            "temperature": req.temperature,
            "stream": streaming,
        ]
        if streaming {
            body["stream_options"] = ["include_usage": true]
        }
        var r = URLRequest(url: baseURL.appending(path: "chat/completions"))
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        r.setValue(streaming ? "text/event-stream" : "application/json", forHTTPHeaderField: "Accept")
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

    // MARK: - Non-streaming response (OpenAI dialect)

    private struct Completion: Decodable {
        struct Choice: Decodable {
            struct Message: Decodable { let content: String? }
            let message: Message
        }
        let choices: [Choice]?
        let usage: Chunk.Usage?
    }
}
