import Foundation

/// A ``RemoteLLMProvider`` for Anthropic's Messages API (`/v1/messages`) with
/// streaming. Parses the named-event SSE dialect and reports exact token usage.
public struct AnthropicProvider: RemoteLLMProvider {
    public let id = "cloud.anthropic"
    public let displayName = "Anthropic (Claude)"
    public let retentionNote = "Sent to Anthropic's API. See their data-retention policy."

    private let apiKey: String
    private let baseURL: URL
    private let version: String

    public init(
        apiKey: String,
        baseURL: URL = URL(string: "https://api.anthropic.com")!,
        version: String = "2023-06-01"
    ) {
        self.apiKey = apiKey
        self.baseURL = baseURL
        self.version = version
    }

    public func stream(_ request: RemoteRequest) -> AsyncThrowingStream<RemoteEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let urlRequest = try makeRequest(request)
                    let session = SSELineStream.makeSession()
                    defer { session.invalidateAndCancel() }
                    var inputTokens: Int?
                    for try await line in SSELineStream.lines(for: urlRequest, session: session) {
                        if Task.isCancelled { break }
                        guard line.hasPrefix("data:") else { continue }
                        let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                        guard !payload.isEmpty, let data = payload.data(using: .utf8),
                              let event = try? JSONDecoder().decode(Event.self, from: data) else { continue }
                        switch event.type {
                        case "content_block_delta":
                            if let text = event.delta?.text, !text.isEmpty {
                                continuation.yield(.token(text))
                            }
                        case "message_start":
                            inputTokens = event.message?.usage?.input_tokens
                        case "message_delta":
                            if let usage = Self.tokenUsage(event.usage, inputTokens: inputTokens) {
                                continuation.yield(.usage(usage))
                            }
                        default:
                            break
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
        // Anthropic keeps `system` at the top level; messages carry only user/assistant.
        let messages = req.messages.filter { $0["role"] != "system" }
        var body: [String: Any] = [
            "model": req.model,
            "max_tokens": req.maxTokens,
            "messages": messages,
            "stream": true,
            "temperature": req.temperature,
        ]
        if let system = req.system { body["system"] = system }

        var r = URLRequest(url: baseURL.appending(path: "v1/messages"))
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        r.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        r.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        r.setValue(version, forHTTPHeaderField: "anthropic-version")
        r.httpBody = try JSONSerialization.data(withJSONObject: body)
        return r
    }

    /// Both counts or nothing: a missing count is unknown, and pricing it as 0 would undercount the call.
    /// `message_delta` may repeat `input_tokens`; otherwise it comes from `message_start`.
    private static func tokenUsage(_ usage: Event.Usage?, inputTokens: Int?) -> TokenUsage? {
        guard let input = usage?.input_tokens ?? inputTokens, let output = usage?.output_tokens else { return nil }
        return TokenUsage(inputTokens: input, outputTokens: output)
    }

    // MARK: - SSE event (Anthropic dialect)

    private struct Event: Decodable {
        struct Delta: Decodable { let text: String? }
        struct Usage: Decodable { let input_tokens: Int?; let output_tokens: Int? }
        struct Message: Decodable { let usage: Usage? }
        let type: String
        let delta: Delta?
        let usage: Usage?
        let message: Message?
    }
}
