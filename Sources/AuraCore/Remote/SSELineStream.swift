import Foundation

/// Shared Server-Sent-Events reader over `URLSession.bytes(for:)`, yielding raw
/// lines for a provider-specific dialect parser. Cancellation-aware: terminating
/// the returned stream cancels the underlying transfer.
enum SSELineStream {

    /// POST `request` and stream response lines. Throws ``AuraError/remoteFailed``
    /// on a non-2xx status (with a snippet of the error body).
    static func lines(for request: URLRequest, session: URLSession) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let (bytes, response) = try await session.bytes(for: request)
                    guard let http = response as? HTTPURLResponse else {
                        throw AuraError.remoteFailed(-1, "No HTTP response")
                    }
                    guard (200..<300).contains(http.statusCode) else {
                        var body = ""
                        for try await line in bytes.lines {
                            body += line + "\n"
                            if body.count > 600 { break }
                        }
                        throw AuraError.remoteFailed(http.statusCode, String(body.prefix(600)))
                    }
                    for try await line in bytes.lines {
                        if Task.isCancelled { break }
                        continuation.yield(line)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// A URLSession tuned for long-lived streaming: fails fast when offline so
    /// escalation degrades to the local answer instead of hanging.
    static func makeSession(requestTimeout: TimeInterval = 120) -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = requestTimeout
        config.waitsForConnectivity = false
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        return URLSession(configuration: config)
    }
}
