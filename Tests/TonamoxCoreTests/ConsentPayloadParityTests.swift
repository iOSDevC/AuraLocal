import Foundation
import Synchronization
import XCTest
@testable import TonamoxCore

/// What `routeAndEscalate` shows the consent gate is what goes over the wire: the HTTP body is
/// captured by a stub `URLProtocol` behind the real providers and compared with the approved payload.
@MainActor
final class ConsentPayloadParityTests: XCTestCase {

    private static let deployKey = "sk-" + String(repeating: "Zq7", count: 8)
    private static let email = "alice@example.com"
    private static let systemPrompt = "You are a release engineer."
    /// About 1,600 tokens: over the 550-token budget of a 2,048-token window at keepRatio 0.3.
    private static let context = String(repeating: "Bananas are a yellow fruit grown in warm places. ", count: 130)
        + "The billing service deploy key is \(deployKey) and its alerts go to \(email)."

    private enum Dialect { case openAI, anthropic }

    private func makeQuestion() -> String {
        "Which deploy key does the billing service use, and should \(Self.email) rotate it? Run \(UUID().uuidString)"
    }

    private func stubbedHost() -> String { "parity-\(UUID().uuidString.lowercased()).invalid" }

    private func openAITarget(host: String, origin: RemoteTarget.Origin) -> RemoteTarget {
        var provider = OpenAICompatibleProvider(
            id: "test.parity-openai", displayName: "Parity OpenAI",
            baseURL: URL(string: "https://\(host)/v1")!, apiKey: "test-key")
        provider.makeSession = { PayloadCaptureProtocol.session() }
        return RemoteTarget(provider: provider, modelID: "test-model", contextLength: 2048, origin: origin)
    }

    private func anthropicTarget(host: String) -> RemoteTarget {
        var provider = AnthropicProvider(apiKey: "test-key", baseURL: URL(string: "https://\(host)")!)
        provider.makeSession = { PayloadCaptureProtocol.session() }
        return RemoteTarget(provider: provider, modelID: "test-model", contextLength: 2048, origin: .cloud)
    }

    private func route(
        _ target: RemoteTarget, policy: EscalationPolicy, gate: PayloadRecordingGate, question: String
    ) async throws -> HybridEscalator.Result? {
        try await HybridEscalator(ledger: CostLedger()).routeAndEscalate(
            policy: policy, systemPrompt: Self.systemPrompt,
            context: Self.context, question: question,
            localAnswer: "I'm not sure.", consent: gate, maxTokens: 64, targets: [target])
    }

    private func assertSentBody(
        host: String, equals payload: EscalationPayload, dialect: Dialect,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let body = try XCTUnwrap(PayloadCaptureProtocol.body(forHost: host), "nothing was sent", file: file, line: line)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any], file: file, line: line)
        let sent = try XCTUnwrap(json["messages"] as? [[String: String]], file: file, line: line)
        switch dialect {
        case .openAI:
            XCTAssertEqual(sent, payload.messages, file: file, line: line)
            XCTAssertNil(json["system"], file: file, line: line)
        case .anthropic:
            XCTAssertEqual(sent, payload.messages.filter { $0["role"] != "system" }, file: file, line: line)
            XCTAssertEqual(json["system"] as? String, payload.systemPrompt, file: file, line: line)
        }
        XCTAssertEqual(json["max_tokens"] as? Int, payload.maxTokens, file: file, line: line)
    }

    private func assertRedactedAndCompressed(
        _ payload: EscalationPayload, host: String, question: String,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let raw = String(decoding: try XCTUnwrap(PayloadCaptureProtocol.body(forHost: host)), as: UTF8.self)
        XCTAssertFalse(raw.contains("alice@example"), file: file, line: line)
        XCTAssertFalse(raw.contains(Self.deployKey), file: file, line: line)
        XCTAssertTrue(payload.redactionEnabled, file: file, line: line)
        XCTAssertEqual(payload.redactedPIICount, 3, "the context's key and email, the question's email", file: file, line: line)
        XCTAssertTrue(payload.compression.keptText.contains("[REDACTED_KEY]"), file: file, line: line)
        XCTAssertTrue(payload.userMessage.contains(payload.compression.keptText), file: file, line: line)

        // keepRatio 0.3 of a 2,048-token window minus 64 is 550; escalate(to:)'s fixed 0.5 would give 960.
        let redactor = PIIRedactor()
        let redactedContext = redactor.redact(Self.context).redacted
        let redactedQuestion = redactor.redact(question).redacted
        let atPolicyRatio = ContextCompressor().compress(context: redactedContext, question: redactedQuestion, budgetTokens: 550)
        let atHalf = ContextCompressor().compress(context: redactedContext, question: redactedQuestion, budgetTokens: 960)
        XCTAssertEqual(payload.compression, atPolicyRatio, file: file, line: line)
        XCTAssertNotEqual(payload.compression, atHalf, file: file, line: line)
        XCTAssertLessThan(atPolicyRatio.compressedTokens, atPolicyRatio.originalTokens, file: file, line: line)
    }

    // MARK: - Parity

    func testOpenAICompatibleCloudSendsExactlyTheApprovedPayload() async throws {
        try XCTSkipUnless(NetworkMonitor.shared.isOnline, "cloud routing needs a network path (R3)")
        let host = stubbedHost()
        let gate = PayloadRecordingGate()
        let question = makeQuestion()

        let result = try await route(
            openAITarget(host: host, origin: .cloud),
            policy: EscalationPolicy(mode: .askEachTime, allowCloud: true, keepRatio: 0.3), gate: gate, question: question)

        let payload = try XCTUnwrap(gate.offers.first?.payload)
        XCTAssertEqual(gate.previews, [payload.compression])
        try assertSentBody(host: host, equals: payload, dialect: .openAI)
        try assertRedactedAndCompressed(payload, host: host, question: question)
        XCTAssertEqual(result?.answer, "ok")
        XCTAssertEqual(result?.redactedPIICount, payload.redactedPIICount)
        XCTAssertEqual(result?.compression, payload.compression)
    }

    func testAnthropicCloudSendsExactlyTheApprovedPayload() async throws {
        try XCTSkipUnless(NetworkMonitor.shared.isOnline, "cloud routing needs a network path (R3)")
        let host = stubbedHost()
        let gate = PayloadRecordingGate()
        let question = makeQuestion()

        let result = try await route(
            anthropicTarget(host: host),
            policy: EscalationPolicy(mode: .askEachTime, allowCloud: true, keepRatio: 0.3), gate: gate, question: question)

        let payload = try XCTUnwrap(gate.offers.first?.payload)
        XCTAssertEqual(gate.previews, [payload.compression])
        try assertSentBody(host: host, equals: payload, dialect: .anthropic)
        try assertRedactedAndCompressed(payload, host: host, question: question)
        XCTAssertEqual(result?.answer, "ok")
    }

    func testLANOfferRedactedForAllTargetsSendsExactlyTheApprovedPayload() async throws {
        let host = stubbedHost()
        let gate = PayloadRecordingGate()
        let question = makeQuestion()

        _ = try await route(
            openAITarget(host: host, origin: .localNetwork(.llamaServer)),
            policy: EscalationPolicy(mode: .askEachTime, keepRatio: 0.3, piiRedaction: .allTargets),
            gate: gate, question: question)

        let payload = try XCTUnwrap(gate.offers.first?.payload)
        XCTAssertEqual(gate.previews, [payload.compression])
        try assertSentBody(host: host, equals: payload, dialect: .openAI)
        try assertRedactedAndCompressed(payload, host: host, question: question)
    }

    func testAutoLANEscalationSendsVerbatimByDefaultWithoutAsking() async throws {
        let host = stubbedHost()
        let gate = PayloadRecordingGate()

        let result = try await route(
            openAITarget(host: host, origin: .localNetwork(.llamaServer)),
            policy: EscalationPolicy(mode: .autoWithConsentMemory), gate: gate, question: makeQuestion())

        XCTAssertTrue(gate.offers.isEmpty, "a LAN target under auto mode is not offered")
        XCTAssertEqual(result?.redactedPIICount, 0)
        let raw = String(decoding: try XCTUnwrap(PayloadCaptureProtocol.body(forHost: host)), as: UTF8.self)
        XCTAssertTrue(raw.contains(Self.deployKey))
    }

    func testCacheAnswersOnlyAnIdenticalPayload() async throws {
        let host = stubbedHost()
        let target = openAITarget(host: host, origin: .localNetwork(.llamaServer))
        let escalator = HybridEscalator(ledger: CostLedger())
        let question = makeQuestion()
        func payload(_ system: String?, _ maxTokens: Int) -> EscalationPayload {
            escalator.preparePayload(
                for: target, systemPrompt: system, context: "", question: question, maxTokens: maxTokens)
        }
        let french = payload("Answer in French.", 8)
        let first = try await escalator.escalate(french, to: target)
        XCTAssertFalse(first.fromCache)

        for variant in [payload("You are a lawyer.", 8), payload("Answer in French.", 512), payload(nil, 8)] {
            let result = try await escalator.escalate(variant, to: target)
            XCTAssertFalse(result.fromCache, "a different system prompt or maxTokens must be sent")
            try assertSentBody(host: host, equals: variant, dialect: .openAI)
        }
        let repeated = try await escalator.escalate(french, to: target)
        XCTAssertTrue(repeated.fromCache)
    }

    func testDeclinedOfferSendsNothing() async throws {
        let host = stubbedHost()
        let gate = PayloadRecordingGate(approves: false)
        do {
            _ = try await route(
                openAITarget(host: host, origin: .localNetwork(.llamaServer)),
                policy: EscalationPolicy(mode: .askEachTime), gate: gate, question: makeQuestion())
            XCTFail("the gate declines, so routeAndEscalate must throw")
        } catch TonamoxError.escalationDeclined {}
        XCTAssertEqual(gate.offers.count, 1)
        XCTAssertNil(PayloadCaptureProtocol.body(forHost: host))
    }

    // MARK: - Payload

    func testRedactionRunsBeforeCompressionSplitsAnEmail() {
        let target = RemoteTarget(provider: AnthropicProvider(apiKey: "unused"), modelID: "m",
                                  contextLength: 2048, origin: .cloud)
        let payload = HybridEscalator().preparePayload(
            for: target, context: Self.context, question: "Who gets the billing alerts?",
            maxTokens: 64, keepRatio: 0.3, redactPII: true)

        XCTAssertLessThan(payload.compression.compressedTokens, payload.compression.originalTokens)
        XCTAssertTrue(payload.compression.keptText.contains("[REDACTED_EMAIL]"))
        XCTAssertFalse(payload.userMessage.contains("alice@example"))
        XCTAssertFalse(payload.userMessage.contains("example. com"), "the email must not survive split in two")
    }

    func testPolicyRedactionScopePerTarget() {
        let cloud = RemoteTarget(provider: AnthropicProvider(apiKey: "unused"), modelID: "m", origin: .cloud)
        let lan = RemoteTarget(provider: AnthropicProvider(apiKey: "unused"), modelID: "m", origin: .localNetwork(.ollama))

        XCTAssertEqual(EscalationPolicy().piiRedaction, .cloudOnly)
        XCTAssertTrue(EscalationPolicy().redactsPII(for: cloud))
        XCTAssertFalse(EscalationPolicy().redactsPII(for: lan))
        XCTAssertTrue(EscalationPolicy(piiRedaction: .allTargets).redactsPII(for: lan))
        XCTAssertFalse(EscalationPolicy(piiRedaction: .never).redactsPII(for: cloud))
    }

    func testPolicySavedWithoutRedactionScopeDecodesAsCloudOnly() throws {
        let saved = #"{"mode":"askEachTime","allowCloud":true,"costCapUSDPerSession":2.5,"keepRatio":0.4}"#
        let policy = try JSONDecoder().decode(EscalationPolicy.self, from: Data(saved.utf8))
        XCTAssertEqual(policy, EscalationPolicy(mode: .askEachTime, allowCloud: true, costCapUSDPerSession: 2.5,
                                                keepRatio: 0.4, piiRedaction: .cloudOnly))

        let never = EscalationPolicy(mode: .askEachTime, piiRedaction: .never)
        XCTAssertEqual(try JSONDecoder().decode(EscalationPolicy.self, from: JSONEncoder().encode(never)), never)
    }
}

// MARK: - Test doubles

/// Approves (or declines) every offer and keeps what it was shown.
@MainActor
private final class PayloadRecordingGate: ConsentGate {
    private let approves: Bool
    private(set) var previews: [CompressionResult] = []
    private(set) var offers: [EscalationOffer] = []

    init(approves: Bool = true) { self.approves = approves }

    func requestConsent(target: RemoteTarget, preview: CompressionResult, projectedCostUSD: Decimal?) async -> Bool {
        XCTFail("routeAndEscalate must call requestConsent(target:preview:offer:)")
        return false
    }

    func requestConsent(target: RemoteTarget, preview: CompressionResult, offer: EscalationOffer) async -> Bool {
        previews.append(preview)
        offers.append(offer)
        return approves
    }
}

/// Keeps each request body by host and answers with a one-token SSE stream in the provider's dialect.
private final class PayloadCaptureProtocol: URLProtocol, @unchecked Sendable {
    private static let bodies = Mutex<[String: Data]>([:])

    private static let openAIStream = """
        data: {"choices":[{"delta":{"content":"ok"}}]}

        data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":1}}

        data: [DONE]


        """

    private static let anthropicStream = """
        event: message_start
        data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}

        event: content_block_delta
        data: {"type":"content_block_delta","delta":{"text":"ok"}}

        event: message_delta
        data: {"type":"message_delta","usage":{"output_tokens":1}}


        """

    static func session() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [PayloadCaptureProtocol.self]
        return URLSession(configuration: configuration)
    }

    static func body(forHost host: String) -> Data? {
        bodies.withLock { $0[host] }
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        guard let url = request.url, let host = url.host() else { return }
        let body = request.httpBody ?? Self.read(request.httpBodyStream)
        Self.bodies.withLock { $0[host] = body }
        let stream = url.path().hasSuffix("v1/messages") ? Self.anthropicStream : Self.openAIStream
        if let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1",
                                          headerFields: ["Content-Type": "text/event-stream"]) {
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        }
        client?.urlProtocol(self, didLoad: Data(stream.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    // URLSession hands a protocol the body as a stream.
    private static func read(_ stream: InputStream?) -> Data {
        guard let stream else { return Data() }
        stream.open()
        defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while stream.hasBytesAvailable {
            let count = stream.read(&buffer, maxLength: buffer.count)
            guard count > 0 else { break }
            data.append(buffer, count: count)
        }
        return data
    }
}
