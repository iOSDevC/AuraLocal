import Foundation

/// Redacts obvious secrets / PII from text before it leaves the device — a
/// best-effort privacy backstop (not a guarantee). Patterns are high-precision
/// (distinctive prefixes/formats) so they don't mangle source code, which the
/// security domain routinely sends.
public struct PIIRedactor: Sendable {
    public struct Result: Sendable {
        public let redacted: String
        public let count: Int
    }

    public init() {}

    private static let rules: [(NSRegularExpression, String)] = {
        func rx(_ pattern: String) -> NSRegularExpression {
            // Patterns are static and known-valid.
            try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
        }
        return [
            (rx(#"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"#), "[REDACTED_EMAIL]"),
            (rx(#"\b(?:sk|pk|rk)[-_][A-Za-z0-9]{16,}\b"#), "[REDACTED_KEY]"),      // sk-…, pk_…
            (rx(#"\bBearer\s+[A-Za-z0-9._\-]{16,}\b"#), "Bearer [REDACTED_TOKEN]"),
            (rx(#"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"#), "[REDACTED_TOKEN]"),        // Slack
            (rx(#"\bAKIA[0-9A-Z]{16}\b"#), "[REDACTED_KEY]"),                       // AWS access key id
            (rx(#"\bgh[posru]_[A-Za-z0-9]{20,}\b"#), "[REDACTED_TOKEN]"),           // GitHub
        ]
    }()

    /// Replace matches with placeholders; returns the redacted text and the count.
    public func redact(_ text: String) -> Result {
        var out = text
        var total = 0
        for (regex, replacement) in Self.rules {
            let range = NSRange(out.startIndex..<out.endIndex, in: out)
            let matches = regex.numberOfMatches(in: out, options: [], range: range)
            guard matches > 0 else { continue }
            total += matches
            out = regex.stringByReplacingMatches(in: out, options: [], range: range, withTemplate: replacement)
        }
        return Result(redacted: out, count: total)
    }
}
