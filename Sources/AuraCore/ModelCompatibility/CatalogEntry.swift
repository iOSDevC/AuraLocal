import Foundation

/// A `models.json` entry for a model the checker found runnable, with the catalog's fields and key order.
///
/// `kvHeads` / `headDim` come from `config.json` (MLX) or the file header (GGUF), 0 when neither has them.
/// `maxContextLength` is set only below 32768 tokens. MLX entries add `kvLayers` when not every layer keeps the
/// whole context, and `experts` for a mixture of experts whose tensor sizes were read.
public struct CatalogEntry: Sendable, Equatable {
    public let id: String
    public let repoID: String
    public let displayName: String
    public let modelCategory: Model.Category
    public let docTags: Bool
    public let weightFormat: ModelFormat
    public let approximateSizeMB: Int
    public let isUncensored: Bool
    public let ggufFilename: String?
    public let defaultDocumentPrompt: String?
    public let numLayers: Int
    public let kvHeads: Int
    public let headDim: Int
    public let maxContextLength: Int?
    public let kvLayers: KVCacheLayout?
    public let experts: ExpertRouting?

    public init(id: String, repoID: String, displayName: String, modelCategory: Model.Category,
                weightFormat: ModelFormat, approximateSizeMB: Int, isUncensored: Bool, ggufFilename: String?,
                numLayers: Int, kvHeads: Int, headDim: Int, maxContextLength: Int?,
                kvLayers: KVCacheLayout? = nil, experts: ExpertRouting? = nil) {
        self.id = id
        self.repoID = repoID
        self.displayName = displayName
        self.modelCategory = modelCategory
        self.docTags = false
        self.weightFormat = weightFormat
        self.approximateSizeMB = approximateSizeMB
        self.isUncensored = isUncensored
        self.ggufFilename = ggufFilename
        self.defaultDocumentPrompt = nil
        self.numLayers = numLayers
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.maxContextLength = maxContextLength
        self.kvLayers = kvLayers
        self.experts = experts
    }

    /// Contexts at or above this need no `maxContextLength` (the memory budget caps them first).
    public static let contextCeiling = 32_768

    /// The entry as `models.json` writes it: explicit `null`s, catalog key order. `indent` shifts every line
    /// (the catalog nests entries 4 spaces deep).
    public func jsonText(indent: Int = 0) -> String {
        let pad = String(repeating: " ", count: indent)
        var fields = [
            ("id", Self.quoted(id)),
            ("repoID", Self.quoted(repoID)),
            ("displayName", Self.quoted(displayName)),
            ("category", Self.quoted(modelCategory.rawValue)),
            ("docTags", docTags ? "true" : "false"),
            ("format", Self.quoted(weightFormat.rawValue)),
            ("approximateSizeMB", "\(approximateSizeMB)"),
            ("isUncensored", isUncensored ? "true" : "false"),
            ("ggufFilename", Self.quoted(ggufFilename)),
            ("defaultDocumentPrompt", Self.quoted(defaultDocumentPrompt)),
            ("numLayers", "\(numLayers)"),
            ("kvHeads", "\(kvHeads)"),
            ("headDim", "\(headDim)"),
        ]
        if let maxContextLength { fields.append(("maxContextLength", "\(maxContextLength)")) }
        if let kvLayers { fields.append(("kvLayers", Self.object(kvLayers))) }
        if let experts { fields.append(("experts", Self.object(experts))) }
        let body = fields.map { "\(pad)  \"\($0.0)\": \($0.1)" }.joined(separator: ",\n")
        return "\(pad){\n\(body)\n\(pad)}"
    }

    /// The entry decoded as the ``Model`` the registry would build from it.
    public func decodedModel() -> Model? {
        try? JSONDecoder().decode(Model.self, from: Data(jsonText().utf8))
    }

    /// `{"fullAttentionLayers": 6, …}` on one line, so a catalog entry stays one screen tall.
    private static func object(_ layout: KVCacheLayout) -> String {
        var pairs = [
            ("fullAttentionLayers", layout.fullAttentionLayers), ("slidingLayers", layout.slidingLayers),
            ("slidingWindow", layout.slidingWindow), ("recurrentLayers", layout.recurrentLayers),
            ("recurrentStateBytes", layout.recurrentStateBytes),
        ]
        if let heads = layout.fullAttentionKVHeads { pairs.append(("fullAttentionKVHeads", heads)) }
        if let width = layout.fullAttentionHeadDim { pairs.append(("fullAttentionHeadDim", width)) }
        return "{" + pairs.map { "\"\($0.0)\": \($0.1)" }.joined(separator: ", ") + "}"
    }

    private static func object(_ routing: ExpertRouting) -> String {
        "{\"routedExperts\": \(routing.routedExperts), \"expertsPerToken\": \(routing.expertsPerToken), "
            + "\"activeSizeMB\": \(routing.activeSizeMB)}"
    }

    // MARK: Naming

    /// `mlx-community/MiniCPM5-1B-4bit` → `minicpm5_1b_4bit`; GGUF entries end in `_<quant>_gguf`.
    static func identifier(repoID: String, quant: String?) -> String {
        let name = repoID.split(separator: "/").last.map(String.init) ?? repoID
        var slug = snakeCase(name)
        guard let quant else { return slug }
        for suffix in ["_gguf", "_mlx"] where slug.hasSuffix(suffix) {
            slug.removeLast(suffix.count)
        }
        return "\(slug)_\(snakeCase(quant))_gguf"
    }

    /// `Qwen2.5-7B-Instruct-GGUF` + `Q4_K_M` → `Qwen2.5 7B Instruct (GGUF Q4_K_M)`.
    static func displayName(repoID: String, format: ModelFormat, quant: String?, bits: Int?) -> String {
        let name = repoID.split(separator: "/").last.map(String.init) ?? repoID
        let words = name.split(whereSeparator: { $0 == "-" || $0 == "_" })
            .filter { !["gguf", "mlx"].contains($0.lowercased()) }
            .joined(separator: " ")
        switch format {
        case .gguf: return "\(words) (GGUF\(quant.map { " \($0)" } ?? ""))"
        case .mlx: return "\(words) (MLX\(bits.map { " \($0)-bit" } ?? ""))"
        }
    }

    static func looksUncensored(repoID: String, tags: [String]) -> Bool {
        let haystack = ([repoID] + tags).joined(separator: " ").lowercased()
        return haystack.contains("uncensored") || haystack.contains("abliterated")
    }

    private static func snakeCase(_ text: String) -> String {
        let mapped = text.lowercased().map { $0.isLetter || $0.isNumber ? String($0) : "_" }.joined()
        return mapped.split(separator: "_").joined(separator: "_")
    }

    private static func quoted(_ text: String?) -> String {
        guard let text else { return "null" }
        let escaped = text.unicodeScalars.map { scalar -> String in
            switch scalar {
            case "\"": return "\\\""
            case "\\": return "\\\\"
            case "\n": return "\\n"
            case "\r": return "\\r"
            case "\t": return "\\t"
            default:
                return scalar.value < 0x20 ? String(format: "\\u%04x", scalar.value) : String(scalar)
            }
        }.joined()
        return "\"\(escaped)\""
    }
}

public extension CatalogEntry {
    /// The same entry pointing at another quant of the same GGUF repo, sized from it. `nil` for an MLX entry, a
    /// split quant, or a quant in a subfolder (the downloaders only create the repo folder).
    func replacingQuant(_ quant: GGUFQuantGroup) -> CatalogEntry? {
        guard weightFormat == .gguf, !quant.isSplit, !quant.firstPath.isEmpty, !quant.firstPath.contains("/") else {
            return nil
        }
        return CatalogEntry(
            id: Self.identifier(repoID: repoID, quant: quant.label ?? "gguf"), repoID: repoID,
            displayName: Self.displayName(repoID: repoID, format: .gguf, quant: quant.label, bits: nil),
            modelCategory: modelCategory, weightFormat: .gguf,
            approximateSizeMB: CompatibilityEvaluator.megabytes(quant.totalBytes), isUncensored: isUncensored,
            ggufFilename: quant.firstPath, numLayers: numLayers, kvHeads: kvHeads, headDim: headDim,
            maxContextLength: maxContextLength, kvLayers: kvLayers, experts: experts)
    }
}
