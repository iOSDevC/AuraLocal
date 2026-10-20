import Foundation

// MARK: - FitEstimate

/// Memory fit on one device, from ``HardwareAnalyzer``.
public struct FitEstimate: Sendable, Equatable {
    public let rating: ModelFitLevel
    /// Weights + runtime overhead + KV cache at 2048 tokens (``Model/estimatedRuntimeMemoryGB``), or the
    /// diffusion peak for image pipelines.
    public let requiredGB: Double
    public let budgetGB: Double
    /// Bandwidth-bound decode estimate; `nil` when the device's bandwidth is unknown. For a mixture of experts,
    /// the low end of ``tokensPerSecondRange``.
    public let tokensPerSecond: Double?
    /// Uncalibrated decode range of a mixture of experts; `nil` for dense models.
    public let tokensPerSecondRange: ClosedRange<Double>?

    public init(rating: ModelFitLevel, requiredGB: Double, budgetGB: Double, tokensPerSecond: Double?,
                tokensPerSecondRange: ClosedRange<Double>? = nil) {
        self.rating = rating
        self.requiredGB = requiredGB
        self.budgetGB = budgetGB
        self.tokensPerSecond = tokensPerSecond
        self.tokensPerSecondRange = tokensPerSecondRange
    }

    /// `Good · 12.0 of 20.0 GB`.
    public var summary: String {
        String(format: "%@ · %.1f of %.1f GB", rating.label, requiredGB, budgetGB)
    }

    /// `~18 tok/s`, or `~37–75 tok/s (uncalibrated MoE range)`; `nil` without an estimate.
    public var decodeText: String? {
        if let range = tokensPerSecondRange {
            return String(format: "~%.0f–%.0f tok/s (uncalibrated MoE range)", range.lowerBound, range.upperBound)
        }
        return tokensPerSecond.map { String(format: "~%.0f tok/s", $0) }
    }
}

// MARK: - QuantFit

/// One GGUF quant with its own fit.
public struct QuantFit: Sendable, Equatable, Identifiable {
    public let option: GGUFQuantGroup
    public let memory: FitEstimate

    public var id: String { option.id }
    /// Split quants are listed for completeness but Tonamox cannot load them.
    public var isUsable: Bool { !option.isSplit }

    /// The largest single-file quant with at least 20 % headroom, else the best-fitting single-file quant.
    public static func recommended(in fits: [QuantFit]) -> QuantFit? {
        let usable = fits.filter(\.isUsable)
        let comfortable = usable.filter { $0.memory.rating <= .good }
        return comfortable.max { $0.option.totalBytes < $1.option.totalBytes }
            ?? usable.min { ($0.memory.rating, $0.option.totalBytes) < ($1.memory.rating, $1.option.totalBytes) }
    }
}

// MARK: - ArchitectureFacts

/// A short description of the model for display.
public struct ArchitectureFacts: Sendable, Equatable {
    /// MLX `model_type` or GGUF `general.architecture`.
    public let family: String?
    public let layers: Int?
    public let trainedContext: Int?
    public let kvHeads: Int?
    public let headDim: Int?
    public let hasVision: Bool
    /// `4-bit` for MLX; the quant labels for GGUF.
    public let quantization: String?
    /// Cache per layer type, from `config.json` or a GGUF header that lists its sliding-window layers; `nil` when
    /// the layer count is unknown and for a GGUF sized uniformly.
    public let kvLayers: KVCacheLayout?
    /// Mixture-of-experts routing measured from the tensor sizes; `nil` for dense models.
    public let experts: ExpertRouting?
}

// MARK: - CompatibilityReport

/// The checker's answer for one repository on one device.
public struct CompatibilityReport: Sendable {
    public let repoID: String
    public let target: DevicePreset
    public let weightFormat: DetectedFormat
    /// How Tonamox would load it (`text` / `vision`); `nil` when it cannot.
    public let modelCategory: Model.Category?
    public let status: CompatibilityVerdict
    /// Blockers first, then caveats, then info.
    public let findings: [CompatibilityFinding]
    public let overview: ArchitectureFacts
    /// Bytes loaded for MLX / image pipelines (weight-map files).
    public let weightsBytes: Int64?
    public let weightsFit: FitEstimate?
    /// Every GGUF quant, smallest first, each with its fit.
    public let quantFits: [QuantFit]
    /// The `models.json` entry, when the model can run.
    public let suggestedEntry: CatalogEntry?
    /// `cardData.license`, `license_name` and HF's gating mode, as listed.
    public let licenseID: String?
    public let licenseName: String?
    public let gatedMode: String?

    public var blockers: [CompatibilityFinding] { findings.filter { $0.level == .blocker } }
    public var caveats: [CompatibilityFinding] { findings.filter { $0.level == .caveat } }

    /// The first thing a user should read: the first blocker or caveat, else how it fits.
    public var headline: String {
        if let first = blockers.first ?? caveats.first { return first.title }
        if status == .unknown { return "Could not read the repository" }
        if let fit = bestFit { return "\(modelCategory?.rawValue.capitalized ?? "Model") · \(fit.summary)" }
        return status.label
    }

    /// The fit that decides: the whole model for MLX, the recommended quant for GGUF.
    public var bestFit: FitEstimate? {
        weightsFit ?? QuantFit.recommended(in: quantFits)?.memory
    }

    public var huggingFaceURL: URL? { URL(string: "https://huggingface.co/\(repoID)") }

    /// `12.34 GB`, in binary gigabytes like the fit math.
    public static func gigabytesText(_ bytes: Int64) -> String {
        String(format: "%.2f GB", Double(bytes) / 1_073_741_824)
    }
}

// MARK: - Evaluator

/// Turns a fetched snapshot into a report for a device. Pure: re-run it to switch devices without refetching.
public enum CompatibilityEvaluator {

    public static func evaluate(_ snapshot: RepoSnapshot, on target: DevicePreset) -> CompatibilityReport {
        guard snapshot.listing != nil else { return unreadable(snapshot, target: target) }
        let input = RuleInput(snapshot: snapshot, target: target)

        let weightsBytes = input.weightFormat == .gguf ? nil : input.safetensorsBytes
        let weightsFit = weightsBytes.flatMap { weightsEstimate(input, bytes: $0) }
        let quantFits = input.weightFormat == .gguf ? input.quantGroups.map { quantFit($0, input: input) } : []

        var findings = CompatibilityRules.all.flatMap { $0.findings(for: input) }
        findings += fitFindings(input, weightsFit: weightsFit, quantFits: quantFits)
        findings += layoutFindings(input, weightsFit: weightsFit)
        let hasBlocker = findings.contains { $0.level == .blocker }
        let category = hasBlocker ? nil : input.inferredCategory
        let entry = hasBlocker
            ? nil : suggestedEntry(input, category: category, weightsBytes: weightsBytes, quantFits: quantFits)
        if let entry, let path = entry.ggufFilename, path.contains("/") {
            findings.append(.caveat("entry", "Suggested quant sits in a subfolder",
                "`\(path)`: the catalog downloader creates only the repo folder, so moving the file into place is untested; "
                + "`Model.fromHuggingFaceURL` handles subfolder URLs."))
        }

        return CompatibilityReport(
            repoID: snapshot.repoID, target: target, weightFormat: input.weightFormat, modelCategory: category,
            status: verdict(input, findings: findings), findings: sorted(findings),
            overview: overview(input, category: category), weightsBytes: weightsBytes, weightsFit: weightsFit,
            quantFits: quantFits, suggestedEntry: entry, licenseID: input.listing?.licenseID,
            licenseName: input.listing?.licenseName, gatedMode: input.listing?.gatedMode)
    }

    // MARK: Verdict

    static func verdict(_ input: RuleInput, findings: [CompatibilityFinding]) -> CompatibilityVerdict {
        if findings.contains(where: { $0.level == .blocker }) { return .notRunnable }
        let missingConfig = input.weightFormat == .mlx && input.settings == nil
        let missingArchitecture = input.weightFormat == .gguf && input.ggufArchitecture == nil
        if missingConfig || missingArchitecture { return .unknown }
        return findings.contains { $0.level == .caveat } ? .runnableWithCaveats : .runnable
    }

    private static func sorted(_ findings: [CompatibilityFinding]) -> [CompatibilityFinding] {
        findings.enumerated()
            .sorted { ($0.element.level, $0.offset) < ($1.element.level, $1.offset) }
            .map(\.element)
    }

    private static func unreadable(_ snapshot: RepoSnapshot, target: DevicePreset) -> CompatibilityReport {
        let problem = snapshot.problem(for: .repository)
        let title: String
        switch problem?.statusCode {
        case 404?: title = "Repository not found"
        case 401?, 403?: title = "Repository is private or needs a token"
        default: title = "Could not reach Hugging Face"
        }
        let finding = CompatibilityFinding.caveat("repository", title, problem?.message ?? "No response.")
        return CompatibilityReport(
            repoID: snapshot.repoID, target: target, weightFormat: .unknown, modelCategory: nil, status: .unknown,
            findings: [finding], overview: ArchitectureFacts(family: nil, layers: nil, trainedContext: nil, kvHeads: nil,
                                                            headDim: nil, hasVision: false, quantization: nil,
                                                            kvLayers: nil, experts: nil),
            weightsBytes: nil, weightsFit: nil, quantFits: [], suggestedEntry: nil, licenseID: nil, licenseName: nil,
            gatedMode: nil)
    }

    // MARK: Fit

    private static func weightsEstimate(_ input: RuleInput, bytes: Int64) -> FitEstimate? {
        guard bytes > 0 else { return nil }
        let profile = input.target.memoryProfile
        if input.weightFormat == .imageGeneration {
            let rating = HardwareAnalyzer.fitLevel(forWeightsBytes: Int(bytes), kind: .diffusion, profile: profile)
            let peak = HardwareAnalyzer.peakMemoryGB(forWeightsBytes: Int(bytes), kind: .diffusion)
            return FitEstimate(rating: rating, requiredGB: peak, budgetGB: profile.availableMemoryGB, tokensPerSecond: nil)
        }
        let model = sizingModel(format: .mlx, bytes: bytes, layers: input.layerCount,
                                kvHeads: input.kvHeadCount, headDim: input.headWidth,
                                kvLayers: input.kvLayout, experts: input.moeRouting)
        return estimate(model, profile: profile)
    }

    private static func quantFit(_ group: GGUFQuantGroup, input: RuleInput) -> QuantFit {
        let model = sizingModel(format: .gguf, bytes: group.totalBytes, layers: input.layerCount,
                                kvHeads: input.kvHeadCount, headDim: input.headWidth, kvLayers: input.kvLayout)
        return QuantFit(option: group, memory: estimate(model, profile: input.target.memoryProfile))
    }

    private static func estimate(_ model: Model, profile: HardwareProfile) -> FitEstimate {
        let assessment = HardwareAnalyzer.assess(model, profile: profile)
        return FitEstimate(rating: assessment.fitLevel, requiredGB: assessment.requiredMemoryGB,
                           budgetGB: assessment.availableMemoryGB,
                           tokensPerSecond: assessment.estimatedDecodeTokensPerSecond,
                           tokensPerSecondRange: assessment.estimatedDecodeTokensPerSecondRange)
    }

    /// A throwaway ``Model`` so ``HardwareAnalyzer/assess(_:profile:)`` can size the download.
    static func sizingModel(format: ModelFormat, bytes: Int64, layers: Int?, kvHeads: Int?, headDim: Int?,
                            kvLayers: KVCacheLayout? = nil, experts: ExpertRouting? = nil) -> Model {
        Model(id: "sizing", repoID: "sizing", displayName: "sizing", category: .text, domain: nil, docTags: false,
              format: format, approximateSizeMB: megabytes(bytes), isUncensored: false, license: nil, ggufFilename: nil,
              defaultDocumentPrompt: nil, numLayers: layers ?? 0, kvHeads: kvHeads ?? 0, headDim: headDim ?? 0,
              kvLayers: kvLayers, experts: experts, maxContextLength: nil, downloadURL: nil, localFileURL: nil)
    }

    /// Catalog megabytes are MiB: `HardwareAnalyzer` divides them by 1024 to get GiB.
    static func megabytes(_ bytes: Int64) -> Int {
        Int((Double(bytes) / 1_048_576).rounded())
    }

    private static func fitFindings(_ input: RuleInput, weightsFit: FitEstimate?,
                                    quantFits: [QuantFit]) -> [CompatibilityFinding] {
        let device = input.target.displayName
        if let weightsFit {
            guard [.mlx, .imageGeneration].contains(input.weightFormat) else {
                return [.info("fit", "Size as shipped", "\(weightsFit.summary) on \(device) — before any conversion.")]
            }
            return [fitFinding(weightsFit, device: device, subject: "The weights")]
        }
        guard !quantFits.isEmpty else { return [] }
        let usable = quantFits.filter(\.isUsable)
        let pool = usable.isEmpty ? quantFits : usable
        let loading = pool.filter { $0.memory.rating <= .marginal }.count
        let streaming = pool.filter { $0.memory.rating == .streamingRequired }.count
        let tally = "\(loading) of \(pool.count)\(usable.isEmpty ? "" : " single-file") quants load fully"
            + (streaming > 0 ? ", \(streaming) only by streaming." : ".")
        guard loading + streaming > 0,
              let best = QuantFit.recommended(in: pool) ?? pool.min(by: { rank($0) < rank($1) }) else {
            let smallest = pool.min { $0.option.totalBytes < $1.option.totalBytes }
            let label = smallest.map { $0.option.label ?? $0.option.firstPath } ?? "?"
            let needed = smallest.map { String(format: "%.1f GB", $0.memory.requiredGB) } ?? "?"
            return [.blocker("fit", "Too large for \(device)",
                "The smallest\(usable.isEmpty ? "" : " single-file") quant (\(label)) needs ≈\(needed); the budget is "
                + "\(input.target.budgetText). \(input.target.source)")]
        }
        let base = fitFinding(best.memory, device: device,
                              subject: "Recommended quant \(best.option.label ?? best.option.firstPath)")
        let title = base.level == .info ? "\(loading) of \(pool.count) quants fit \(device)" : base.title
        return [CompatibilityFinding(rule: "fit", level: base.level, title: title, detail: base.detail + " " + tally)]
    }

    /// Context sampled for the KV figure in the `kv` finding.
    static let kvSampleContext = 8192

    /// How the KV cache grows when not every layer keeps the whole context, and a mixture of experts' decode range.
    private static func layoutFindings(_ input: RuleInput, weightsFit: FitEstimate?) -> [CompatibilityFinding] {
        [kvFinding(input), moeFinding(input, weightsFit: weightsFit)].compactMap { $0 }
    }

    private static func kvFinding(_ input: RuleInput) -> CompatibilityFinding? {
        guard let layout = input.kvLayout, let heads = input.kvHeadCount, let width = input.headWidth else { return nil }
        let layers = input.layerCount ?? layout.cachedLayers
        guard layout.slidingLayers + layout.recurrentLayers > 0 || layout.cachedLayers != layers else { return nil }
        if layout.slidingLayers > 0, layout.slidingWindow == 0 {
            return fullSizeKVFinding(input, layout: layout, layers: layers, heads: heads, width: width)
        }
        let wholePrompt = layout.slidingLayers > 0 && layout.slidingPrefillStep == nil
        var parts = ["\(layout.fullAttentionLayers) of \(layers) layers keep the whole context"]
        if layout.slidingLayers > 0 {
            let peak = layout.slidingPrefillStep.map {
                " (up to \(layout.slidingWindow + $0 - 1) while the prompt is prefilled \($0) tokens at a time)"
            } ?? " once decoding starts; the runtime prefills the whole prompt in one pass, so until then they hold all of it"
            parts.append("\(layout.slidingLayers) keep the last \(layout.slidingWindow) tokens" + peak)
        }
        if layout.recurrentLayers > 0 {
            parts.append(String(format: "%d keep a fixed %.1f MB state", layout.recurrentLayers,
                                Double(layout.recurrentStateBytes) / 1_048_576))
        }
        let peak = layout.cacheBytes(contextLength: kvSampleContext, kvHeads: heads, headDim: width)
        let decoding = layout.decodingCacheBytes(contextLength: kvSampleContext, kvHeads: heads, headDim: width)
        let uniform = KVCacheLayout(fullAttentionLayers: layers)
            .cacheBytes(contextLength: kvSampleContext, kvHeads: heads, headDim: width)
        let sampled = peak == decoding
            ? String(format: "%.0f MB", peak / 1_048_576)
            : String(format: "%.0f MB at the prefill peak, %.0f MB once decoding starts", peak / 1_048_576,
                     decoding / 1_048_576)
        let sizes = String(format: ". At %d tokens: %@ (%.0f MB if every layer kept it).",
                           kvSampleContext, sampled, uniform / 1_048_576)
        let title = "KV cache grows on \(layout.fullAttentionLayers) of \(layers) layers"
        return .info("kv", wholePrompt ? title + " once decoding starts" : title, parts.joined(separator: ", ") + sizes)
    }

    /// llama.cpp sizes sliding-window layers for the whole context too (`swa_full`); they differ only in width.
    private static func fullSizeKVFinding(_ input: RuleInput, layout: KVCacheLayout, layers: Int, heads: Int,
                                          width: Int) -> CompatibilityFinding {
        let globalHeads = layout.fullAttentionKVHeads ?? heads
        let globalWidth = layout.fullAttentionHeadDim ?? width
        var parts = ["llama.cpp sizes every cached layer for the whole context: \(layout.fullAttentionLayers) global "
                     + "at \(globalHeads) × \(globalWidth), \(layout.slidingLayers) sliding-window at \(heads) × \(width)"]
        if layout.cachedLayers < layers {
            parts.append("\(layers - layout.cachedLayers) reuse an earlier layer's cache")
        }
        let sampled = layout.cacheBytes(contextLength: kvSampleContext, kvHeads: heads, headDim: width)
        var sizes = String(format: ". At %d tokens: %.0f MB", kvSampleContext, sampled / 1_048_576)
        if let headerHeads = input.ggufMetadata?.headCountKV, let headerWidth = input.ggufMetadata?.headDimension {
            let uniform = KVCacheLayout(fullAttentionLayers: layers)
                .cacheBytes(contextLength: kvSampleContext, kvHeads: headerHeads, headDim: headerWidth)
            sizes += String(format: " (%.0f MB with every layer at %d × %d).", uniform / 1_048_576, headerHeads,
                            headerWidth)
        } else {
            sizes += "."
        }
        return .info("kv", "KV cache: \(layout.fullAttentionLayers) global and \(layout.slidingLayers) sliding-window layers",
                     parts.joined(separator: ", ") + sizes)
    }

    private static func moeFinding(_ input: RuleInput, weightsFit: FitEstimate?) -> CompatibilityFinding? {
        guard let routing = input.moeRouting else { return nil }
        let read = String(format: "A token reads ≈%.2f GB of the weights (non-expert tensors plus %d/%d of the experts).",
                          Double(routing.activeSizeMB) / 1024, routing.expertsPerToken, routing.routedExperts)
        let speed = weightsFit?.decodeText.map { " Decode \($0) on \(input.target.displayName)." } ?? ""
        return .info("moe", "Mixture of experts: \(routing.expertsPerToken) of \(routing.routedExperts) per token",
                     read + speed + " The range stays uncalibrated until a MoE is measured.")
    }

    /// Best rating first; among equals, the largest (highest quality) quant.
    private static func rank(_ fit: QuantFit) -> (ModelFitLevel, Double) {
        (fit.memory.rating, -fit.memory.requiredGB)
    }

    private static func fitFinding(_ fit: FitEstimate, device: String, subject: String) -> CompatibilityFinding {
        let detail = "\(subject): \(fit.summary) on \(device)."
        switch fit.rating {
        case .tooLarge:
            return .blocker("fit", "Too large for \(device)", detail)
        case .streamingRequired:
            return .caveat("fit", "Only by layer streaming on \(device)", detail + " Decode is much slower than a full load.")
        case .marginal:
            return .caveat("fit", "Tight fit on \(device)", detail + " Under 20 % headroom.")
        case .good, .excellent:
            return .info("fit", "Fits \(device)", detail)
        }
    }

    // MARK: Entry & overview

    private static func suggestedEntry(_ input: RuleInput, category: Model.Category?, weightsBytes: Int64?,
                                       quantFits: [QuantFit]) -> CatalogEntry? {
        guard let category, let listing = input.listing else { return nil }
        let context = input.trainedContext.flatMap { $0 < CatalogEntry.contextCeiling ? $0 : nil }
        let uncensored = CatalogEntry.looksUncensored(repoID: listing.repoID, tags: listing.tags)
        let license = CatalogEntry.catalogLicense(id: listing.licenseID, name: listing.licenseName)
        switch input.weightFormat {
        case .mlx:
            guard let weightsBytes else { return nil }
            let bits = input.settings?.quantizationBits
            let uniform = KVCacheLayout(fullAttentionLayers: input.layerCount ?? -1)
            let layout = input.kvLayout.flatMap { $0 == uniform ? nil : $0 }
            return CatalogEntry(
                id: CatalogEntry.identifier(repoID: listing.repoID, quant: nil), repoID: listing.repoID,
                displayName: CatalogEntry.displayName(repoID: listing.repoID, format: .mlx, quant: nil, bits: bits),
                modelCategory: category, weightFormat: .mlx, approximateSizeMB: megabytes(weightsBytes),
                isUncensored: uncensored, ggufFilename: nil, numLayers: input.layerCount ?? 0,
                kvHeads: input.kvHeadCount ?? 0, headDim: input.headWidth ?? 0, maxContextLength: context,
                kvLayers: layout, experts: input.moeRouting, license: license)
        case .gguf:
            guard let pick = QuantFit.recommended(in: quantFits) else { return nil }
            return CatalogEntry(
                id: CatalogEntry.identifier(repoID: listing.repoID, quant: pick.option.label ?? "gguf"),
                repoID: listing.repoID,
                displayName: CatalogEntry.displayName(repoID: listing.repoID, format: .gguf, quant: pick.option.label, bits: nil),
                modelCategory: .text, weightFormat: .gguf, approximateSizeMB: megabytes(pick.option.totalBytes),
                isUncensored: uncensored, ggufFilename: pick.option.firstPath, numLayers: input.layerCount ?? 0,
                kvHeads: input.kvHeadCount ?? 0, headDim: input.headWidth ?? 0, maxContextLength: context,
                kvLayers: input.kvLayout, license: license)
        default:
            return nil
        }
    }

    private static func overview(_ input: RuleInput, category: Model.Category?) -> ArchitectureFacts {
        let config = input.settings
        switch input.weightFormat {
        case .gguf:
            let labels = input.quantGroups.compactMap(\.label)
            return ArchitectureFacts(
                family: input.ggufArchitecture, layers: input.layerCount, trainedContext: input.trainedContext,
                kvHeads: input.kvHeadCount, headDim: input.headWidth, hasVision: false,
                quantization: labels.isEmpty ? nil : "\(labels.count) quants", kvLayers: input.kvLayout, experts: nil)
        default:
            return ArchitectureFacts(
                family: config?.modelType, layers: input.layerCount, trainedContext: input.trainedContext,
                kvHeads: input.kvHeadCount, headDim: input.headWidth,
                hasVision: category == .vision || (category == nil && input.hasVisionTensors == true),
                quantization: config?.quantizationBits.map { "\($0)-bit" } ?? config?.foreignQuantMethod,
                kvLayers: input.kvLayout, experts: input.moeRouting)
        }
    }
}
