import Foundation
import TonamoxCore

/// One folder of a repo judged for one device: the findings, the verdict and the defaults to run it with.
public struct CoreMLDiffusionVariant: Sendable, Equatable, Identifiable {
    public let repoID: String
    /// The commit the listing and the probe were read at; a download is pinned to it.
    public let revision: String
    public let folder: CoreMLDiffusionFolder
    /// `nil` when the Unet's `metadata.json` was not read (rule R1).
    public let unet: CoreMLUnetFacts?
    public let licenseName: String
    public let licenseURL: URL
    /// Blockers first.
    public let findings: [CompatibilityFinding]
    public let verdict: CompatibilityVerdict
    /// Memory fit from the exact download size; `nil` when the listing had no sizes.
    public let fit: ModelFitLevel?
    public let preferredComputeUnits: DiffusionComputeUnits
    public let recommendedSteps: Int
    public let recommendedGuidance: Float

    public var id: String { "\(repoID)/\(folder.path)" }
}

/// The rules that decide whether `ml-stable-diffusion` 1.1.1, as TonamoxImageGen drives it, can run a folder. Pure:
/// tests feed recorded `metadata.json` files. Each finding's `rule` is its id (`R1`…`R19`, `FIT`).
public enum CoreMLDiffusionRules {
    static let sdInputs: Set<String> = ["sample", "timestep", "encoder_hidden_states"]
    static let sdxlInputs: Set<String> = ["time_ids", "text_embeds"]
    static let sdxlEmbeddingWidth = 2048
    static let gibibyte: Int64 = 1_073_741_824
    static let defaultSteps = (steps: 20, guidance: Float(7.5))
    static let latentConsistencySteps = (steps: 8, guidance: Float(1.0))
    static let distilledSteps = (steps: 4, guidance: Float(1.0))
    static let distilledKeywords = ["turbo", "lightning", "hyper", "dmd", "tcd"]

    /// `osMajor` is the target's OS major version, compared with the Unet's availability and op set (R8).
    public static func evaluate(_ folder: CoreMLDiffusionFolder, unet: CoreMLUnetFacts?, listing: HFRepoInfo,
                                on target: DevicePreset, osMajor: Int) -> CoreMLDiffusionVariant {
        evaluate(folder, unet: unet, listing: listing, on: target, osMajor: osMajor, unetProblem: nil)
    }

    /// `unetProblem` says why `unet` is `nil` (a failed or skipped probe).
    static func evaluate(_ folder: CoreMLDiffusionFolder, unet: CoreMLUnetFacts?, listing: HFRepoInfo,
                         on target: DevicePreset, osMajor: Int, unetProblem: String?) -> CoreMLDiffusionVariant {
        let computeUnits: DiffusionComputeUnits = unet?.usesSplitEinsum == false ? .gpu : .neuralEngine
        let names = [listing.repoID] + listing.tags
        let fit = folder.downloadBytes.map {
            HardwareAnalyzer.fitLevel(forWeightsBytes: Int($0), kind: .diffusion, profile: target.memoryProfile)
        }
        var findings = listingFindings(folder, listing: listing)
        if let unet {
            findings += unetFindings(unet, folder: folder, names: names, target: target, osMajor: osMajor,
                                     computeUnits: computeUnits)
        } else if listing.gatedMode == nil {
            findings.append(unverified(unetProblem ?? "metadata.json was not read."))
        }
        if listing.revision == nil {
            findings.append(unverified("The listing has no commit sha to pin the download to."))
        }
        findings += nameFindings(names) + licenseFindings(listing)
        if folder.hasSafetyChecker {
            findings.append(info("R19", "The repo's optional safety checker is not downloaded; images are not filtered",
                                 "SafetyChecker.mlmodelc stays on Hugging Face, so nothing screens the images."))
        }
        findings.append(fitFinding(bytes: folder.downloadBytes, fit: fit, target: target))

        let sorted = findings.enumerated()
            .sorted { ($0.element.level, $0.offset) < ($1.element.level, $1.offset) }
            .map(\.element)
        let defaults = recommendedDefaults(names)
        let terms = license(of: listing)
        return CoreMLDiffusionVariant(
            repoID: listing.repoID, revision: listing.revision ?? "", folder: folder, unet: unet,
            licenseName: terms.name, licenseURL: terms.url, findings: sorted,
            verdict: verdict(sorted, fit: fit), fit: fit, preferredComputeUnits: computeUnits,
            recommendedSteps: defaults.steps, recommendedGuidance: defaults.guidance)
    }

    /// Any blocker → won't run; else anything unverified → unknown; else any caveat → runs with caveats.
    static func verdict(_ findings: [CompatibilityFinding], fit: ModelFitLevel?) -> CompatibilityVerdict {
        if findings.contains(where: { $0.level == .blocker }) { return .notRunnable }
        if fit == nil || findings.contains(where: { $0.rule == "R1" }) { return .unknown }
        if findings.contains(where: { $0.level == .caveat }) { return .runnableWithCaveats }
        return .runnable
    }

    /// Runs before caveats, then the Neural Engine before the GPU, then the smaller download.
    static func isPreferred(_ lhs: CoreMLDiffusionVariant, over rhs: CoreMLDiffusionVariant) -> Bool {
        let order: [CompatibilityVerdict] = [.runnable, .runnableWithCaveats, .unknown, .notRunnable]
        let lhsRank = order.firstIndex(of: lhs.verdict) ?? order.count
        let rhsRank = order.firstIndex(of: rhs.verdict) ?? order.count
        if lhsRank != rhsRank { return lhsRank < rhsRank }
        if lhs.preferredComputeUnits != rhs.preferredComputeUnits { return lhs.preferredComputeUnits == .neuralEngine }
        return (lhs.folder.downloadBytes ?? .max) < (rhs.folder.downloadBytes ?? .max)
    }

    // MARK: Listing (R5 TextEncoder2-only, R16, R17)

    static func listingFindings(_ folder: CoreMLDiffusionFolder, listing: HFRepoInfo) -> [CompatibilityFinding] {
        var findings: [CompatibilityFinding] = []
        if listing.gatedMode != nil {
            findings.append(blocker("R17", "Gated repo: Tonamox downloads image models without a Hugging Face token",
                                    "Its files need an account that accepted the terms; nothing was probed."))
        }
        if folder.isRepoRoot {
            findings.append(blocker("R16", "Resources at the repo root are not supported yet",
                                    "Tonamox downloads one subfolder of a repo; this one keeps the models at "
                                    + "the top."))
        }
        if folder.hasOnlyTextEncoder2 {
            findings.append(sdxl("The folder has TextEncoder2 only, an SDXL text encoder."))
        }
        return findings
    }

    // MARK: Unet (R2–R12)

    static func unetFindings(_ unet: CoreMLUnetFacts, folder: CoreMLDiffusionFolder, names: [String],
                             target: DevicePreset, osMajor: Int,
                             computeUnits: DiffusionComputeUnits) -> [CompatibilityFinding] {
        let inputs = Set(unet.inputShapes.keys)
        guard sdInputs.isSubset(of: inputs) else {
            return [blocker("R2", "Not an Apple ml-stable-diffusion Unet",
                            "Inputs: \(inputs.sorted().joined(separator: ", ")); the runtime feeds sample, "
                            + "timestep and encoder_hidden_states.")]
        }
        let width = unet.textEmbeddingWidth
        var findings = shapeFindings(unet) + embeddingFindings(width)
        if unet.isStateful {
            findings.append(blocker("R6", "Stateful Unet",
                                    "It declares iOS 18 MLState inputs, which ml-stable-diffusion 1.1.1 does not "
                                    + "drive."))
        }
        let allowed = width == sdxlEmbeddingWidth ? sdInputs.union(sdxlInputs) : sdInputs
        let extra = inputs.subtracting(allowed).sorted()
        if !extra.isEmpty {
            findings.append(blocker("R7", "Unexpected Unet inputs (ControlNet residuals or an unknown export)",
                                    "Extra inputs: \(extra.joined(separator: ", ")); TonamoxImageGen passes none."))
        }
        findings += osFindings(unet, target: target, osMajor: osMajor)
        if computeUnits == .gpu {
            findings.append(caveat("R10", "Attention for the GPU: runs on the GPU, not the Neural Engine",
                                   "The Unet uses ORIGINAL attention (no einsum ops), so TonamoxImageGen runs it with "
                                   + "CPU and GPU."))
        }
        findings += chunkingFindings(unet, folder: folder, target: target, computeUnits: computeUnits)
        if isVPrediction(unet, width: width, names: names) {
            findings.append(blocker("R12", "v-prediction model; the runtime's schedulers assume epsilon",
                                    "SD 2.x at 768 px that is not a base model, or a v-pred name. DPM-Solver++ in "
                                    + "ml-stable-diffusion 1.1.1 assumes epsilon, so images come out wrong."))
        }
        return findings
    }

    /// R3 and R9.
    static func shapeFindings(_ unet: CoreMLUnetFacts) -> [CompatibilityFinding] {
        guard let sample = unet.inputShapes["sample"], sample.count == 4 else {
            return [blocker("R3", "Unexpected latent shape",
                            "sample is \(describe(unet.inputShapes["sample"])); the runtime expects [2, 4, h, w].")]
        }
        var findings: [CompatibilityFinding] = []
        let timestep = unet.inputShapes["timestep"] ?? []
        if sample[0] != 2 || timestep != [2] {
            findings.append(blocker("R3", "Converted without classifier-free guidance; the runtime always runs batch 2",
                                    "sample is \(describe(sample)) and timestep \(describe(timestep)); "
                                    + "ml-stable-diffusion 1.1.1 always feeds batch 2."))
        }
        if sample[1] == 8 || sample[1] == 9 {
            findings.append(blocker("R3", "Inpainting / instruct-pix2pix Unet",
                                    "sample has \(sample[1]) channels; text-to-image needs 4 and the runtime has "
                                    + "no inpainting."))
        } else if sample[1] != 4 {
            findings.append(blocker("R3", "Unexpected latent channels",
                                    "sample has \(sample[1]) channels; Stable Diffusion's latent has 4."))
        }
        if sample[2] != sample[3] {
            findings.append(blocker("R9", "Non-square output is not supported",
                                    "The latent is \(sample[2])×\(sample[3]); TonamoxImageGen models a square side."))
        }
        return findings
    }

    /// R4 and R5.
    static func embeddingFindings(_ width: Int?) -> [CompatibilityFinding] {
        switch width {
        case 768, 1024:
            return []
        case sdxlEmbeddingWidth:
            return [sdxl("encoder_hidden_states is 2048 wide, an SDXL Unet.")]
        default:
            let text = width.map(String.init) ?? "missing"
            return [blocker("R4", "Unknown text encoder width",
                            "encoder_hidden_states width is \(text), neither SD 1.x (768) nor SD 2.x (1024).")]
        }
    }

    /// R8: availability, and on iOS also the newest op set.
    static func osFindings(_ unet: CoreMLUnetFacts, target: DevicePreset, osMajor: Int) -> [CompatibilityFinding] {
        let platform = target.platform == .iOS ? "iOS" : "macOS"
        let opSet = target.platform == .iOS ? unet.highestIOSOpSet : nil
        guard let needed = [unet.minimumOS[platform], opSet].compactMap({ $0 }).max(), needed > osMajor else {
            return []
        }
        return [blocker("R8", "Needs \(platform) \(needed)",
                        "The Unet requires \(platform) \(needed) (availability or op set); the target runs "
                        + "\(platform) \(osMajor).")]
    }

    /// R11: Apple's README requires chunking for the Neural Engine on iOS unless weights are 6-bit or less.
    static func chunkingFindings(_ unet: CoreMLUnetFacts, folder: CoreMLDiffusionFolder, target: DevicePreset,
                                 computeUnits: DiffusionComputeUnits) -> [CompatibilityFinding] {
        guard unet.storagePrecision == "Float16", !folder.isChunked, computeUnits == .neuralEngine,
              let bytes = folder.unetBytes, bytes >= gibibyte else { return [] }
        let title = "Apple requires a chunked Unet for the Neural Engine on iPhone unless weights are ≤ 6-bit"
        let detail = "The Unet is fp16, unchunked, \(gigabytes(bytes))."
        if target.platform == .iOS { return [blocker("R11", title, detail)] }
        return [info("R11", title, detail + " A Mac runs it.")]
    }

    static func isVPrediction(_ unet: CoreMLUnetFacts, width: Int?, names: [String]) -> Bool {
        let lowered = names.map { $0.lowercased() }
        if lowered.contains(where: { text in ["v-pred", "vpred", "v_pred"].contains { text.contains($0) } }) {
            return true
        }
        let isBase = unet.version?.lowercased().contains("base") ?? false
        return width == 1024 && unet.imageSide == 768 && !isBase
    }

    // MARK: Names (R13, R14), license (R18)

    static func nameFindings(_ names: [String]) -> [CompatibilityFinding] {
        if mentions(["lcm"], in: names) {
            let defaults = latentConsistencySteps
            return [caveat("R13", "Needs an LCM scheduler, which the runtime lacks; quality not guaranteed",
                           "Defaults: \(defaults.steps) steps, guidance \(defaults.guidance) (estimate).")]
        }
        if mentions(distilledKeywords, in: names) {
            let defaults = distilledSteps
            return [info("R14", "Distilled for few steps",
                         "Defaults: \(defaults.steps) steps, guidance \(defaults.guidance) (estimate).")]
        }
        return []
    }

    static func recommendedDefaults(_ names: [String]) -> (steps: Int, guidance: Float) {
        if mentions(["lcm"], in: names) { return latentConsistencySteps }
        if mentions(distilledKeywords, in: names) { return distilledSteps }
        return defaultSteps
    }

    /// Token match, so "hyperrealistic" is not Hyper-SD; a keyword may carry a version ("dmd2").
    static func mentions(_ keywords: [String], in names: [String]) -> Bool {
        names.contains { name in
            name.lowercased().split { !$0.isLetter && !$0.isNumber }.contains { token in
                keywords.contains { keyword in
                    token.hasPrefix(keyword) && token.dropFirst(keyword.count).allSatisfy(\.isNumber)
                }
            }
        }
    }

    static func licenseFindings(_ listing: HFRepoInfo) -> [CompatibilityFinding] {
        let texts = [listing.licenseID, listing.licenseName].compactMap { $0?.lowercased() }
        let tokens = texts.flatMap { $0.split { !$0.isLetter && !$0.isNumber } }
        let restrictive = texts.isEmpty || texts.contains("other") || tokens.contains("nc")
            || tokens.contains("community") || texts.contains { $0.contains("non-commercial") }
            || texts.contains { $0.contains("noncommercial") }
        guard restrictive else { return [] }
        let named = listing.licenseName ?? listing.licenseID
        return [caveat("R18", "Read the license before use",
                       named.map { "The model card says \($0)." } ?? "The model card names no license.")]
    }

    static func license(of listing: HFRepoInfo) -> (name: String, url: URL) {
        let name = listing.licenseName ?? listing.licenseID ?? "No license on the model card"
        if let link = listing.licenseLink, let url = URL(string: link), url.scheme?.lowercased() == "https" {
            return (name, url)
        }
        let page = URL(string: "https://huggingface.co/\(listing.repoID)") ?? URL(fileURLWithPath: "/")
        return (name, page)
    }

    // MARK: Memory (FIT)

    /// Same 2.2× estimate as ``CoreMLDiffusionEngine``'s gate, but against the device budget: the gate compares it
    /// with the memory free when the model loads, so a folder rated Fits or Tight can still be refused then.
    static func fitFinding(bytes: Int64?, fit: ModelFitLevel?, target: DevicePreset) -> CompatibilityFinding {
        guard let bytes, let fit else {
            return caveat("FIT", "Size unknown", "The listing has no file sizes, so the memory fit can't be checked.")
        }
        let need = String(format: "%.1f GB", HardwareAnalyzer.peakMemoryGB(forWeightsBytes: Int(bytes),
                                                                           kind: .diffusion))
        let budget = String(format: "%.1f GB", target.budgetGB)
        let detail = "Estimated peak: 2.2× the \(gigabytes(bytes)) download (a factor measured on FLUX, not Core ML). "
            + "Budget on \(target.displayName): \(target.budgetText). Loading also needs that much free at the time, "
            + "beyond what the app already holds."
        switch fit {
        case .tooLarge:
            return blocker("FIT", "Won't fit: needs about \(need), \(budget) budget", detail)
        case .marginal, .streamingRequired:
            return caveat("FIT", "Tight: needs about \(need) of a \(budget) budget", detail)
        case .excellent, .good:
            return info("FIT", "Fits: needs about \(need) of a \(budget) budget", detail)
        }
    }

    // MARK: Helpers

    static func unverified(_ detail: String) -> CompatibilityFinding {
        caveat("R1", "Can't verify the Core ML Unet", detail)
    }

    static func sdxl(_ detail: String) -> CompatibilityFinding {
        blocker("R5", "SDXL — TonamoxImageGen's runner does not support it yet",
                detail + " TonamoxImageGen runs StableDiffusionPipeline (SD 1.x/2.x) only.")
    }

    static func describe(_ shape: [Int]?) -> String {
        shape.map { "[\($0.map(String.init).joined(separator: ", "))]" } ?? "missing"
    }

    static func gigabytes(_ bytes: Int64) -> String {
        String(format: "%.2f GB", Double(bytes) / 1e9)
    }

    static func blocker(_ rule: String, _ title: String, _ detail: String) -> CompatibilityFinding {
        CompatibilityFinding(rule: rule, level: .blocker, title: title, detail: detail)
    }

    static func caveat(_ rule: String, _ title: String, _ detail: String) -> CompatibilityFinding {
        CompatibilityFinding(rule: rule, level: .caveat, title: title, detail: detail)
    }

    static func info(_ rule: String, _ title: String, _ detail: String) -> CompatibilityFinding {
        CompatibilityFinding(rule: rule, level: .info, title: title, detail: detail)
    }
}
