import Foundation
import AuraCore

/// `aura models search|check|devices` — find Hugging Face models that AuraLocal's pinned runtimes can load on a
/// given device. Reports go to stdout, progress to stderr.
extension AuraCLI {

    static func runModels(_ args: [String]) async throws {
        guard let subcommand = args.first else { printModelsUsage(); exit(2) }
        let rest = Array(args.dropFirst())
        switch subcommand {
        case "search": try await runModelsSearch(rest)
        case "check": try await runModelsCheck(rest)
        case "devices": printDevicePresets()
        case "help", "-h", "--help": printModelsUsage()
        default:
            err("Unknown models subcommand: \(subcommand)\n")
            printModelsUsage(); exit(2)
        }
    }

    static func printModelsUsage() {
        print("""
        aura models — Hugging Face models that AuraLocal can really run (MLX \(PinnedRuntimes.mlxSwiftLMVersion), \
        llama.cpp \(PinnedRuntimes.llamaCppBuild))

          aura models search "<query>" [--format mlx|gguf] [--device <preset>] [--limit N]
          aura models check <owner/repo | URL> [--device <preset>] [--json] [--entry]
          aura models devices                     List device presets and where their budgets come from

        --device defaults to this-device. --entry prints only the models.json entry.
        """)
    }

    // MARK: - devices

    static func printDevicePresets() {
        for preset in DevicePreset.all() {
            print("\(preset.id.padding(toLength: 15, withPad: " ", startingAt: 0)) \(preset.displayName) — \(preset.budgetText)")
            print("    \(preset.source)")
        }
    }

    // MARK: - check

    static func runModelsCheck(_ args: [String]) async throws {
        let usage = "usage: aura models check <owner/repo> [--device <preset>] [--json] [--entry]"
        let parsed = try CLIArguments(args, options: ["--device"], flags: ["--json", "--entry"])
        guard parsed.positionals.count == 1, let repo = parsed.positionals.first else { throw CLIUsageError(usage) }
        let device = try devicePreset(parsed.value("--device"))

        err("Checking \(repo) for \(device.displayName)…\n")
        let report = await ModelCompatibilityChecker().check(repo, on: device)

        if parsed.has("--entry") {
            guard let entry = report.suggestedEntry else {
                throw CLIUsageError("No catalog entry: \(report.status.label) — \(report.headline)")
            }
            print(entry.jsonText())
        } else if parsed.has("--json") {
            print(try jsonText(report))
        } else {
            printReport(report)
        }
    }

    static func printReport(_ report: CompatibilityReport) {
        print("\(report.repoID) — \(report.weightFormat.label) · \(report.status.label.uppercased())")
        print("Device: \(report.target.displayName) · budget \(report.target.budgetText)")
        let facts = report.overview
        let described = [
            facts.family,
            facts.layers.map { "\($0) layers" },
            facts.trainedContext.map { "context \($0)" },
            facts.quantization,
            report.modelCategory.map { "loads as \($0.rawValue)" },
        ].compactMap { $0 }
        if !described.isEmpty { print("Model: \(described.joined(separator: " · "))") }
        if let bytes = report.weightsBytes {
            let fit = report.weightsFit.map { " · \($0.summary)" } ?? ""
            print("Weights: \(CompatibilityReport.gigabytesText(bytes))\(fit)")
        }
        print("")
        for finding in report.findings {
            print("  \(marker(finding.level)) \(finding.title)")
            print("      \(finding.detail)")
        }
        if !report.quantFits.isEmpty {
            print("\nQuants (\(report.quantFits.count)):")
            for quant in report.quantFits {
                print("  " + quantLine(quant))
            }
        }
        if let entry = report.suggestedEntry {
            print("\nmodels.json entry:\n\(entry.jsonText())")
        }
    }

    static func quantLine(_ quant: QuantFit) -> String {
        let name = (quant.option.label ?? quant.option.firstPath).padding(toLength: 10, withPad: " ", startingAt: 0)
        let parts = quant.option.isSplit ? "  \(quant.option.paths.count) parts, not loadable" : ""
        return "\(name) \(CompatibilityReport.gigabytesText(quant.option.totalBytes).leftPadded(9))  \(quant.memory.summary)\(parts)"
    }

    static func marker(_ level: FindingSeverity) -> String {
        switch level {
        case .blocker: "BLOCKER"
        case .caveat: "caveat "
        case .info: "info   "
        }
    }

    static func jsonText(_ report: CompatibilityReport) throws -> String {
        let fit: (FitEstimate?) -> Any = { estimate in
            guard let estimate else { return NSNull() }
            return ["rating": estimate.rating.label, "requiredGB": estimate.requiredGB, "budgetGB": estimate.budgetGB]
        }
        let object: [String: Any] = [
            "repoID": report.repoID,
            "device": report.target.id,
            "format": report.weightFormat.rawValue,
            "verdict": report.status.rawValue,
            "category": report.modelCategory?.rawValue ?? NSNull(),
            "headline": report.headline,
            "weightsBytes": report.weightsBytes ?? NSNull(),
            "weightsFit": fit(report.weightsFit),
            "findings": report.findings.map {
                ["rule": $0.rule, "severity": $0.level.label.lowercased(), "title": $0.title, "detail": $0.detail]
            },
            "quants": report.quantFits.map {
                ["path": $0.option.firstPath, "label": $0.option.label ?? NSNull(), "bytes": $0.option.totalBytes,
                 "parts": $0.option.paths.count, "fit": fit($0.memory)] as [String: Any]
            },
            "entry": report.suggestedEntry.map { (try? JSONSerialization.jsonObject(with: Data($0.jsonText().utf8))) ?? NSNull() }
                ?? NSNull(),
        ]
        let data = try JSONSerialization.data(withJSONObject: object,
                                              options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        return String(decoding: data, as: UTF8.self)
    }

    // MARK: - search

    static func runModelsSearch(_ args: [String]) async throws {
        let usage = "usage: aura models search \"<query>\" [--format mlx|gguf] [--device <preset>] [--limit N]"
        let parsed = try CLIArguments(args, options: ["--format", "--device", "--limit"])
        guard !parsed.positionals.isEmpty else { throw CLIUsageError(usage) }
        let query = parsed.positionals.joined(separator: " ")
        let format = parsed.value("--format")?.lowercased()
        guard format == nil || format == "mlx" || format == "gguf" else {
            throw CLIUsageError("--format expects mlx or gguf, got \(format ?? "").")
        }
        let limit = try parsed.int("--limit") ?? 15
        guard (1...100).contains(limit) else { throw CLIUsageError("--limit expects 1…100.") }
        let device = try devicePreset(parsed.value("--device"))

        let hits = try await HuggingFaceSearch.search(query, limit: limit, sort: .downloads, tag: format)
        guard !hits.isEmpty else { err("No models match \"\(query)\".\n"); return }
        err("Checking \(hits.count) repos for \(device.displayName)…\n")
        let reports = await checkAll(hits.map(\.id), on: device)

        print("VERDICT            FORMAT  FIT                            REPO")
        for report in reports {
            print(searchRow(report))
            print("                   \(report.headline)")
        }
    }

    static func searchRow(_ report: CompatibilityReport) -> String {
        let verdict = report.status.label.padding(toLength: 18, withPad: " ", startingAt: 0)
        let format = report.weightFormat.label.prefix(6).padding(toLength: 7, withPad: " ", startingAt: 0)
        let fit = (report.bestFit?.summary ?? "—").padding(toLength: 30, withPad: " ", startingAt: 0)
        return "\(verdict) \(format) \(fit) \(report.repoID)"
    }

    /// At most four repos in flight; results keep the search order.
    static func checkAll(_ repos: [String], on device: DevicePreset) async -> [CompatibilityReport] {
        let checker = ModelCompatibilityChecker()
        var reports = [CompatibilityReport?](repeating: nil, count: repos.count)
        await withTaskGroup(of: (Int, CompatibilityReport).self) { group in
            var next = 0
            func enqueue() {
                guard next < repos.count else { return }
                let index = next
                group.addTask { (index, await checker.check(repos[index], on: device)) }
                next += 1
            }
            (0..<4).forEach { _ in enqueue() }
            while let (index, report) = await group.next() {
                reports[index] = report
                enqueue()
            }
        }
        return reports.compactMap { $0 }
    }

    // MARK: - Helpers

    static func devicePreset(_ id: String?) throws -> DevicePreset {
        guard let id else { return .thisDevice() }
        guard let preset = DevicePreset.named(id) else {
            let known = DevicePreset.all().map(\.id).joined(separator: ", ")
            throw CLIUsageError("Unknown device \(id). Presets: \(known).")
        }
        return preset
    }
}

private extension String {
    func leftPadded(_ width: Int) -> String {
        count >= width ? self : String(repeating: " ", count: width - count) + self
    }
}
