import Foundation
import AuraCore
import AuraImageGen

/// `aura` — a small headless CLI that drives AuraLocal's hybrid, native-tool and
/// on-device ML features, as an integration reference and CI smoke-test harness. macOS.
@main
@MainActor
struct AuraCLI {
    static func main() async {
        let args = Array(CommandLine.arguments.dropFirst())
        guard let command = args.first else { printUsage(); exit(2) }
        let rest = Array(args.dropFirst())
        do {
            switch command {
            case "providers":        await runProviders()
            case "tools":            await runTools()
            case "ask":              try await runAsk(rest)
            case "ocr":              try runOCR(rest)
            case "ml":               try await runML(rest)
            case "imagegen":         try await runImageGen(rest)
            case "models":           try await runModels(rest)
            case "help", "-h", "--help": printUsage()
            default:
                err("Unknown command: \(command)\n")
                printUsage(); exit(2)
            }
        } catch {
            err("Error: \(error.localizedDescription)\n")
            exit(1)
        }
    }

    // MARK: - providers

    /// Detect local inference providers (Ollama / llama-server) and their models.
    static func runProviders() async {
        print("Network: \(NetworkMonitor.shared.isOnline ? "online" : "offline")")
        let statuses = await LocalProviderDetector.detectAll()
        for status in statuses {
            let name = status.kind == .ollama ? "Ollama" : "llama-server"
            let url = status.baseURL.absoluteString
            if status.isAvailable {
                print("● \(name) — \(url) (\(status.models.count) models)")
                for model in status.models {
                    let note = model.runsLocally ? "" : " (forwarded to \(model.remoteHost ?? "a remote host"); skipped by ask)"
                    print("    - \(model.name)\(note)")
                }
            } else if status.reachable {
                print("◐ \(name) — reachable, no models — \(url)")
            } else {
                print("○ \(name) — not running — \(url)")
            }
        }
    }

    // MARK: - tools

    /// List the on-device tools by category, each with its availability on this Mac.
    static func runTools() async {
        let tools = await SystemToolRegistry.discover()
        for category in SystemToolCategory.allCases {
            let members = tools.filter { $0.category == category }
            print(category.displayName)
            if category == .customModel && members.isEmpty {
                print("  Created per model file (CoreMLModelTool, TextClassifierTool) — see `aura ml coreml-describe`.")
            }
            for tool in members {
                print("  \(tool.isAvailable ? "●" : "○") \(tool.displayName)  [\(tool.id)]")
                print("      \(tool.isAvailable ? tool.summary : (tool.availability.reason ?? "unavailable"))")
            }
        }
    }

    // MARK: - ask (local-first remote)

    static let askValueFlags: Set<String> = ["--provider", "--model", "--base-url", "--max-tokens"]

    static let askUsage = """
        usage: aura ask "<prompt>" [--provider auto|local|openai|anthropic] [--model <id>]
                                   [--base-url <url>] [--max-tokens N]

        """

    /// Ask a bigger model. The default (`auto`) uses a running llama-server, else
    /// Ollama, and never a cloud API; see `AskTargetResolver`. Answer → stdout,
    /// receipt → stderr (so stdout stays clean for piping).
    static func runAsk(_ args: [String]) async throws {
        var prompt: String?
        var choice = AskTargetResolver.Choice.auto
        var model: String?
        var baseURL: String?
        var maxTokens = 512
        var i = 0
        while i < args.count {
            let arg = args[i]
            if arg == "--help" || arg == "-h" {
                print(askUsage, terminator: "")
                exit(0)
            }
            if arg.hasPrefix("--") {
                guard askValueFlags.contains(arg) else { askUsageError("unknown flag \(arg)") }
                i += 1
                guard i < args.count else { askUsageError("\(arg) needs a value") }
                let value = args[i]
                switch arg {
                case "--provider":
                    choice = parseChoice(value)
                case "--model":
                    model = value
                case "--base-url":
                    baseURL = value
                case "--max-tokens":
                    maxTokens = parseMaxTokens(value)
                default:
                    askUsageError("unknown flag \(arg)")
                }
            } else if prompt == nil {
                prompt = arg
            } else {
                askUsageError("unexpected argument \"\(arg)\" (quote the prompt)")
            }
            i += 1
        }
        guard let prompt, !prompt.isEmpty else { askUsageError("missing prompt") }

        let probeLocal = baseURL == nil && (choice == .auto || choice == .local)
        let localProviders = probeLocal ? await LocalProviderDetector.detectAll() : []
        let target: RemoteTarget
        do {
            target = try AskTargetResolver.resolve(
                choice, model: model, baseURL: baseURL,
                environment: ProcessInfo.processInfo.environment,
                readKey: KeychainStore.read(for:),
                localProviders: localProviders)
        } catch {
            err((error.errorDescription ?? "\(error)") + "\n")
            exit(error.isUsageError ? 2 : 1)
        }

        let result = try await HybridEscalator().escalate(
            to: target, systemPrompt: nil, context: "", question: prompt,
            maxTokens: maxTokens, redactPII: !target.isLocalNetwork)

        print(result.answer)
        var receipt = "— via \(result.providerName) · \(target.modelID)"
        if let usage = result.usage { receipt += " · \(usage.inputTokens) in / \(usage.outputTokens) out" }
        if result.fromCache { receipt += " · cached" }
        err(receipt + "\n")
    }

    static func parseChoice(_ value: String) -> AskTargetResolver.Choice {
        guard let choice = AskTargetResolver.Choice(rawValue: value.lowercased()) else {
            askUsageError("unknown provider \"\(value)\"")
        }
        return choice
    }

    static func parseMaxTokens(_ value: String) -> Int {
        guard let tokens = Int(value), tokens > 0 else { askUsageError("--max-tokens needs a positive integer") }
        return tokens
    }

    static func askUsageError(_ message: String) -> Never {
        err("aura ask: \(message)\n" + askUsage)
        exit(2)
    }

    // MARK: - ocr (native Vision)

    /// Extract text from an image via the Vision framework — no model download.
    static func runOCR(_ args: [String]) throws {
        guard let path = args.first else { err("usage: aura ocr <image>\n"); exit(2) }
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let result = try VisionOCRTool().recognizeText(inImageData: data)
        if result.isEmpty {
            err("(no text found)\n")
        } else {
            print(result.text)
            err("— \(result.lineCount) lines, avg confidence \(String(format: "%.2f", result.averageConfidence))\n")
        }
    }

    // MARK: - imagegen

    /// `aura imagegen "<prompt>" [--lora <path> [--lora-scale <s>]]... [--model schnell|dev]
    ///  [--seed N] [--steps N] [--quantize 3|4|6|8] [--low-ram] [--out <path>]`
    /// Drives mflux (FLUX on MLX) — macOS-only, requires `uv tool install mflux`.
    static func runImageGen(_ args: [String]) async throws {
        guard let prompt = args.first, !prompt.hasPrefix("--") else {
            err("usage: aura imagegen \"<prompt>\" [--lora <path.safetensors> [--lora-scale <s>]] "
                + "[--model schnell|dev] [--seed N] [--steps N] [--quantize 4] [--low-ram] [--out <dir>]\n")
            exit(2)
        }
        var model = "schnell", baseModel: String? = nil
        var steps: Int? = nil, seed: UInt64? = nil, quantize: Int? = 4
        var lowRAM = false, outPath: String? = nil
        var loras: [LoRA] = []

        var i = 1
        while i < args.count {
            switch args[i] {
            case "--lora":
                i += 1; guard i < args.count else { break }
                loras.append(LoRA(url: URL(fileURLWithPath: args[i])))
            case "--lora-scale":
                i += 1
                if i < args.count, let s = Float(args[i]), var last = loras.last {
                    last.scale = s; loras[loras.count - 1] = last
                }
            case "--model":    i += 1; if i < args.count { model = args[i] }
            case "--base-model": i += 1; if i < args.count { baseModel = args[i] }
            case "--seed":     i += 1; if i < args.count { seed = UInt64(args[i]) }
            case "--steps":    i += 1; if i < args.count { steps = Int(args[i]) }
            case "--quantize": i += 1; if i < args.count { quantize = Int(args[i]) }
            case "--low-ram":  lowRAM = true
            case "--out":      i += 1; if i < args.count { outPath = args[i] }
            default:           err("Unknown flag: \(args[i])\n")
            }
            i += 1
        }

        let engine = MFluxEngine()
        guard engine.isAvailable else {
            throw ImageGenError.engineNotFound(installHint: MFluxEngine.installHint)
        }

        let request = ImageGenRequest(
            prompt: prompt, model: model, baseModel: baseModel, steps: steps, seed: seed,
            quantize: quantize, lowRAM: lowRAM, loras: loras)
        err("Generating (\(model)\(loras.isEmpty ? "" : ", \(loras.count) LoRA")) — this can take a while…\n")

        let outDir = outPath.map { URL(fileURLWithPath: $0, isDirectory: true) }
        let result = try await engine.generate(request, outputDirectory: outDir)
        print(result.fileURL.path)
    }

    // MARK: - Helpers

    static func printUsage() {
        print("""
        aura — AuraLocal integration CLI

        USAGE:
          aura providers                      Detect local providers (Ollama / llama-server)
          aura tools                          List on-device ML tools by category, with availability
          aura ask "<prompt>" [--provider auto|local|openai|anthropic] [--model <id>]
                   [--base-url <url>] [--max-tokens N]
                                              Ask a bigger model: a running llama-server, else Ollama
                                                (auto, the default); a cloud API only when named
                                                (default models \(RemoteTarget.defaultOpenAIModel),
                                                \(RemoteTarget.defaultAnthropicModel))
          aura ocr <image>                    Extract text from an image via native Vision OCR
          aura ml <subcommand> …              Run an on-device ML tool (`aura ml help` lists them):
                                                classify-image, barcodes, faces, ocr-lines, language,
                                                entities, sentiment, similarity, sounds, coreml-describe,
                                                coreml-predict, train-text, classify-text
          aura imagegen "<prompt>" [--lora <p>]  Generate an image via mflux/FLUX (macOS; needs mflux)
          aura models search "<query>" [--format mlx|gguf] [--device <preset>] [--limit N]
                                              Search Hugging Face with a runs-here verdict per repo
          aura models check <repo> [--device <preset>] [--json] [--entry]
                                              Why a repo does or doesn't run; its models.json entry
          aura models devices                 Device presets (this device, iPhone classes, Macs)

        ENV (used by `ask`):
          OPENAI_API_KEY      --provider openai; else Keychain account cloud.openai
          ANTHROPIC_API_KEY   --provider anthropic; else Keychain account cloud.anthropic
                              (Keychain service dev.auralocal.remote)
          AURA_API_KEY        --base-url, optional, env only (sent as a Bearer token)
        """)
    }

    static func err(_ message: String) {
        fflush(stdout)   // keep stdout and stderr in order when both go to one pipe
        FileHandle.standardError.write(Data(message.utf8))
    }
}
