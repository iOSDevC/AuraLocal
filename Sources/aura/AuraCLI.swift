import Foundation
import AuraCore
import AuraImageGen

/// `aura` — a small headless CLI that drives AuraLocal's hybrid + native-tool
/// features, as an integration reference and CI smoke-test harness. macOS.
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
            case "imagegen":         try await runImageGen(rest)
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
                for model in status.models { print("    - \(model.name)") }
            } else if status.reachable {
                print("◐ \(name) — reachable, no models — \(url)")
            } else {
                print("○ \(name) — not running — \(url)")
            }
        }
    }

    // MARK: - tools

    /// List on-device native tools (Vision OCR, NaturalLanguage embeddings) + availability.
    static func runTools() async {
        for tool in await SystemToolRegistry.discover() {
            print("\(tool.isAvailable ? "●" : "○") \(tool.displayName)  [\(tool.id)]")
            print("    \(tool.isAvailable ? tool.summary : (tool.availability.reason ?? "unavailable"))")
        }
    }

    // MARK: - ask (hybrid remote → GitHub Models)

    /// Ask a bigger model via GitHub Models. Token from AURA_GITHUB_TOKEN /
    /// GITHUB_TOKEN, or the Keychain (cloud.github-models). Answer → stdout,
    /// receipt → stderr (so stdout stays clean for piping).
    static func runAsk(_ args: [String]) async throws {
        var prompt: String?
        var model = "openai/gpt-4o"
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--model": i += 1; if i < args.count { model = args[i] }
            case "--remote": break   // remote is the only mode in v1
            default: if prompt == nil { prompt = args[i] }
            }
            i += 1
        }
        guard let prompt, !prompt.isEmpty else {
            err("usage: aura ask \"<prompt>\" [--model <id>]\n"); exit(2)
        }

        let env = ProcessInfo.processInfo.environment
        let token = env["AURA_GITHUB_TOKEN"]
            ?? env["GITHUB_TOKEN"]
            ?? KeychainStore.read(for: "cloud.github-models")
        guard let token, !token.isEmpty else {
            err("No GitHub token. Set AURA_GITHUB_TOKEN (models:read PAT) or save one to the Keychain (cloud.github-models).\n")
            exit(1)
        }

        let target = RemoteTarget(
            provider: OpenAICompatibleProvider.gitHubModels(apiKey: token),
            modelID: model, contextLength: 128_000, origin: .cloud)
        let result = try await HybridEscalator().escalate(
            to: target, systemPrompt: nil, context: "", question: prompt,
            maxTokens: 512, redactPII: true)

        print(result.answer)
        var receipt = "— via \(result.providerName) · \(model)"
        if let usage = result.usage { receipt += " · \(usage.inputTokens) in / \(usage.outputTokens) out" }
        if result.fromCache { receipt += " · cached" }
        err(receipt + "\n")
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
        var model = "schnell", steps: Int? = nil, seed: UInt64? = nil, quantize: Int? = 4
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
            prompt: prompt, model: model, steps: steps, seed: seed,
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
          aura tools                          List on-device tools (Vision OCR, embeddings)
          aura ask "<prompt>" [--model <id>]  Ask GitHub Models (hybrid remote); default openai/gpt-4o
          aura ocr <image>                    Extract text from an image via native Vision OCR
          aura imagegen "<prompt>" [--lora <p>]  Generate an image via mflux/FLUX (macOS; needs mflux)

        ENV:
          AURA_GITHUB_TOKEN   GitHub fine-grained PAT with models:read (used by `ask`)
        """)
    }

    static func err(_ message: String) {
        FileHandle.standardError.write(Data(message.utf8))
    }
}
