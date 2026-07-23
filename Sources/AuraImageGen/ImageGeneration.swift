import Foundation
import AuraCore   // PlatformImage

// MARK: - Public request / result types

/// A LoRA adapter to fold into generation: a local `.safetensors` file plus its weight.
public struct LoRA: Sendable, Equatable {
    public var url: URL
    public var scale: Float
    public init(url: URL, scale: Float = 1.0) {
        self.url = url
        self.scale = scale
    }
}

/// A text-to-image request. Maps directly onto mflux's CLI surface.
public struct ImageGenRequest: Sendable {
    public var prompt: String
    /// mflux `--model`: "schnell" (few-step, distilled) or "dev".
    public var model: String
    /// mflux `--steps`; `nil` uses the model's default (schnell needs very few).
    public var steps: Int?
    public var seed: UInt64?
    public var width: Int
    public var height: Int
    /// mflux `--quantize` (3/4/6/8); `nil` runs full precision.
    public var quantize: Int?
    /// mflux `--low-ram`: trade speed for a smaller peak working set. Relevant on tight Macs.
    public var lowRAM: Bool
    /// mflux `--lora-paths` / `--lora-scales`. Empty = no LoRA.
    public var loras: [LoRA]

    public init(
        prompt: String,
        model: String = "schnell",
        steps: Int? = nil,
        seed: UInt64? = nil,
        width: Int = 1024,
        height: Int = 1024,
        quantize: Int? = 4,
        lowRAM: Bool = false,
        loras: [LoRA] = []
    ) {
        self.prompt = prompt
        self.model = model
        self.steps = steps
        self.seed = seed
        self.width = width
        self.height = height
        self.quantize = quantize
        self.lowRAM = lowRAM
        self.loras = loras
    }
}

/// The generated image plus the PNG file mflux wrote.
public struct ImageGenResult: Sendable {
    public let image: PlatformImage
    public let fileURL: URL
}

// MARK: - Errors

public enum ImageGenError: LocalizedError {
    /// Image generation is macOS-only; this platform can't run the mflux engine.
    case unsupportedPlatform
    /// `mflux-generate` was not found. `installHint` is the exact command to install it.
    case engineNotFound(installHint: String)
    /// A supplied LoRA `.safetensors` path does not exist.
    case loraNotFound(URL)
    /// mflux ran but exited non-zero; carries its stderr tail.
    case generationFailed(String)
    /// mflux exited 0 but no output image was produced.
    case outputMissing

    public var errorDescription: String? {
        switch self {
        case .unsupportedPlatform:
            return "Image generation runs on macOS only (mflux needs Apple Silicon + Python)."
        case .engineNotFound(let hint):
            return "mflux is not installed. Install it with: \(hint)"
        case .loraNotFound(let url):
            return "LoRA file not found: \(url.path)"
        case .generationFailed(let detail):
            return "Image generation failed: \(detail)"
        case .outputMissing:
            return "mflux reported success but wrote no image."
        }
    }
}

// MARK: - MFluxEngine

/// Drives Filip Strand's **mflux** (FLUX on Apple MLX) as a subprocess to generate images and
/// fold in `lora.safetensors` adapters.
///
/// **macOS-only, and non-sandboxed-only.** mflux is a Python CLI; a Swift package cannot embed a
/// Python runtime, so this locates and execs an *already-installed* `mflux-generate`. The macOS
/// App Sandbox forbids exec-ing external binaries, so this serves the `aura` CLI and non-sandboxed
/// apps — not App-Store-sandboxed apps. An in-process, sandbox-safe engine (mzbac/flux.swift) is a
/// separate, dependency-gated path and is intentionally not wired here.
///
/// FLUX is far too large for iPhone (quantized schnell weights alone are ~9 GB), so there is no iOS
/// path — `isAvailable` is `false` off macOS and `generate` throws `.unsupportedPlatform`.
public struct MFluxEngine: Sendable {

    /// An explicit path to `mflux-generate`, if the caller knows it. Otherwise discovery is used.
    public var executableOverride: URL?

    public init(executableOverride: URL? = nil) {
        self.executableOverride = executableOverride
    }

    /// Whether an mflux engine can run on this machine right now (macOS + binary discoverable).
    public var isAvailable: Bool {
        #if os(macOS)
        return (try? resolveExecutable()) != nil
        #else
        return false
        #endif
    }

    /// The exact command a user should run to install mflux.
    public static let installHint = "uv tool install mflux"

    /// Generate an image. `outputDirectory` defaults to a unique temp dir; the PNG is named by seed/uuid.
    public func generate(
        _ request: ImageGenRequest,
        outputDirectory: URL? = nil
    ) async throws -> ImageGenResult {
        #if os(macOS)
        let exe = try resolveExecutable()

        for lora in request.loras where !FileManager.default.fileExists(atPath: lora.url.path) {
            throw ImageGenError.loraNotFound(lora.url)
        }

        let outDir = outputDirectory ?? FileManager.default.temporaryDirectory
            .appendingPathComponent("auraimagegen-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let outputURL = outDir.appendingPathComponent("image.png")

        let args = Self.buildArguments(for: request, output: outputURL)
        try await Self.run(executable: exe, arguments: args)

        guard FileManager.default.fileExists(atPath: outputURL.path),
              let image = PlatformImage(contentsOfFile: outputURL.path) else {
            throw ImageGenError.outputMissing
        }
        return ImageGenResult(image: image, fileURL: outputURL)
        #else
        throw ImageGenError.unsupportedPlatform
        #endif
    }

    // MARK: - Argument construction (pure, unit-tested without running mflux)

    /// Build the `mflux-generate` argument vector. Verified against mflux's CLI docs:
    /// `--model --prompt --steps --seed --width --height --quantize --low-ram --output
    /// --lora-paths --lora-scales`.
    static func buildArguments(for r: ImageGenRequest, output: URL) -> [String] {
        var a: [String] = ["--model", r.model, "--prompt", r.prompt]
        if let steps = r.steps { a += ["--steps", String(steps)] }
        if let seed = r.seed { a += ["--seed", String(seed)] }
        a += ["--width", String(r.width), "--height", String(r.height)]
        if let q = r.quantize { a += ["--quantize", String(q)] }
        if r.lowRAM { a += ["--low-ram"] }
        if !r.loras.isEmpty {
            a += ["--lora-paths"] + r.loras.map(\.url.path)
            a += ["--lora-scales"] + r.loras.map { String($0.scale) }
        }
        a += ["--output", output.path]
        return a
    }

    // MARK: - Discovery

    #if os(macOS)
    /// Locate `mflux-generate`: explicit override → `~/.local/bin` (uv's default) → `PATH`.
    func resolveExecutable() throws -> URL {
        if let override = executableOverride {
            if FileManager.default.isExecutableFile(atPath: override.path) { return override }
            throw ImageGenError.engineNotFound(installHint: Self.installHint)
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        let uvBin = home.appendingPathComponent(".local/bin/mflux-generate")
        if FileManager.default.isExecutableFile(atPath: uvBin.path) { return uvBin }

        if let onPath = Self.searchPATH("mflux-generate") { return onPath }
        throw ImageGenError.engineNotFound(installHint: Self.installHint)
    }

    /// Resolve a command name against the `PATH` environment variable.
    static func searchPATH(_ name: String) -> URL? {
        guard let path = ProcessInfo.processInfo.environment["PATH"] else { return nil }
        for dir in path.split(separator: ":") {
            let candidate = URL(fileURLWithPath: String(dir)).appendingPathComponent(name)
            if FileManager.default.isExecutableFile(atPath: candidate.path) { return candidate }
        }
        return nil
    }

    /// Run mflux, streaming nothing but capturing stderr for diagnostics. Throws on non-zero exit.
    static func run(executable: URL, arguments: [String]) async throws {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        let errPipe = Pipe()
        process.standardError = errPipe
        process.standardOutput = FileHandle.nullDevice

        try process.run()
        // Read stderr off the run so a large log can't deadlock the pipe.
        let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let tail = String(decoding: errData.suffix(2000), as: UTF8.self)
            throw ImageGenError.generationFailed(tail.isEmpty ? "exit \(process.terminationStatus)" : tail)
        }
    }
    #endif
}
