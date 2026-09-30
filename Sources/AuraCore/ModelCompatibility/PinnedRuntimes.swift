import Foundation

// What the runtimes AuraLocal pins can load. Generated from their sources; regenerate when a pin moves.
//
// MLX — mlx-swift-lm 3.31.3 (Package.resolved). The keys of `LLMTypeRegistry.shared` in
//   .build/checkouts/mlx-swift-lm/Libraries/MLXLLM/LLMModelFactory.swift and of `VLMTypeRegistry.shared` in
//   .build/checkouts/mlx-swift-lm/Libraries/MLXVLM/VLMModelFactory.swift (`"<model_type>": create(...)` lines).
//   ModelCompatibilityTests diffs both sets against the checkout when it exists.
//   `mlxRopeModelTypes`: registry entries whose model file calls `initializeRope` with the config's rope
//   scaling (MLXLMCommon/RoPEUtils.swift, which `fatalError`s on unknown rope types).
// llama.cpp — LocalLLMClient 0.5.0 declares `llamaVersion = "b8851"` in its Package.swift. Names are the
//   values of `LLM_ARCH_NAMES` in https://raw.githubusercontent.com/ggml-org/llama.cpp/b8851/src/llama-arch.cpp,
//   minus `clip` (a projector placeholder) and `(unknown)`.

/// Model types and architectures the pinned runtimes accept. See the header comment for provenance.
public enum PinnedRuntimes {

    public static let mlxSwiftLMVersion = "3.31.3"
    public static let llamaCppBuild = "b8851"

    /// `model_type` values `LLMModelFactory` can instantiate (text).
    public static let mlxLLMModelTypes: Set<String> = [
        "acereason", "afmoe", "apertus", "baichuan_m1", "bailing_moe", "bitnet", "cohere", "deepseek_v3",
        "ernie4_5", "exaone4", "falcon_h1", "gemma", "gemma2", "gemma3", "gemma3_text", "gemma3n", "gemma4",
        "gemma4_text", "glm4", "glm4_moe", "glm4_moe_lite", "gpt_oss", "granite", "granitemoehybrid", "internlm2",
        "jamba_3b", "lfm2", "lfm2_moe", "lille-130m", "llama", "mimo", "mimo_v2_flash", "minicpm", "minimax",
        "mistral", "mistral3", "nanochat", "nemotron_h", "olmo2", "olmo3", "olmoe", "openelm", "phi", "phi3",
        "phimoe", "qwen2", "qwen3", "qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_moe", "qwen3_next",
        "smollm3", "starcoder2",
    ]

    /// `model_type` values `VLMModelFactory` can instantiate (image + text).
    public static let mlxVLMModelTypes: Set<String> = [
        "fastvlm", "gemma3", "gemma4", "glm_ocr", "idefics3", "lfm2-vl", "lfm2_vl", "llava_qwen2", "mistral3",
        "paligemma", "pixtral", "qwen2_5_vl", "qwen2_vl", "qwen3_5", "qwen3_5_moe", "qwen3_vl", "smolvlm",
    ]

    /// Model types whose implementation hands `rope_scaling` / `rope_parameters` to `initializeRope`.
    static let mlxRopeModelTypes: Set<String> = [
        "afmoe", "apertus", "bailing_moe", "bitnet", "deepseek_v3", "ernie4_5", "exaone4", "falcon_h1", "gemma3",
        "gemma3_text", "gemma4", "gemma4_text", "glm4", "glm4_moe", "glm4_moe_lite", "granite", "granitemoehybrid",
        "llama", "mimo", "minicpm", "mistral", "mistral3", "olmo2", "olmo3", "olmoe", "qwen3_5", "qwen3_5_text",
        "qwen3_next",
    ]

    /// Rope types `initializeRope` accepts; anything else hits `fatalError("Unsupported RoPE type")`.
    static let mlxRopeTypes: Set<String> = [
        "default", "linear", "proportional", "llama3", "yarn", "deepseek_yarn", "telechat3-yarn", "longrope", "mrope",
    ]

    /// Architectures llama.cpp b8851 knows (`general.architecture`).
    public static let llamaCppArchitectures: Set<String> = [
        "llama", "llama4", "deci", "falcon", "grok", "gpt2", "gptj", "gptneox", "mpt", "baichuan", "starcoder",
        "refact", "bert", "modern-bert", "nomic-bert", "nomic-bert-moe", "neo-bert", "jina-bert-v2", "jina-bert-v3",
        "eurobert", "bloom", "stablelm", "qwen", "qwen2", "qwen2moe", "qwen2vl", "qwen3", "qwen3moe", "qwen3next",
        "qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe", "phi2", "phi3", "phimoe", "plamo", "plamo2", "plamo3",
        "codeshell", "orion", "internlm2", "minicpm", "minicpm3", "gemma", "gemma2", "gemma3", "gemma3n", "gemma4",
        "gemma-embedding", "starcoder2", "mamba", "mamba2", "jamba", "falcon-h1", "xverse", "command-r", "cohere2",
        "dbrx", "olmo", "olmo2", "olmoe", "openelm", "arctic", "deepseek", "deepseek2", "deepseek2-ocr", "chatglm",
        "glm4", "glm4moe", "glm-dsa", "bitnet", "t5", "t5encoder", "jais", "jais2", "nemotron", "nemotron_h",
        "nemotron_h_moe", "exaone", "exaone4", "exaone-moe", "rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7", "granite",
        "granitemoe", "granitehybrid", "chameleon", "wavtokenizer-dec", "plm", "bailingmoe", "bailingmoe2", "dots1",
        "arcee", "afmoe", "ernie4_5", "ernie4_5-moe", "hunyuan-moe", "hunyuan-dense", "smollm3", "gpt-oss", "lfm2",
        "lfm2moe", "dream", "smallthinker", "llada", "llada-moe", "seed_oss", "grovemoe", "apertus", "minimax-m2",
        "cogvlm", "rnd1", "pangu-embedded", "mistral3", "mistral4", "paddleocr", "mimo2", "step35", "llama-embed",
        "maincoder", "kimi-linear",
    ]

    /// b8851 architectures that load but cannot answer a chat prompt through LocalLLMClient, with the reason.
    static let llamaCppNonChatArchitectures: [String: String] = {
        let encoder = "an encoder / embedding model — it produces vectors, not text"
        let diffusion = "a diffusion language model — llama.cpp flags it `llm_arch_is_diffusion` and generates it "
            + "only through its diffusion example, not the autoregressive loop LocalLLMClient drives"
        return [
            "bert": encoder, "modern-bert": encoder, "nomic-bert": encoder, "nomic-bert-moe": encoder,
            "neo-bert": encoder, "jina-bert-v2": encoder, "jina-bert-v3": encoder, "eurobert": encoder,
            "t5encoder": encoder, "gemma-embedding": encoder, "llama-embed": encoder,
            "t5": "an encoder-decoder model — it needs `llama_encode`, which LocalLLMClient never calls",
            "wavtokenizer-dec": "an audio-token decoder used by text-to-speech pipelines, not a chat model",
            "dream": diffusion, "llada": diffusion, "llada-moe": diffusion, "rnd1": diffusion,
        ]
    }()

    /// Architectures added to llama.cpp after b8851: first build that knows them.
    static let laterLlamaCppArchitectures: [String: LlamaCppMilestone] = [
        "qwen4exp": LlamaCppMilestone(build: "b10660", date: "2026-08-27", change: "adds the qwen4exp architecture"),
    ]

    /// qwen35 / qwen35moe GGUFs that carry an MTP (NextN) block fail on b8851; fixed later.
    static let qwen35NextNFix = LlamaCppMilestone(
        build: "b9495", date: "2026-06-03",
        change: "PR #24025 excludes the NextN block from the recurrent-layer pattern")
}

/// A llama.cpp release that changed what loads.
struct LlamaCppMilestone: Sendable, Equatable {
    let build: String
    let date: String
    let change: String
}
