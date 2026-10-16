# Third-party notices

AuraLocal is licensed under AGPL-3.0-or-later (see [LICENSE](LICENSE)) or a commercial license (see
[License](README.md#license)). It depends on the Swift packages below, resolved by Swift Package Manager from
[`Package.resolved`](Package.resolved). Each keeps its own license; all are compatible with AGPL-3.0, and their
permissive terms also allow proprietary use under the commercial license. Binaries built from AuraLocal (the `aura` CLI, the example apps) include code from
these packages, so their license texts and NOTICE files must ship with those binaries.

| Package | Version | License |
|---|---|---|
| [LocalLLMClient](https://github.com/tattn/LocalLLMClient) (bundles [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT) | revision `edc39ef` (0.5.0) | MIT |
| [mlx-swift](https://github.com/ml-explore/mlx-swift) (bundles MLX and mlx-c, MIT; {fmt}, MIT; nlohmann/json, MIT; metal-cpp, Apache-2.0) | 0.31.3 | MIT |
| [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | 3.31.3 | MIT |
| [ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion) | 1.1.1 | MIT |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | 1.3.3 | Apache-2.0 |
| [swift-huggingface](https://github.com/huggingface/swift-huggingface) | 0.9.0 | Apache-2.0 |
| [swift-jinja](https://github.com/huggingface/swift-jinja) | 2.4.1 | Apache-2.0 |
| [EventSource](https://github.com/mattt/EventSource) | 1.4.1 | MIT |
| [yyjson](https://github.com/ibireme/yyjson) | 0.12.0 | MIT |
| [swift-argument-parser](https://github.com/apple/swift-argument-parser) | 1.8.2 | Apache-2.0 with Runtime Library Exception |
| [swift-asn1](https://github.com/apple/swift-asn1) | 1.7.1 | Apache-2.0 (has a NOTICE file) |
| [swift-atomics](https://github.com/apple/swift-atomics) | 1.3.1 | Apache-2.0 |
| [swift-collections](https://github.com/apple/swift-collections) | 1.4.1 | Apache-2.0 |
| [swift-crypto](https://github.com/apple/swift-crypto) | 4.5.1 | Apache-2.0 (has a NOTICE file) |
| [swift-nio](https://github.com/apple/swift-nio) | 2.101.3 | Apache-2.0 (has a NOTICE file) |
| [swift-numerics](https://github.com/apple/swift-numerics) | 1.1.1 | Apache-2.0 |
| [swift-syntax](https://github.com/swiftlang/swift-syntax) | 600.0.1 | Apache-2.0 |
| [swift-system](https://github.com/apple/swift-system) | 1.7.4 | Apache-2.0 |

Versions are the ones pinned in `Package.resolved` when this file was written; check that file for the current
pins. The full license and NOTICE texts are in each package's repository at the pinned version.

Models (GGUF, MLX, Core ML) are downloaded at run time from Hugging Face and are not part of AuraLocal. Each
carries its own license, shown on its model card.
