// Re-export LocalLLMClient's tool-calling surface so host apps can define function-calling tools using only
// `import AuraCore`. A tool is a type conforming to `LLMTool` — a `name`, a `description`, an `Arguments`
// type (`Decodable & ToolSchemaGeneratable`, i.e. it declares an `argumentsSchema`), and an async
// `call(arguments:) -> ToolOutput`. Pass tools to `ModelManager.load(_:tools:)` (or an `AuraProfile`) with a GGUF
// model, and the llama.cpp backend runs it with automatic tool-calling: the model decides when to invoke a tool,
// `call(arguments:)` executes it, and the result is fed back for up to `ToolLoopOptions.maxToolRounds` rounds per
// turn (default 1; set it on `AuraLocal.toolLoop` or `AuraProfile.toolLoop`).
//
// Example (no macro needed):
//
//     struct TapTool: LLMTool {
//         let name = "tap"
//         let description = "Tap the screen at a pixel coordinate."
//         struct Arguments: Decodable, ToolSchemaGeneratable {
//             let x: Int
//             let y: Int
//             static var argumentsSchema: LLMToolArgumentsSchema {
//                 ["x": .integer(description: "X pixel"), "y": .integer(description: "Y pixel")]
//             }
//         }
//         func call(arguments: Arguments) async throws -> ToolOutput {
//             // …perform the tap…
//             ToolOutput(data: ["ok": true])
//         }
//     }
@_exported import LocalLLMClientCore
