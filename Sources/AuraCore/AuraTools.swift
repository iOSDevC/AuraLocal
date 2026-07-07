// Re-export LocalLLMClient's tool-calling surface so host apps can define function-calling tools using only
// `import AuraCore`. A tool is a type conforming to `LLMTool` — a `name`, a `description`, an `Arguments`
// type (`Decodable & ToolSchemaGeneratable`, i.e. it declares an `argumentsSchema`), and an async
// `call(arguments:) -> ToolOutput`. Pass tools to `ModelManager.load(_:tools:)` (or `AuraLocal(model:tools:)`)
// and the GGUF backend runs the model with automatic tool-calling: the model decides when to invoke a tool,
// `call(arguments:)` executes it, and the result is fed back until the model produces its final answer.
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
