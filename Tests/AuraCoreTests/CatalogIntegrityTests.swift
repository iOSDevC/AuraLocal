import XCTest
@testable import AuraCore

/// Every `Model` constant is `registry.model(id:) ?? models.first(where: { $0.format == .gguf })!`.
/// That `??` is silent: while the MLX products were undeclared, `canImport(MLXLLM)` was false,
/// ModelRegistry filtered all 32 MLX entries out, and all 32 MLX constants resolved to the SAME
/// arbitrary GGUF — the first in file order. `Model.qwen3_0_6b`, a 400 MB model, WAS a 4.7 GB
/// Llama 3.1 8B. Nothing failed; the app just quietly sized memory and picked backends off a
/// model the caller never asked for. These tests make that fallback loud.
final class CatalogIntegrityTests: XCTestCase {

    /// The canary. If MLX ever gets un-declared again, this fails instead of lying.
    func testModelConstantsResolveToTheModelTheyName() {
        XCTAssertEqual(Model.qwen3_0_6b.id, "qwen3_0_6b",
            "qwen3_0_6b resolved to '\(Model.qwen3_0_6b.id)' — the silent GGUF fallback is back")
        XCTAssertEqual(Model.qwen3_0_6b.format, .mlx)
        XCTAssertLessThan(Model.qwen3_0_6b.approximateSizeMB, 1_500,
            "a 0.6B model cannot weigh \(Model.qwen3_0_6b.approximateSizeMB) MB")
    }

    /// No two distinct constants may collapse onto one model — that is the fallback's signature.
    func testDistinctConstantsAreDistinctModels() {
        let ids = [Model.qwen3_0_6b.id, Model.qwen3_1_7b.id, Model.qwen3_4b.id, Model.llama3_1_8b_gguf.id]
        XCTAssertEqual(Set(ids).count, ids.count, "constants collapsed onto one model: \(ids)")
    }

    /// The bundled catalog must reach users, not be filtered down to a third of itself.
    func testCatalogIsNotSilentlyHidden() {
        let models = ModelRegistry.shared.models
        let mlx = models.filter { $0.format == .mlx }
        #if canImport(MLXLLM)
        XCTAssertGreaterThan(mlx.count, 0, "MLX is linked, so MLX models must be offered")
        XCTAssertEqual(Model.mlxModels.count, mlx.count)
        #else
        XCTAssertEqual(mlx.count, 0, "no MLX backend linked — unrunnable entries must stay hidden")
        #endif
        XCTAssertGreaterThan(models.filter { $0.format == .gguf }.count, 0)
    }

    /// AppState sizes its memory budget from these two; both were the 4.7 GB fallback.
    func testModelsUsedForMemoryBudgetingAreReal() {
        XCTAssertEqual(Model.qwen3_1_7b.id, "qwen3_1_7b")
        XCTAssertLessThan(Model.qwen3_1_7b.approximateSizeMB, 3_000)
    }
}
