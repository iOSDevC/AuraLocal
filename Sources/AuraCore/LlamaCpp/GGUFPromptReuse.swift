import Foundation

// MARK: - GGUFPromptReuse

/// Whether a turn can keep the KV cache the turn before left: its prompt must start with exactly the text the cache
/// holds, byte for byte, and go on with a special token.
///
/// llama.cpp's tokenizer splits the text on special tokens before anything else, so text that starts with one
/// tokenizes the same on its own as at the end of the whole prompt. The cache holds the tokens the model sampled for
/// its reply, which can differ from a fresh tokenization of the same text: a reused turn continues from the model's
/// own generation, which a full prefill would re-tokenize.
struct GGUFPromptReuse: Sendable, Equatable {
    /// Texts of the vocabulary's special tokens (``GGUFChatMetadata/controlTokenTexts``); empty never reuses.
    let boundaries: [String]

    init(boundaries: [String]) {
        self.boundaries = boundaries.filter { !$0.isEmpty }
    }

    /// The part of `prompt` still to decode when the cache holds `resident`; `nil` when the cache cannot be kept.
    func suffix(of prompt: String, after resident: String) -> String? {
        let held = resident.utf8
        let full = prompt.utf8
        guard !held.isEmpty, full.count > held.count, full.starts(with: held) else { return nil }
        let rest = full.dropFirst(held.count)
        guard boundaries.contains(where: { rest.starts(with: $0.utf8) }) else { return nil }
        return String(decoding: rest, as: UTF8.self)
    }
}
