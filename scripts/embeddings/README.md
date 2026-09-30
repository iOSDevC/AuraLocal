# Embedding model conversion

`convert_e5_coreml.py` builds the text-embedding bundle that `CoreMLTextEmbeddingTool`,
`CoreMLEmbeddingProvider` and `AutoEmbeddingProvider(embeddingModelAt:)` load: multilingual-e5-small
converted to Core ML for the Neural Engine.

```sh
uv run scripts/embeddings/convert_e5_coreml.py --out ~/models/multilingual-e5-small
```

`uv` reads the pinned dependencies from the script header (Python 3.11, torch 2.7.0,
coremltools 9.0, transformers 4.57.6, sentence-transformers 5.2.0) and runs it in an environment it
keeps in its cache (`~/.cache/uv/environments-v2/convert-e5-coreml-*`, ~670 MB; delete that folder
when done). macOS only: the parity check runs the converted model through Core ML.

## What it does

1. Downloads [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
   at revision `614241f622f53c4eeff9890bdc4f31cfecc418b3` (MIT), only the files it needs (~490 MB).
2. Rewrites the encoder in Apple's Neural Engine layout (`(B, C, 1, S)` tensors, 1×1 convolutions,
   per-head attention, channel LayerNorm) and checks it against sentence-transformers in PyTorch.
3. Converts it to an fp16 ML Program (iOS 17+) with enumerated sequence lengths 64/128/256/512.
   The model returns token states `[1, S, 384]`; the caller mean-pools over non-padding tokens and
   L2-normalises (Swift does this in `CoreMLTextEmbeddingTool`).
4. Runs the Core ML model on `CPU_ONLY` and `CPU_AND_NE` for every bucket each sentence fits,
   on Spanish and English sentences plus one text longer than 512 tokens, and exits non-zero if the
   minimum cosine against sentence-transformers is below 0.999 (`--min-cosine`).
5. Writes the bundle; an existing `--out` is replaced only after every check passed.

```
multilingual-e5-small/
  embedding-model.json            manifest, schema aura.text-embedding/1
  MultilingualE5Small.mlpackage   ~225 MB
  tokenizer.json, tokenizer_config.json, special_tokens_map.json
```

Downloads and intermediates go to `--work-dir` (a temporary directory by default), which is deleted
at the end unless you pass `--keep-work-dir`; Hugging Face caches are kept inside it too.

## Conversion pitfalls it avoids

- Hugging Face BERT masks padding with `finfo(float32).min`, which is `-inf` in fp16: `0 × -inf`
  gives NaN on CPU/GPU (the Neural Engine hides it). The script uses a finite `-1e4` mask and
  validates on `CPU_ONLY`.
- LayerNorm epsilon `1e-12` underflows in fp16; it is raised to `1e-5`.
- Below iOS 18, coremltools rejects enumerated shapes on more than one input, so the attention mask
  is derived inside the graph from `input_ids != 1` and the model has a single input.
- A pooled output makes enumerated-shape models fall back to fp32 on the CPU, so pooling stays in
  the caller.

## Install it for the app

`DocsTab` looks for the bundle at `Application Support/AuraLocal/embeddings/multilingual-e5-small`;
its **Import embedding model…** action copies a bundle folder there. In code:

```swift
let embedder = AutoEmbeddingProvider(embeddingModelAt: bundleURL)   // TF-IDF if the bundle is unusable
try await embedder.warmUp()
```

See [Document RAG](../../docs/guide/rag.md) for the manifest format and how indexes switch providers.
