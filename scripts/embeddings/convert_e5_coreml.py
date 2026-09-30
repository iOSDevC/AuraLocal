# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "torch==2.7.0",
#     "coremltools==9.0",
#     "transformers==4.57.6",
#     "tokenizers==0.22.2",
#     "sentence-transformers==5.2.0",
#     "huggingface-hub==0.36.2",
#     "numpy==2.2.6",
# ]
# ///
"""Convert intfloat/multilingual-e5-small into an AuraLocal text-embedding bundle.

The bundle is what `CoreMLTextEmbeddingTool(bundleAt:)` loads:

    <out>/
      embedding-model.json        manifest, schema aura.text-embedding/1
      MultilingualE5Small.mlpackage
      tokenizer.json, tokenizer_config.json, special_tokens_map.json

The encoder is re-expressed in Apple's Neural Engine layout ((B, C, 1, S) tensors, 1x1
convolutions, per-head attention, channel LayerNorm) and converted to an fp16 ML Program with
enumerated sequence lengths 64/128/256/512. It returns token states; the caller does masked
mean pooling and L2 normalisation. Parity with sentence-transformers is checked on CPU_ONLY and
CPU_AND_NE before anything is written; the script exits non-zero if it is below the threshold.

    uv run scripts/embeddings/convert_e5_coreml.py --out ~/models/multilingual-e5-small
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

MODEL_ID = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
LICENSE = "MIT"
BUCKETS = (64, 128, 256, 512)
MAX_TOKENS = 512
PAD_ID = 1
DIMENSIONS = 384
MODEL_FILE = "MultilingualE5Small.mlpackage"
MANIFEST_FILE = "embedding-model.json"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
DOWNLOAD_PATTERNS = [
    "config.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "sentencepiece.bpe.model",
    *TOKENIZER_FILES,
]
MIN_COSINE = 0.999

PARITY_TEXTS = [
    "query: ¿Cuántas proteínas debe comer una mujer al día?",
    "passage: El niño pequeño comió piña y jalapeños en la montaña; señor Muñoz, ¿vio el cañón?",
    "query: how much protein should a female eat",
    "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
    "query: La cigüeña y el pingüino están en el año 2026, ¡qué ñoño!",
    "passage: Configura el iPhone 12 como nodo ligero de embeddings conectado por USB al Mac.",
    "passage: Precio: 1.234,56 € — 25 % de descuento; «oferta» válida hasta el 3 de marzo.",
    "query: What is the late payment penalty in the contract?",
]
# Longer than 512 tokens, so the truncation path is compared too.
LONG_TEXT = "passage: " + " ".join(
    f"La cláusula {n} del contrato establece que el pago se realizará en un plazo de treinta días." for n in range(1, 60)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="bundle directory to write (replaced if it exists)")
    parser.add_argument("--work-dir", type=Path, help="downloads and intermediates (default: a temporary directory)")
    parser.add_argument("--keep-work-dir", action="store_true", help="do not delete --work-dir afterwards")
    parser.add_argument("--min-cosine", type=float, default=MIN_COSINE)
    return parser.parse_args()


def download(work: Path) -> Path:
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_ID}@{REVISION[:7]} …", flush=True)
    return Path(
        snapshot_download(
            MODEL_ID,
            revision=REVISION,
            allow_patterns=DOWNLOAD_PATTERNS,
            cache_dir=work / "hf-cache",
        )
    )


# --- ANE-layout encoder --------------------------------------------------------------------------


def build_ane_encoder(bert):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LayerNormANE(nn.Module):
        def __init__(self, ln):
            super().__init__()
            # 1e-12 underflows in fp16.
            self.eps = max(ln.eps, 1e-5)
            self.weight = nn.Parameter(ln.weight.detach().clone().view(1, -1, 1, 1))
            self.bias = nn.Parameter(ln.bias.detach().clone().view(1, -1, 1, 1))

        def forward(self, x):
            centered = x - x.mean(dim=1, keepdim=True)
            variance = centered.pow(2).mean(dim=1, keepdim=True)
            return centered * torch.rsqrt(variance + self.eps) * self.weight + self.bias

    def conv(linear):
        layer = nn.Conv2d(linear.in_features, linear.out_features, 1)
        layer.weight.data = linear.weight.detach().clone()[:, :, None, None]
        layer.bias.data = linear.bias.detach().clone()
        return layer

    class LayerANE(nn.Module):
        def __init__(self, layer, heads):
            super().__init__()
            attention = layer.attention
            self.q, self.k, self.v = conv(attention.self.query), conv(attention.self.key), conv(attention.self.value)
            self.o, self.ln1 = conv(attention.output.dense), LayerNormANE(attention.output.LayerNorm)
            self.fc1, self.fc2 = conv(layer.intermediate.dense), conv(layer.output.dense)
            self.ln2 = LayerNormANE(layer.output.LayerNorm)
            self.heads = heads

        def forward(self, x, mask):  # x (B, C, 1, S); mask (B, S, 1, 1), additive over keys
            head_dim = x.shape[1] // self.heads
            scale = head_dim**-0.5
            outputs = []
            for qi, ki, vi in zip(self.q(x).split(head_dim, 1), self.k(x).split(head_dim, 1), self.v(x).split(head_dim, 1)):
                weights = torch.einsum("bchq,bkhc->bkhq", qi, ki.transpose(1, 3)) * scale + mask
                outputs.append(torch.einsum("bkhq,bchk->bchq", weights.softmax(dim=1), vi))
            x = self.ln1(self.o(torch.cat(outputs, dim=1)) + x)
            return self.ln2(self.fc2(F.gelu(self.fc1(x))) + x)

    class E5TokenEncoder(nn.Module):
        """Returns (1, S, 384) token states: pooled outputs lose the enumerated shapes on the ANE."""

        def __init__(self):
            super().__init__()
            self.emb = bert.embeddings
            self.layers = nn.ModuleList(LayerANE(layer, bert.config.num_attention_heads) for layer in bert.encoder.layer)

        def forward(self, input_ids):
            # The mask is derived in-graph: iOS 17 enumerated shapes allow only one flexible input.
            keep = (input_ids != PAD_ID).to(torch.float32)
            hidden = self.emb(input_ids=input_ids, token_type_ids=torch.zeros_like(input_ids))
            x = hidden.transpose(1, 2).unsqueeze(2)
            # HF's finfo.min mask is -inf in fp16 and turns into NaN on CPU/GPU; -1e4 stays finite.
            additive = ((1.0 - keep) * -1e4)[:, :, None, None]
            for layer in self.layers:
                x = layer(x, additive)
            return x.squeeze(2).transpose(1, 2)

    return E5TokenEncoder().eval()


# --- Helpers ------------------------------------------------------------------------------------


def token_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, truncation=True, max_length=MAX_TOKENS)["input_ids"]


def smallest_bucket(count: int) -> int:
    return next(size for size in BUCKETS if size >= count)


def padded(ids: list[int], bucket: int):
    import numpy as np

    row = np.full((1, bucket), PAD_ID, dtype=np.int32)
    row[0, : len(ids)] = ids
    return row


def pooled(hidden, ids_row):
    import numpy as np

    keep = (ids_row[0] != PAD_ID).astype(np.float32)[:, None]
    mean = (hidden[0].astype(np.float32) * keep).sum(axis=0) / max(keep.sum(), 1.0)
    return mean / max(float(np.linalg.norm(mean)), 1e-12)


def reference_embeddings(model_dir: Path, texts: list[str]):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_dir), device="cpu")
    model.max_seq_length = MAX_TOKENS
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def check_torch_parity(encoder, tokenizer, texts, reference) -> float:
    import numpy as np
    import torch

    cosines = []
    with torch.no_grad():
        for text, ref in zip(texts, reference):
            ids = token_ids(tokenizer, text)
            row = padded(ids, smallest_bucket(len(ids)))
            hidden = encoder(torch.from_numpy(row)).numpy()
            cosines.append(float(np.dot(pooled(hidden, row), ref)))
    return min(cosines)


def convert(encoder):
    import coremltools as ct
    import numpy as np
    import torch

    example = torch.full((1, 128), 5, dtype=torch.int32)
    example[0, 64:] = PAD_ID
    with torch.no_grad():
        traced = torch.jit.trace(encoder, (example,))
    shapes = ct.EnumeratedShapes(shapes=[(1, size) for size in BUCKETS], default=(1, 128))
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=shapes, dtype=np.int32)],
        outputs=[ct.TensorType(name="last_hidden_state", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    model.author = f"Converted from {MODEL_ID} ({LICENSE}) rev {REVISION}; ANE layout per apple/ml-ane-transformers"
    model.license = LICENSE
    model.short_description = (
        "multilingual-e5-small encoder, enumerated sequence lengths 64/128/256/512; "
        "the caller does masked mean pooling + L2"
    )
    model.input_description["input_ids"] = (
        "XLM-R ids incl. <s>/</s>, right-padded with <pad>=1 to 64/128/256/512; text prefixed 'query: '/'passage: '"
    )
    model.output_description["last_hidden_state"] = (
        "[1, S, 384] token states; mean over positions where input_ids != 1, then L2-normalise"
    )
    model.user_defined_metadata["hf_revision"] = REVISION
    model.user_defined_metadata["pad_token_id"] = str(PAD_ID)
    model.user_defined_metadata["seq_buckets"] = ",".join(map(str, BUCKETS))
    return model


def check_coreml_parity(package: Path, tokenizer, texts, reference) -> dict[str, float]:
    import coremltools as ct
    import numpy as np

    results = {}
    for name in ("CPU_ONLY", "CPU_AND_NE"):
        started = time.perf_counter()
        model = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, name))
        print(f"  {name}: loaded in {time.perf_counter() - started:.1f}s", flush=True)
        cosines = []
        for text, ref in zip(texts, reference):
            ids = token_ids(tokenizer, text)
            for bucket in (size for size in BUCKETS if size >= len(ids)):
                row = padded(ids, bucket)
                hidden = model.predict({"input_ids": row})["last_hidden_state"]
                if not np.isfinite(hidden).all():
                    sys.exit(f"{name}: non-finite output at sequence length {bucket}")
                cosines.append(float(np.dot(pooled(hidden, row), ref)))
        results[name] = round(min(cosines), 6)
        print(f"  {name}: min cosine {results[name]} over {len(cosines)} predictions", flush=True)
    return results


def manifest(parity: dict[str, float]) -> dict:
    return {
        "schema": "aura.text-embedding/1",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "model_file": MODEL_FILE,
        "input_name": "input_ids",
        "output_name": "last_hidden_state",
        "buckets": list(BUCKETS),
        "pad_token_id": PAD_ID,
        "pooling": "mean",
        "normalize": True,
        "dimensions": DIMENSIONS,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "max_tokens": MAX_TOKENS,
        "license": LICENSE,
        "parity_min_cosine": parity,
    }


def main() -> None:
    args = parse_args()
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="e5-coreml-"))
    work.mkdir(parents=True, exist_ok=True)
    # Keeps Hugging Face caches inside the work directory, so cleaning it removes them.
    os.environ["HF_HOME"] = str(work / "hf-home")
    out = args.out.expanduser().resolve()
    staging = out.with_name(out.name + ".partial")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.manual_seed(0)
        model_dir = download(work)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        bert = AutoModel.from_pretrained(model_dir, attn_implementation="eager").eval()

        texts = PARITY_TEXTS + [LONG_TEXT]
        if len(tokenizer(LONG_TEXT)["input_ids"]) <= MAX_TOKENS:
            sys.exit("LONG_TEXT no longer exercises truncation")
        reference = reference_embeddings(model_dir, texts)

        encoder = build_ane_encoder(bert)
        torch_cosine = check_torch_parity(encoder, tokenizer, texts, reference)
        print(f"PyTorch ANE-layout parity: min cosine {torch_cosine:.6f}", flush=True)
        if torch_cosine < args.min_cosine:
            sys.exit("the ANE-layout encoder diverges from sentence-transformers")

        print("Converting to Core ML (fp16, enumerated shapes) …", flush=True)
        package = work / MODEL_FILE
        shutil.rmtree(package, ignore_errors=True)
        convert(encoder).save(str(package))

        print("Checking Core ML parity …", flush=True)
        parity = check_coreml_parity(package, tokenizer, texts, reference)
        failing = {name: value for name, value in parity.items() if value < args.min_cosine}
        if failing:
            sys.exit(f"Core ML parity below {args.min_cosine}: {failing}")

        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        shutil.copytree(package, staging / MODEL_FILE)
        for name in TOKENIZER_FILES:
            shutil.copy2(model_dir / name, staging / name)
        (staging / MANIFEST_FILE).write_text(json.dumps(manifest(parity), indent=2) + "\n", encoding="utf-8")
        shutil.rmtree(out, ignore_errors=True)
        staging.rename(out)
        print(f"Bundle written to {out}", flush=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if not args.keep_work_dir:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
