# translation_service.py
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
import os

import ctranslate2 as ct2
import sentencepiece as spm

# ---- Config ----
BASE = Path(__file__).parent.resolve()
REGISTRY = Path(os.getenv("MT_REGISTRY_PATH", BASE / "registry.tsv"))
DEVICE = os.getenv("MT_DEVICE", "cpu")  # "cpu" | "cuda"

# ---- Registry ----
_pairs: Dict[Tuple[str, str], dict] = {}
if REGISTRY.exists():
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 4:
            pair, ct2_dir, spm_src, spm_tgt = parts[:4]
            src, tgt = pair.split("->")
            _pairs[(src, tgt)] = dict(ct2_dir=ct2_dir, spm_src=spm_src, spm_tgt=spm_tgt)
else:
    print(f"WARNING: registry.tsv not found at {REGISTRY}. Run convert.py first.")

# ---- Lazy caches ----
_translators: Dict[Tuple[str, str], ct2.Translator] = {}
_src_tok: Dict[str, spm.SentencePieceProcessor] = {}
_tgt_tok: Dict[str, spm.SentencePieceProcessor] = {}

def list_pairs() -> List[str]:
    """Return available pairs like 'en->it'."""
    return [f"{s}->{t}" for (s, t) in _pairs.keys()]

def _get_pair(src: str, tgt: str):
    """Resolve and lazy-load the translator and tokenizers for (src, tgt)."""
    key = (src, tgt)
    cfg = _pairs.get(key)
    if not cfg:
        raise ValueError(f"unsupported_pair {src}->{tgt}")

    if key not in _translators:
        _translators[key] = ct2.Translator(cfg["ct2_dir"], device=DEVICE)

    if cfg["spm_src"] not in _src_tok:
        _src_tok[cfg["spm_src"]] = spm.SentencePieceProcessor(model_file=cfg["spm_src"])
    if cfg["spm_tgt"] not in _tgt_tok:
        _tgt_tok[cfg["spm_tgt"]] = spm.SentencePieceProcessor(model_file=cfg["spm_tgt"])
    return _translators[key], _src_tok[cfg["spm_src"]], _tgt_tok[cfg["spm_tgt"]]

def _decode_options(opts: Optional[dict]) -> tuple[int, int]:
    """Get (beam_size, max_new_tokens) with defaults."""
    opts = opts or {}
    return int(opts.get("beam_size", 4)), int(opts.get("max_new_tokens", 200))

def translate_many(texts: List[str], src: str, tgt: str, options: Optional[dict] = None) -> List[str]:
    """
    Translate a list of texts with a single language pair.
    Identity fast-path when src == tgt.
    """
    if src == tgt:
        return texts[:]

    translator, sp_src, sp_tgt = _get_pair(src, tgt)
    beam, max_len = _decode_options(options)
    batch_inputs = [(sp_src.encode(t, out_type=str) + ["</s>"]) for t in texts]

    results = translator.translate_batch(
        batch_inputs,
        beam_size=beam,
        max_decoding_length=max_len,
        end_token="</s>",
    )

    outs: List[str] = []
    for r in results:
        hyp = r.hypotheses[0] if r and r.hypotheses else []
        hyp = [t for t in hyp if t != "</s>"]
        outs.append(sp_tgt.decode_pieces(hyp))
    return outs

def translate_mixed(items: List[dict]) -> List[str]:
    """
    Translate items with potentially different pairs/options.
    Each item dict: {text, src_lang, tgt_lang, options?}
    Preserves input order.
    """
    if not items:
        return []

    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, it in enumerate(items):
        groups[(it["src_lang"], it["tgt_lang"])].append(i)

    out: List[str] = [""] * len(items)

    for (src, tgt), idxs in groups.items():
        if src == tgt:
            for i in idxs:
                out[i] = items[i]["text"]
            continue

        translator, sp_src, sp_tgt = _get_pair(src, tgt)

        batch_inputs = []
        map_back: List[int] = []
        group_beam, group_max_len = 4, 200

        for i in idxs:
            it = items[i]
            pieces = sp_src.encode(it["text"], out_type=str)
            batch_inputs.append(pieces + ["</s>"])
            map_back.append(i)

            beam, mx = _decode_options(it.get("options"))
            group_beam = max(group_beam, beam)
            group_max_len = max(group_max_len, mx)

        results = translator.translate_batch(
            batch_inputs,
            beam_size=group_beam,
            max_decoding_length=group_max_len,
            end_token="</s>",
        )

        for k, i in enumerate(map_back):
            r = results[k]
            hyp = r.hypotheses[0] if r and r.hypotheses else []
            hyp = [t for t in hyp if t != "</s>"]
            out[i] = sp_tgt.decode_pieces(hyp)

    return out
