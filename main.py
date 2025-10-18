from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Tuple, List
import ctranslate2 as ct2
import sentencepiece as spm
from collections import defaultdict



BASE = Path(__file__).parent.resolve()
REGISTRY = BASE / "registry.tsv"

app = FastAPI(title="MT (CTranslate2 + Marian/OPUS-MT)")

class Req(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str
    options: dict | None = None

class BatchReq(BaseModel):
    items: List[Req]

# Load registry lines:
# pair \t ct2_dir \t spm_src \t spm_tgt \t tok_dir(optional)
_pairs: Dict[Tuple[str, str], dict] = {}
if REGISTRY.exists():
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        # tolerate both 4- and 5-column registries
        if len(parts) >= 4:
            pair, ct2_dir, spm_src, spm_tgt = parts[:4]
            src, tgt = pair.split("->")
            _pairs[(src, tgt)] = dict(ct2_dir=ct2_dir, spm_src=spm_src, spm_tgt=spm_tgt)
else:
    print("WARNING: registry.tsv not found. Run convert.py first.")

_translators: Dict[Tuple[str, str], ct2.Translator] = {}
_src_tok: Dict[str, spm.SentencePieceProcessor] = {}
_tgt_tok: Dict[str, spm.SentencePieceProcessor] = {}

def get_pair(src: str, tgt: str):
    key = (src, tgt)
    cfg = _pairs.get(key)
    if not cfg:
        raise HTTPException(status_code=400, detail=f"unsupported_pair {src}->{tgt}")
    if key not in _translators:
        _translators[key] = ct2.Translator(cfg["ct2_dir"], device="cpu")
    if cfg["spm_src"] not in _src_tok:
        _src_tok[cfg["spm_src"]] = spm.SentencePieceProcessor(model_file=cfg["spm_src"])
    if cfg["spm_tgt"] not in _tgt_tok:
        _tgt_tok[cfg["spm_tgt"]] = spm.SentencePieceProcessor(model_file=cfg["spm_tgt"])
    return _translators[key], _src_tok[cfg["spm_src"]], _tgt_tok[cfg["spm_tgt"]]

def translate_one(text: str, src: str, tgt: str, options: dict | None = None) -> str:
    """Single-item translation using the same logic as your original /translate."""
    if src == tgt:
        return text

    translator, sp_src, sp_tgt = get_pair(src, tgt)

    pieces = sp_src.encode(text, out_type=str)
    pieces_in = pieces + ["</s>"]

    opts = options or {}
    beam_size = int(opts.get("beam_size", 4))
    max_len = int(opts.get("max_new_tokens", 200))

    results = translator.translate_batch(
        [pieces_in],
        beam_size=beam_size,
        max_decoding_length=max_len,
        end_token="</s>",
    )

    best = results[0].hypotheses[0] if results and results[0].hypotheses else []
    best = [t for t in best if t != "</s>"]

    return sp_tgt.decode_pieces(best)


@app.get("/health")
def health():
    return {"status": "ok", "pairs": [f"{s}->{t}" for (s, t) in _pairs.keys()], "device": "cpu"}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/translate")
def translate(r: Req):
    return {"translation": translate_one(r.text, r.src_lang, r.tgt_lang, r.options)}

@app.post("/translate_batch")
def translate_batch(req: BatchReq):
    """
    Batch translation endpoint.
    - Groups items by (src_lang, tgt_lang) to reuse the same model & tokenizers.
    - Calls CT2 translate_batch once per group for efficiency.
    - Keeps output aligned to input order.
    """
    if not req.items:
        return {"translations": []}

    # Group input indices by language pair
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, it in enumerate(req.items):
        groups[(it.src_lang, it.tgt_lang)].append(i)

    out: List[str] = [""] * len(req.items)

    for (src, tgt), idxs in groups.items():
        # Fast-path: identities
        all_identity = True
        for i in idxs:
            it = req.items[i]
            if it.src_lang != it.tgt_lang:
                all_identity = False
                break
        if all_identity:
            for i in idxs:
                out[i] = req.items[i].text
            continue

        # Load model + tokenizers once per group
        translator, sp_src, sp_tgt = get_pair(src, tgt)

        # Prepare batch (skip identity items)
        batch_inputs = []
        map_back: List[int] = []  # indices within idxs that correspond to batch rows
        beam_sizes = []
        max_lens = []

        for local_pos, i in enumerate(idxs):
            it = req.items[i]
            if it.src_lang == it.tgt_lang:
                out[i] = it.text
                continue

            pieces = sp_src.encode(it.text, out_type=str)
            batch_inputs.append(pieces + ["</s>"])
            map_back.append(i)

            opts = (it.options or {})
            beam_sizes.append(int(opts.get("beam_size", 4)))
            max_lens.append(int(opts.get("max_new_tokens", 200)))

        if not batch_inputs:
            continue

        # Use a common beam_size/max_len for the sub-batch (fast path).
        beam = max(beam_sizes) if beam_sizes else 4
        max_len = max(max_lens) if max_lens else 200

        results = translator.translate_batch(
            batch_inputs,
            beam_size=beam,
            max_decoding_length=max_len,
            end_token="</s>",
        )

        # Decode each hypothesis back into the correct original position
        for k, i in enumerate(map_back):
            hyp = results[k].hypotheses[0] if results and results[k].hypotheses else []
            hyp = [t for t in hyp if t != "</s>"]
            out[i] = sp_tgt.decode_pieces(hyp)

    return {"translations": out}
