# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

from schemas import TranslateRequest, TranslateResponse, MixedBatchRequest
import translation_service as svc

app = FastAPI(title="MT (CTranslate2 + Marian/OPUS-MT)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://freefu.it", "chrome-extension://*"],  # or ["*"] if you don't use cookies
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/health")
def health() -> Dict[str, str | list]:
    return {
        "status": "ok",
        "pairs": svc.list_pairs(),
        "device": svc.DEVICE,
    }

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    try:
        translations = svc.translate_many(req.texts, req.src_lang, req.tgt_lang, req.options)
        return TranslateResponse(translations=translations)
    except ValueError as e:
        # e.g. unsupported pair
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/translate/mixed", response_model=TranslateResponse)
def translate_mixed(req: MixedBatchRequest):
    try:
        items = [it.model_dump() for it in req.items]
        translations = svc.translate_mixed(items)
        return TranslateResponse(translations=translations)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# (Optional) keep your old contract alive
@app.post("/translate_batch")
def translate_batch_compat(req: MixedBatchRequest):
    try:
        items = [it.model_dump() for it in req.items]
        return {"translations": svc.translate_mixed(items)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
