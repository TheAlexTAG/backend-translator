# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
from text_processing import process
from schemas import TranslateRequest, TranslateResponse, MixedBatchRequest
import translation_service as svc

def create_app(allow_all_origins: bool = True,
               specific_origins: Optional[List[str]] = None) -> FastAPI:
    app = FastAPI(title="MT (CTranslate2 + Marian/OPUS-MT)")

    if allow_all_origins:
        origins = ["*"]
        allow_credentials = False
    else:
        origins = specific_origins or []
        allow_credentials = False  # set True only if you really need cookies + explicit origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
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
            processesTexts = process(req.texts)
            translations = svc.translate_many(processesTexts, req.src_lang, req.tgt_lang, req.options)
            return TranslateResponse(translations=translations)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/translate/mixed", response_model=TranslateResponse)
    def translate_mixed(req: MixedBatchRequest):
        try:
            items = [it.model_dump() for it in req.items]
            translations = svc.translate_mixed(items)
            return TranslateResponse(translations=translations)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/translate_batch")
    def translate_batch_compat(req: MixedBatchRequest):
        try:
            items = [it.model_dump() for it in req.items]
            return {"translations": svc.translate_mixed(items)}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return app

# default app so `uvicorn main:app --host 0.0.0.0 --port 8000` still works
app = create_app()
