# schemas.py
from typing import List, Optional
from pydantic import BaseModel

class TranslateRequest(BaseModel):
    texts: List[str]
    src_lang: str
    tgt_lang: str
    options: Optional[dict] = None

class TranslateResponse(BaseModel):
    translations: List[str]

class MixedItem(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str
    options: Optional[dict] = None

class MixedBatchRequest(BaseModel):
    items: List[MixedItem]
