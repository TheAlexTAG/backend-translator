# MT Service — CTranslate2 + OPUS-MT (it↔en)

Minimal translation microservice that powers the demo app.

- **Models:** Helsinki-NLP OPUS-MT (Marian) `it→en`, `en→it`
- **Runtime:** CTranslate2 (INT8, **CPU by default**)
- **Server:** FastAPI
- **Location:** `apps/mt`

> Large artifacts (models, venv, registry) are **gitignored**. Each dev converts locally.

---

## Requirements

- **Python** 3.12 or 3.13
- **pip** 23+
- Windows PowerShell or macOS/Linux shell
- Internet access (to download models from Hugging Face)

---

## Quick Start

```powershell
# 0) go to the service
cd apps\mt

# 1) create + activate venv
py -3.12 -m venv .venv            # or: py -3.13 -m venv .venv
. .\.venv\Scripts\Activate.ps1     # macOS/Linux: source .venv/bin/activate

# 2) install deps (CPU)
python -m pip install --upgrade pip
python -m pip install "torch>=2.6,<2.7" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

# (optional) verify torch
python -m pip show torch

# 3) download & convert models → creates ct2 models + registry.tsv
python convert.py

# 4) run the API
uvicorn main:app --host 0.0.0.0 --port 6001
```

## Test the service

After starting the server (`uvicorn main:app --host 0.0.0.0 --port 6001`), you can verify it with:

```powershell
(Invoke-RestMethod -Uri "http://localhost:6001/translate" -Method Post -ContentType "application/json" -Body '{"text":"Ciao mondo","src_lang":"it","tgt_lang":"en"}').translation
```

---

## Deactivate the venv

To return to the deafult cmd, simply run:

```powershell
deactivate
```
