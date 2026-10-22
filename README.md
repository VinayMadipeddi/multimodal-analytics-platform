# Multimodal Multi-Agent Intelligent Data Analytics Platform

A runnable full-stack capstone project built from the supplied HTML UI, with a FastAPI backend, SQLite persistence, real CSV/Excel processing, data cleaning, statistics, charts, six-step workflow audit logs, document text search, and downloadable PDF reports.

## Features
- 11-page dashboard UI from the supplied HTML file.
- Upload CSV, XLSX/XLS, PDF, images, audio, and text files (50 MB/file limit).
- CSV/Excel uploads become working datasets; the raw uploaded file is retained separately.
- Data profiling, schema, preview, missingness, duplicate detection.
- Cleaning: imputation, drop missing rows, deduplication, IQR outlier filtering/winsorization, text normalization.
- Export cleaned data to CSV or Excel.
- Interactive Plotly charts and statistical tests (Welch t-test, one-way ANOVA, chi-square).
- Six-agent workflow execution records persisted to SQLite.
- Text extraction from TXT and PDFs when dependencies support it; optional image OCR with Tesseract.
- Keyword retrieval over extracted document text and deterministic, dataset-grounded analyst responses.
- PDF executive reports.
- Session, file, workflow, report, and cleaning history persistence.

## Requirements
- Python 3.10+
- For image OCR, install the Tesseract executable separately and make it available on PATH. Image OCR is optional.
- Audio files are accepted and stored; speech-to-text is an adapter placeholder and needs a transcription provider/model to transcribe audio.

## Run locally (macOS/Linux)
From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. API documentation: http://127.0.0.1:8000/docs

On Windows, activate with `.venv\\Scripts\\activate` instead.

## Docker
```bash
docker compose up --build
```
Then open http://localhost:8000.

## Data storage
The SQLite database and uploaded/processed files are created under `data/`. They persist between restarts. Do not upload sensitive data to an internet-facing deployment without adding authentication, authorization, encryption, rate limiting, and a retention policy.

## Notes / honest capability boundaries
- The six agents are implemented as deterministic workflow stages with persisted start/completion records; this starter does not require an external LLM or pretend that agents run independently in the background.
- Chat answers use calculated dataset metrics and retrieved document text. No cloud LLM is called by default.
- OCR and audio transcription depend on local/external engines; PDF extraction works when the PDF contains extractable text. Scanned PDFs may need OCR.
- The app is a development/capstone baseline, not a hardened multi-user production service.
