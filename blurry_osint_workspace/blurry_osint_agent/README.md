# Blurry OSINT Tracing Agent (Demo)

Python runnable demo that implements the full closed-loop workflow described in your specification, including tool orchestration, iteration limits, structured logs, and deterministic mock tools. Real adapters are wired for SauceNAO, Nominatim, and Web-Check; they are best-effort and safe to fail without breaking the demo.

## What This System Does

This system traces the likely origin of blurry or low-quality images using a closed-loop OSINT workflow. It extracts visual features, enhances images, performs multi-engine source discovery, mines metadata, fuses evidence, and iterates when confidence is low. Outputs are structured for downstream automation and auditability.

## Quick Start

```powershell
cd e:\agent_perplexity\blurry_osint_workspace\blurry_osint_agent
python -m src.cli --image "path\\to\\image.jpg" --mode mock
```

If no image is available, you can still run with a placeholder path:

```powershell
python -m src.cli --image "C:\\fake\\image.jpg" --mode mock
```

## JSON Output

```powershell
python -m src.cli --image "path\\to\\image.jpg" --mode mock --output json
```

## Gradio UI

```powershell
python -m src.ui_gradio
```

## LangChain Integration

This demo includes a LangChain runnable wrapper. Install:

```powershell
pip install langchain-core
```

## Real-Mode Demo (API-Ready + Network Calls)

```powershell
python -m src.cli --image "path\\to\\image.jpg" --mode real
```

Environment variables for real mode:

- `SAUCENAO_API_KEY`: SauceNAO API key
- `WEB_CHECK_BASE_URL`: Web-Check base URL, e.g. `https://web-check.as93.net`
- `WEB_CHECK_ENDPOINT`: Endpoint path, default `/api/check`
- `WEB_CHECK_METHOD`: `GET` or `POST` (default `GET`)
- `WEB_CHECK_API_KEY`: Optional token if your Web-Check requires it
- `NOMINATIM_BASE_URL`: Default `https://nominatim.openstreetmap.org/reverse`

Note: real external calls can fail if keys are missing or endpoints differ from your deployment.

## Dependencies (Optional)

```powershell
pip install requests exifread opencv-python numpy lmdeploy gradio
```

## API Stack (Your Selected Demo Combo)

- Qwen-VL-Chat + LMDeploy (VLM perception)
- SauceNAO (source discovery)
- ExifRead (EXIF parsing)
- Nominatim (reverse geocoding)
- Web-Check (web metadata)

## Project Layout

- `src/agent.py` Core agent pipeline
- `src/tools.py` Tool stubs + real adapters
- `src/models.py` Data models and result structures
- `src/cli.py` Command-line entry point
- `src/ui_gradio.py` Gradio UI
- `src/langchain_adapter.py` LangChain wrapper
- `src/config.py` Constants and thresholds

## Notes

- Mock tools never call external services.
- Real adapters are best-effort and fail gracefully.
- Public Nominatim has usage limits; prefer self-host for heavy usage.
