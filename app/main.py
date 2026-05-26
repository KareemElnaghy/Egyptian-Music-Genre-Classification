import os
import sys
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Final", "src"))
from predict import predict  # noqa: E402  (import after sys.path patch)

app = FastAPI(title="Egyptian Music Genre Classifier")
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

_ALLOWED = {".wav", ".mp3"}
_MAX_BYTES = 50 * 1024 * 1024  # 50 MB


@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Please upload a .wav or .mp3 file.",
        )

    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 50 MB.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        result = predict(tmp_path)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

print("http://localhost:8000")
