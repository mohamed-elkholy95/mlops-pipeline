"""FastAPI."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
app = FastAPI(title="MLOps Pipeline API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health(): return {"status": "healthy"}

@app.get("/pipeline/status")
async def pipeline_status():
    return {"status": "idle", "pipelines_run": 0}

@app.get("/models")
async def list_models():
    from src.pipeline import ModelRegistry
    return {"models": ModelRegistry().list_models()}

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8015)
