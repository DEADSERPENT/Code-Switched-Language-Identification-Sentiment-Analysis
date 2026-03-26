"""
FastAPI backend for CoSwitchNLP.

Endpoints:
    POST /analyze       — analyse a single piece of text
    POST /batch         — analyse a list of texts
    GET  /health        — server + model status
    GET  /examples      — return preloaded demo sentences

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from inference import CoSwitchInference

MODEL_DIR = os.environ.get("MODEL_DIR", "../models/coswitchnlp_v1")
pipeline: CoSwitchInference | None = None

DEMO_EXAMPLES = [
    {
        "text": "yaar ye movie bohot amazing thi",
        "label": "Hindi-dominant, positive",
    },
    {
        "text": "kya bakwas hai ye product, waste of money",
        "label": "Negative, strong code-mix",
    },
    {
        "text": "thik thak hai, nothing special about it",
        "label": "Neutral",
    },
    {
        "text": "bhai maza aa gaya, what a performance!",
        "label": "Positive, equal mix",
    },
    {
        "text": "mere paas abhi time nahi hai, will check later",
        "label": "Neutral, mostly Hindi",
    },
    {
        "text": "totally loved it yaar, dil khush ho gaya",
        "label": "Positive, English-start Hindi-end",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    if os.path.exists(MODEL_DIR) and os.path.exists(
        os.path.join(MODEL_DIR, "model.pt")
    ):
        pipeline = CoSwitchInference(MODEL_DIR)
    else:
        print(
            f"WARNING: Model not found at {MODEL_DIR}. "
            "Run train.py first. The /analyze endpoint will return 503 until the model is loaded."
        )
    yield


app = FastAPI(
    title="CoSwitchNLP API",
    description="Token-level language identification and sentiment analysis for Hinglish text",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas ─────────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        if len(v) > 1000:
            raise ValueError("text must be ≤1000 characters")
        return v


class BatchRequest(BaseModel):
    texts: list[str]

    @field_validator("texts")
    @classmethod
    def texts_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("texts list must not be empty")
        if len(v) > 50:
            raise ValueError("batch size must be ≤50")
        return v


class TokenResult(BaseModel):
    token: str
    language: str
    confidence: float


class AnalysisResponse(BaseModel):
    tokens: list[TokenResult]
    sentiment: str
    sentiment_confidence: float
    sentiment_scores: dict[str, float]
    code_mixing_index: float
    language_distribution: dict[str, float]
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_dir: str


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return HealthResponse(
        status="ok",
        model_loaded=pipeline is not None,
        device=device,
        model_dir=MODEL_DIR,
    )


@app.get("/examples")
async def examples():
    return {"examples": DEMO_EXAMPLES}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(req: AnalyzeRequest):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py first and ensure MODEL_DIR is correct.",
        )
    try:
        t0 = time.perf_counter()
        result = pipeline.predict(req.text)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return AnalysisResponse(
            tokens=[
                TokenResult(
                    token=t.token, language=t.language, confidence=t.confidence
                )
                for t in result.tokens
            ],
            sentiment=result.sentiment,
            sentiment_confidence=result.sentiment_confidence,
            sentiment_scores=result.sentiment_scores,
            code_mixing_index=result.code_mixing_index,
            language_distribution=result.language_distribution,
            processing_time_ms=round(elapsed_ms, 2),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/batch")
async def batch_analyze(req: BatchRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        t0 = time.perf_counter()
        results = pipeline.batch_predict(req.texts)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        output = []
        for result in results:
            output.append(
                {
                    "tokens": [
                        {
                            "token": t.token,
                            "language": t.language,
                            "confidence": t.confidence,
                        }
                        for t in result.tokens
                    ],
                    "sentiment": result.sentiment,
                    "sentiment_confidence": result.sentiment_confidence,
                    "sentiment_scores": result.sentiment_scores,
                    "code_mixing_index": result.code_mixing_index,
                    "language_distribution": result.language_distribution,
                }
            )
        return {"results": output, "processing_time_ms": round(elapsed_ms, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
