"""
Agent API (no LangChain) — Tavily search + Gemini summarize + optional scoring via Embedder.
Run: uvicorn services.agent.agent:app --reload --port 8100
"""

import os, json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tavily import TavilyClient
import google.generativeai as genai
import re
import httpx

# App + external client setup
load_dotenv()

app = FastAPI(title="Agent API (SDK)", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")  # from your model list
EMBEDDER_URL = os.getenv("EMBEDDER_URL", "http://localhost:8000")

if not TAVILY_API_KEY:
    raise RuntimeError("Missing TAVILY_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")

tavily = TavilyClient(api_key=TAVILY_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel(GEMINI_MODEL)

# Pydantic models
class NormalizeResumeRequest(BaseModel):
    resume_text: str = Field(..., description="Raw resume text pasted by the user.")

class NormalizedResume(BaseModel):
    summary: str = ""
    skills: List[str] = []
    experience: List[Dict[str, Any]] = []
    education: List[Dict[str, Any]] = []

class NormalizeResumeResponse(BaseModel):
    resume: NormalizedResume

class JobRecord(BaseModel):
    id: str
    title: str = ""
    company: str = ""
    url: str
    summary: str = ""
    raw_excerpt: Optional[str] = None

class JobSearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = Field(default=10, ge=1, le=25)
    summarize: bool = True

class JobSearchResponse(BaseModel):
    jobs: List[JobRecord]

class JobScoringRequest(BaseModel):
    resume_sections: List[str]
    jobs: List[JobRecord]

class JobScoringResponse(BaseModel):
    scores: List[Dict[str, Any]]  # [{job_id, similarity}]


GEN_CONFIG = {
    "temperature": 0.2,
    "response_mime_type": "application/json",  # <-- force JSON
}

def gemini_json(prompt: str) -> dict:
    """
    Call Gemini and parse JSON. If parsing fails, try to salvage with a regex,
    then raise a 502 with the first 400 chars of what Gemini returned.
    """
    try:
        resp = gemini.generate_content(prompt, generation_config=GEN_CONFIG)
        text = (resp.text or "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # salvage: extract the first JSON-looking block
            m = re.search(r'\{(?:.|\n)*\}|\[(?:.|\n)*\]', text)
            if m:
                return json.loads(m.group(0))
            raise HTTPException(
                status_code=502,
                detail=f"Gemini JSON parse failed: {text[:400]}"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini call failed: {e}")


def build_resume_prompt(raw_text: str) -> str:
    return (
        "Extract structured JSON from the following resume text.\n"
        "Return **ONLY** valid JSON. No prose.\n"
        "Schema exactly:\n"
        "{\n"
        '  "summary": string,\n'
        '  "skills": [string],\n'
        '  "experience": [ {"company": string, "role": string, "bullets": [string]} ],\n'
        '  "education": [ {"school": string, "degree": string, "year": string} ]\n'
        "}\n\n"
        f"RESUME START\n{raw_text}\nRESUME END\n"
    )

def build_job_normalize_prompt(items: list[dict]) -> str:
    return (
        "Normalize these search results into a JSON array of job objects.\n"
        "Return **ONLY** valid JSON. No prose.\n"
        "Array schema:\n"
        "[{\n"
        '  "id": "job_001",\n'
        '  "title": string,\n'
        '  "company": string,\n'
        '  "url": string,\n'
        '  "summary": string,\n'
        '  "raw_excerpt": string\n'
        "}]\n\n"
        f"RESULTS:\n{json.dumps(items, ensure_ascii=False)}\n"
    )


DEBUG_MODE = os.getenv("DEBUG_AGENT", "0") == "1"

@app.get("/gemini_health")
def gemini_health():
    try:
        test = gemini.generate_content("ping", generation_config={"response_mime_type":"text/plain"}).text
        return {"ok": True, "model": GEMINI_MODEL, "echo": (test or "")[:20]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/normalize_resume", response_model=NormalizeResumeResponse)
def normalize_resume(req: NormalizeResumeRequest) -> NormalizeResumeResponse:
    prompt = build_resume_prompt(req.resume_text)
    data = gemini_json(prompt)
    # validate into pydantic model
    normalized = NormalizedResume(**data)
    return NormalizeResumeResponse(resume=normalized)

@app.post("/agent/search_jobs", response_model=JobSearchResponse)
def search_jobs(req: JobSearchRequest) -> JobSearchResponse:
    raw = tavily.search(query=req.query, max_results=req.limit)
    results = raw.get("results", [])
    if not results:
        return JobSearchResponse(jobs=[])

    if req.summarize:
        condensed = []
        for index, response in enumerate(results[:req.limit], start=1):
            condensed.append({
                "index": index,
                "title": response.get("title", "") or "",
                "url": response.get("url", "") or "",
                "content": (response.get("content", "") or "")[:1200],
            })
        prompt = build_job_normalize_prompt(condensed)
        parsed = gemini_json(prompt)
        jobs: List[JobRecord] = []
        for j, row in enumerate(parsed, start=1):
            jobs.append(JobRecord(
                id=row.get("id") or f"job_{j:03d}",
                title=row.get("title",""),
                company=row.get("company",""),
                url=row.get("url",""),
                summary=row.get("summary","")[:600],
                raw_excerpt=row.get("raw_excerpt")
            ))
        return JobSearchResponse(jobs=jobs)

    # Passthrough fallback
    jobs = []
    for index, response in enumerate(results[:req.limit], start=1):
        jobs.append(JobRecord(
            id=f"job_{index:03d}",
            title=response.get("title","") or "Job",
            company="",
            url=response.get("url",""),
            summary=(response.get("content","") or "")[:600],
            raw_excerpt=(response.get("content","") or "")[:600],
        ))
    return JobSearchResponse(jobs=jobs)

@app.post("/agent/score_jobs", response_model=JobScoringResponse)
async def score_jobs(req: JobScoringRequest) -> JobScoringResponse:
    # Forward to embedder /score_jobs
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            response = await client.post(f"{EMBEDDER_URL}/score_jobs", json=req.dict())
            response.raise_for_status()
            data = response.json()
            return JobScoringResponse(**data)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Embedder error: {e}")
