import json
import os
import re
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import google.generativeai as genai
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()

app = FastAPI(title="Agent API (SDK)", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Setup application third party services
TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
EMBEDDER_BASE_URL: str = os.getenv("EMBEDDER_URL", "http://localhost:8000")

if not TAVILY_API_KEY:
    raise RuntimeError("Missing TAVILY_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

DEBUG_MODE: bool = os.getenv("DEBUG_AGENT", "0") == "1"


# Setup Pydantic models for application
class NormalizeResumeRequest(BaseModel):
    """Request body: raw resume text to be normalized into structured JSON."""
    resume_text: str = Field(..., description="Raw resume text pasted by the user.")

class NormalizedResume(BaseModel):
    """Structured representation of a resume after LLM normalization."""
    summary: str = ""
    skills: List[str] = []
    experience: List[Dict[str, Any]] = []
    education: List[Dict[str, Any]] = []

class NormalizeResumeResponse(BaseModel):
    """Response body: normalized resume object."""
    resume: NormalizedResume

class JobRecord(BaseModel):
    """Normalized job posting record used throughout the API."""
    id: str
    title: str = ""
    company: str = ""
    url: str
    summary: str = ""
    raw_excerpt: Optional[str] = None

class JobSearchRequest(BaseModel):
    """Request body: job search parameters."""
    query: str
    filters: Optional[Dict[str, Any]] = None
    # Limit must be between 1 and 25 inclusive
    limit: int = Field(default=10, ge=1, le=25)
    summarize: bool = True

class JobSearchResponse(BaseModel):
    """Response body: a list of normalized job records."""
    jobs: List[JobRecord]

class JobScoringRequest(BaseModel):
    """Request body: resume sections and jobs to score for similarity."""
    resume_sections: List[str]
    jobs: List[JobRecord]

class JobScoringResponse(BaseModel):
    """Response body: similarity scores for jobs, e.g., [{job_id, similarity}]."""
    scores: List[Dict[str, Any]]

class AnalyzeAllRequest(BaseModel):
    """Request body: orchestrated end-to-end analysis (normalize + search + score)."""
    resume_text: str
    job_query: str
    limit: int = Field(default=8, ge=1, le=25)

class AnalyzeAllResponse(BaseModel):
    """Response body: normalized resume, job list, and similarity scores."""
    resume: NormalizedResume
    jobs: List[JobRecord]
    # Example element: { "job_id": "job_001", "similarity": 0.87 }
    scores: List[Dict[str, Any]]

class ChatRequest(BaseModel):
    """Request body: lightweight chat endpoint with optional inlined resume."""
    message: str
    resume_text: Optional[str] = None
    limit: int = Field(default=8, ge=1, le=25)

class ChatResponse(BaseModel):
    """Response body: chat reply and optionally resume, jobs, and scores."""
    reply: str
    resume: Optional[NormalizedResume] = None
    jobs: Optional[List[JobRecord]] = None
    scores: Optional[List[Dict[str, Any]]] = None


# LLM Config
GENERATION_CONFIG: Dict[str, Any] = {
    "temperature": 0.2,
    "response_mime_type": "application/json",
}

def gemini_json(user_prompt: str) -> dict:
    """
    Call Gemini with a prompt that should return JSON and parse the response.

    Behavior:
    - Attempts to parse the response text as JSON directly.
    - If initial parse fails, uses a regex to extract the first JSON-looking block,
      then attempts parsing again.
    - If still unsuccessful, raises HTTP 502 with the first 400 characters of the
      raw model output to aid debugging.

    Args:
        user_prompt: The instruction/prompt sent to the Gemini model.

    Returns:
        A Python dictionary parsed from the model's JSON response.

    Raises:
        HTTPException(502): If the LLM call fails or JSON parsing ultimately fails.
    """
    try:
        gemini_response = gemini_model.generate_content(user_prompt, generation_config=GENERATION_CONFIG)
        response_text = (gemini_response.text or "").strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to salvage a JSON object/array via regex.
            json_like_match = re.search(r"\{(?:.|\n)*\}|\[(?:.|\n)*\]", response_text)
            if json_like_match:
                return json.loads(json_like_match.group(0))

            raise HTTPException(
                status_code=502,
                detail=f"Gemini JSON parse failed: {response_text[:400]}"
            )
    except HTTPException:
        raise
    except Exception as unexpected_error:
        raise HTTPException(status_code=502, detail=f"Gemini call failed: {unexpected_error}")


def build_resume_prompt(raw_resume_text: str) -> str:
    """
    Build a JSON instruction prompt for resume normalization.

    Args:
        raw_resume_text: Resume text provided by the user.

    Returns:
        A formatted prompt string instructing the model to output JSON only.
    """
    return (
        "Extract structured JSON from the following resume text.\n"
        "Return **ONLY** valid JSON. No prose.\n"
        "Schema exactly:\n"
        "{\n"
        '  \"summary\": string,\n'
        '  \"skills\": [string],\n'
        '  \"experience\": [ {\"company\": string, \"role\": string, \"bullets\": [string]} ],\n'
        '  \"education\": [ {\"school\": string, \"degree\": string, \"year\": string} ]\n'
        "}\n\n"
        f"RESUME START\n{raw_resume_text}\nRESUME END\n"
    )


def build_job_normalize_prompt(raw_items: List[Dict[str, Any]]) -> str:
    """
    Build a JSON instruction prompt to normalize a list of raw search results
    into job objects. The model is asked to return ONLY valid JSON matching the
    specified schema.

    Args:
        raw_items: A list of condensed search result dictionaries (title, url, content).

    Returns:
        A formatted prompt string instructing the model to output JSON only.
    """
    return (
        "Normalize these search results into a JSON array of job objects.\n"
        "Return **ONLY** valid JSON. No prose.\n"
        "Array schema:\n"
        "[{\n"
        '  \"id\": \"job_001\",\n'
        '  \"title\": string,\n'
        '  \"company\": string,\n'
        '  \"url\": string,\n'
        '  \"summary\": string,\n'
        '  \"raw_excerpt\": string\n'
        "}]\n\n"
        f"RESULTS:\n{json.dumps(raw_items, ensure_ascii=False)}\n"
    )

@app.get("/gemini_health")
def gemini_health() -> Dict[str, Any]:
    """
    Health check to confirm Gemini is reachable and responding.

    Returns:
        JSON containing ok flag, model name, and a small echo of the response.
    """
    try:
        # Ask for plain text to avoid JSON parsing here.
        test_echo = gemini_model.generate_content(
            "ping",
            generation_config={"response_mime_type": "text/plain"}
        ).text
        return {"ok": True, "model": GEMINI_MODEL_NAME, "echo": (test_echo or "")[:20]}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/agent/normalize_resume", response_model=NormalizeResumeResponse)
def normalize_resume(request: NormalizeResumeRequest) -> NormalizeResumeResponse:
    """
    Normalize a raw resume into structured JSON using Gemini.

    Args:
        request: Contains the raw resume text.

    Returns:
        NormalizeResumeResponse containing a NormalizedResume object.
    """
    prompt = build_resume_prompt(request.resume_text)
    normalized_dict = gemini_json(prompt)
    # Pydantic validation
    normalized_resume = NormalizedResume(**normalized_dict)  
    return NormalizeResumeResponse(resume=normalized_resume)


@app.post("/agent/search_jobs", response_model=JobSearchResponse)
def search_jobs(request: JobSearchRequest) -> JobSearchResponse:
    """
    Search for jobs via Tavily and optionally summarize/normalize results via Gemini.

    Args:
        request: Contains the search query, filters, limit, and summarize flag.

    Returns:
        JobSearchResponse containing a list of JobRecord entries.
    """
    # Execute the search query using Tavily.
    tavily_raw_response = tavily_client.search(query=request.query, max_results=request.limit)
    tavily_results = tavily_raw_response.get("results", [])

    if not tavily_results:
        return JobSearchResponse(jobs=[])

    # If summarization/normalization is requested, use Gemini to condense to our schema.
    if request.summarize:
        condensed_items: List[Dict[str, Any]] = []
        for search_index, search_result in enumerate(tavily_results[:request.limit], start=1):
            condensed_items.append({
                "index": search_index,
                "title": search_result.get("title", "") or "",
                "url": search_result.get("url", "") or "",
                # Truncate for token safety
                "content": (search_result.get("content", "") or "")[:1200],
            })

        prompt = build_job_normalize_prompt(condensed_items)
        parsed_job_list = gemini_json(prompt)

        normalized_jobs: List[JobRecord] = []
        for job_position_index, parsed_row in enumerate(parsed_job_list, start=1):
            normalized_jobs.append(JobRecord(
                id=parsed_row.get("id") or f"job_{job_position_index:03d}",
                title=parsed_row.get("title", ""),
                company=parsed_row.get("company", ""),
                url=parsed_row.get("url", ""),
                # Trim long summaries
                summary=parsed_row.get("summary", "")[:600],   
                raw_excerpt=parsed_row.get("raw_excerpt")
            ))
        return JobSearchResponse(jobs=normalized_jobs)

    # If we are not summarizing, return a fallback of Tavily results.
    passthrough_jobs: List[JobRecord] = []
    for search_index, search_result in enumerate(tavily_results[:request.limit], start=1):
        passthrough_jobs.append(JobRecord(
            id=f"job_{search_index:03d}",
            title=search_result.get("title", "") or "Job",
             # Unknown when not summarized
            company="", 
            url=search_result.get("url", ""),
            summary=(search_result.get("content", "") or "")[:600],
            raw_excerpt=(search_result.get("content", "") or "")[:600],
        ))
    return JobSearchResponse(jobs=passthrough_jobs)


@app.post("/agent/score_jobs", response_model=JobScoringResponse)
async def score_jobs(request: JobScoringRequest) -> JobScoringResponse:
    """
    Forward resume sections and job records to the embedder service for scoring.

    Args:
        request: Contains resume_sections and jobs as JobRecord instances.

    Returns:
        JobScoringResponse with a list of similarity score dictionaries.

    Raises:
        HTTPException(502): If the embedder service returns an error.
    """
    # HTTP client to call the embedder service.
    async with httpx.AsyncClient(timeout=60) as async_client:
        try:
            embedder_response = await async_client.post(
                f"{EMBEDDER_BASE_URL}/score_jobs",
                json=request.dict()
            )
            embedder_response.raise_for_status()
            embedder_payload = embedder_response.json()
            return JobScoringResponse(**embedder_payload)
        except httpx.HTTPError as http_error:
            raise HTTPException(status_code=502, detail=f"Embedder error: {http_error}")


@app.post("/agent/analyze_resume_and_jobs", response_model=AnalyzeAllResponse)
def analyze_resume_and_jobs(request: AnalyzeAllRequest) -> AnalyzeAllResponse:
    """
    Orchestrated pipeline:
    Normalize resume via Gemini (JSON).
    Search and summarize jobs via Tavily + Gemini (JSON).
    Score jobs against resume sections via Embedder service.

    Args:
        request: Resume text, job query, and limit for number of jobs.

    Returns:
        AnalyzeAllResponse containing normalized resume, jobs, and scores.

    Raises:
        HTTPException(502): If the embedder scoring call fails.
    """
    # Normalize the user's resume into structured JSON.
    normalized_resume = _normalize_resume(request.resume_text)

    # Prepare compact, deduplicated resume sections for embedding/scoring.
    resume_sections = _prepare_resume_sections_for_embedding(normalized_resume)

    # Perform job search and summarization to our normalized job schema.
    normalized_jobs = _search_and_summarize_jobs(request.job_query, request.limit)

    # Submit to embedder for similarity scoring between resume and jobs.
    embedder_payload = {
        "resume_sections": resume_sections[:20],
        "jobs": [job.dict() for job in normalized_jobs],
    }

    try:
        with httpx.Client(timeout=60) as sync_client:
            embedder_response = sync_client.post(f"{EMBEDDER_BASE_URL}/score_jobs", json=embedder_payload)
            embedder_response.raise_for_status()
            scoring_payload = embedder_response.json()
            similarity_scores: List[Dict[str, Any]] = scoring_payload.get("scores", [])
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Embedder scoring failed: {error}")

    return AnalyzeAllResponse(
        resume=normalized_resume,
        jobs=normalized_jobs,
        scores=similarity_scores
    )


@app.post("/agent/chat", response_model=ChatResponse)
def agent_chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint with behaviors:
    - If the message looks like a job search, run the full orchestrator (requires resume_text).
    - If the message asks to normalize a resume, return normalized JSON (requires resume_text).
    - Otherwise, return brief usage guidance.

    Args:
        request: Contains a free-form message and optional resume_text.

    Returns:
        ChatResponse with a reply and optionally resume/jobs/scores.
    """
    user_message = request.message.strip()

    if _looks_like_job_search(user_message):
        if not request.resume_text:
            return ChatResponse(
                reply="I can search and rank roles, but I need your resume text. Please include `resume_text` in your request.",
            )

        # Treat the entire user message as the search query.
        orchestrator_input = AnalyzeAllRequest(
            resume_text=request.resume_text,
            job_query=user_message,
            limit=request.limit
        )

        # Call the orchestrator directly (without issuing an HTTP request).
        orchestrator_output = analyze_resume_and_jobs(orchestrator_input)

        # Produce a  human-friendly reply summarizing the result.
        top_match_reply = (
            f"Found {len(orchestrator_output.jobs)} jobs and ranked them against your resume. "
            f"Top match: {orchestrator_output.jobs[0].title} at "
            f"{orchestrator_output.jobs[0].company or 'unknown'}"
            if orchestrator_output.jobs else
            "No jobs found for that query."
        )

        return ChatResponse(
            reply=top_match_reply,
            resume=orchestrator_output.resume,
            jobs=orchestrator_output.jobs,
            scores=orchestrator_output.scores
        )

    # If the user is asking to normalize a resume explicitly.
    if "normalize" in user_message or "parse resume" in user_message:
        if not request.resume_text:
            return ChatResponse(
                reply="Please include `resume_text` so I can normalize it."
            )
        normalized_resume = _normalize_resume(request.resume_text)
        return ChatResponse(
            reply="Here’s your normalized resume JSON.",
            resume=normalized_resume
        )

    # Default message.
    return ChatResponse(
        reply=(
            "Hi! You can:\n"
            "• Ask me to find roles (e.g., “Find junior engineer roles in SF”).\n"
            "• Include `resume_text` so I can rank results against your resume.\n"
            "• Say “normalize” with `resume_text` to get structured JSON."
        )
    )


def _normalize_resume(resume_text: str) -> NormalizedResume:
    """
    Call Gemini to normalize a resume to our NormalizedResume schema.

    Args:
        resume_text: Raw resume text to normalize.

    Returns:
        NormalizedResume instance produced by LLM and validated by Pydantic.
    """
    prompt = build_resume_prompt(resume_text)
    normalized_payload = gemini_json(prompt)
    return NormalizedResume(**normalized_payload)


def _search_and_summarize_jobs(query: str, limit: int) -> List[JobRecord]:
    """
    Helper: Search jobs via Tavily, then summarize/normalize via Gemini into JobRecord entries.

    Steps:
    - Query Tavily.
    - Condense the results (title/url/content) and truncate content for token budget.
    - Ask Gemini to normalize results to our job schema.
    - Return a list of JobRecord objects.

    Args:
        query: The job search query (e.g., "junior frontend engineer San Francisco").
        limit: Maximum number of results to fetch and normalize (1..25).

    Returns:
        A list of JobRecord instances.
    """
    # Fetch raw results from Tavily.
    tavily_raw_response = tavily_client.search(query=query, max_results=limit)
    tavily_results = tavily_raw_response.get("results", [])

    # Condense each raw result to the fields we care about; truncate content for LLM safety.
    condensed_items: List[Dict[str, Any]] = []
    for result_index, result_item in enumerate(tavily_results[:limit], start=1):
        condensed_items.append({
            "index": result_index,
            "title": result_item.get("title", "") or "",
            "url": result_item.get("url", "") or "",
            "content": (result_item.get("content", "") or "")[:1200],
        })

    # Ask Gemini to normalize into our JobRecord-like schema.
    prompt = build_job_normalize_prompt(condensed_items)
    parsed_job_list = gemini_json(prompt)

    # Convert into JobRecord models.
    normalized_jobs: List[JobRecord] = []
    for job_position_index, parsed_row in enumerate(parsed_job_list, start=1):
        normalized_jobs.append(JobRecord(
            id=parsed_row.get("id") or f"job_{job_position_index:03d}",
            title=parsed_row.get("title", ""),
            company=parsed_row.get("company", ""),
            url=parsed_row.get("url", ""),
            summary=parsed_row.get("summary", "")[:600],
            raw_excerpt=parsed_row.get("raw_excerpt"),
        ))
    return normalized_jobs


def _prepare_resume_sections_for_embedding(
    resume: NormalizedResume,
    max_sections: int = 20,
    max_chars_per_section: int = 280
) -> List[str]:
    """
    Produce compact, deduplicated text sections from a normalized resume
    to improve embedding quality and latency.

    Strategy:
      - Include summary (if present).
      - Include a single "Skills: ..." line (dedup skills and keep order).
      - Include top bullets from work experience, trimmed for length.
      - Optionally include a compact education line.
      - Deduplicate identical/near-identical sections case/space-insensitively.
      - Limit both number of sections and per-section length.

    Args:
        resume: The normalized resume object.
        max_sections: Maximum number of sections to return (default: 20).
        max_chars_per_section: Maximum characters per section (default: 280).

    Returns:
        A list of short, deduplicated text sections suitable for embedding.
    """
    prepared_sections: List[str] = []

    def shorten(text_value: str) -> str:
        """Normalize whitespace and clamp to max_chars_per_section."""
        collapsed = re.sub(r"\s+", " ", text_value).strip()
        return collapsed[:max_chars_per_section]

    # 1) Summary line
    if resume.summary:
        prepared_sections.append(shorten(resume.summary))

    # 2) Skills line (unique + stable order)
    if resume.skills:
        unique_skills_in_order = list(OrderedDict.fromkeys(
            [skill.strip() for skill in resume.skills if skill.strip()]
        ))
        if unique_skills_in_order:
            prepared_sections.append(shorten("Skills: " + ", ".join(unique_skills_in_order)))

    # 3) Top bullets from experience (prefer concise, action-oriented statements)
    for experience_record in resume.experience or []:
        bullet_list: Iterable[str] = experience_record.get("bullets") or []
        for bullet in bullet_list:
            trimmed_bullet = bullet.strip()
            if not trimmed_bullet:
                continue
            prepared_sections.append(shorten(trimmed_bullet))
            if len(prepared_sections) >= max_sections:
                break
        if len(prepared_sections) >= max_sections:
            break

    # 4) Education (compact)
    education_pieces: List[str] = []
    for education_entry in (resume.education or [])[:2]:  # Only include up to two entries
        school_name = (education_entry.get("school") or "").strip()
        degree_name = (education_entry.get("degree") or "").strip()
        graduation_year = (education_entry.get("year") or "").strip()
        compact_line = " ".join(part for part in [school_name, degree_name, graduation_year] if part)
        if compact_line:
            education_pieces.append(compact_line)

    if education_pieces and len(prepared_sections) < max_sections:
        prepared_sections.append(shorten("Education: " + " | ".join(education_pieces)))

    # Deduplicate sections (case/space-insensitive).
    seen_keys: set = set()
    deduplicated_sections: List[str] = []
    for section_text in prepared_sections:
        normalized_key = re.sub(r"\s+", " ", section_text).strip().lower()
        if normalized_key in seen_keys:
            continue
        seen_keys.add(normalized_key)
        deduplicated_sections.append(section_text)

    # Return the capped list of sections.
    return deduplicated_sections[:max_sections]


def _looks_like_job_search(text: str) -> bool:
    """
    Determine if a free-text message likely represents a job search intent.

    Args:
        text: User provided message string.

    Returns:
        True if the message contains common job-search keywords; otherwise False.
    """
    lowercased_text = (text or "").lower()
    job_intent_keywords = [
        "job", "role", "opening", "hiring", "find", "search",
        "positions", "junior", "engineer"
    ]
    return any(keyword in lowercased_text for keyword in job_intent_keywords)
