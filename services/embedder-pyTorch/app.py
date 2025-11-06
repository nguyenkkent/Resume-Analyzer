from pydantic import BaseModel
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI

app = FastAPI()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

class Job(BaseModel):
    id: str
    title: str = ""
    company: str = ""
    url: str = ""
    summary: str

class ScoreJobsReq(BaseModel):
    resume_sections: List[str]
    jobs: List[Job]

@app.get("/health")
def health():
    return {"status": "ok"}
    
@app.post("/score_jobs")
def score_jobs(req: ScoreJobsReq):
    resume_vectors = model.encode(req.resume_sections)
    scores = []
    for job in req.jobs:
        job_vector = model.encode([job.summary])[0]
        sim = max(cosine(resume_vector, job_vector) for resume_vector in resume_vectors) 
        scores.append({"job_id": job.id, "similarity": sim})
    scores.sort(key=lambda x: x["similarity"], reverse=True)
    return {"scores": scores}
