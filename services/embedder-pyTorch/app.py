from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

app = FastAPI()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class Payload(BaseModel):
    resume_sections: list[str]
    job_description: str
    skills_taxonomy: list[str] = []

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

@app.post("/analyze")
def analyze(payload: Payload):
    
    job_description_vectors = model.encode([payload.job_description])[0]
    resume_sections_vectors = model.encode(payload.resume_sections)
    section_scores = [cosine(vector, job_description_vectors) for vector in resume_sections_vectors]

    missing_skills = []
    if payload.skills_taxonomy:
        skill_vectors = model.encode(payload.skills_taxonomy)
        sims = [cosine(skill_vector, job_description_vectors) for skill_vector in skill_vectors]

        ranked_skills = [s for _, s in sorted(zip(sims, payload.skills_taxonomy), reverse=True)]
        missing_skills = ranked_skills[:10]

    return {"section_scores": section_scores, "missing_skills_guess": missing_skills}
