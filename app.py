from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI()

# Allow CORS for frontend (change to specific origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load toxicity classification pipeline
classifier = pipeline("text-classification", model="unitary/toxic-bert")

# Define request body schema
class ScanInput(BaseModel):
    text: str

# Define route for AI text scanning
@app.post("/api/text-scan")
async def scan_text(data: ScanInput):
    result = classifier(data.text)[0]
    label, score = result["label"], result["score"]

    flagged = label.lower() == "toxic" and score > 0.6

    return {
        "flagged": flagged,
        "label": label,
        "score": round(score, 3),
        "message": "⚠️ Unsafe content detected" if flagged else "✅ Content is safe"
    }

