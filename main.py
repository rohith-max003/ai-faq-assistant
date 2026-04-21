"""FastAPI entrypoint for the AI FAQ Assistant service."""

import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from faq_assistant import FAQAssistant
from config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()
assistant = FAQAssistant()

SAMPLE_FAQS = [
    {"id": "1", "category": "benefits", "content": "To renew your SNAP benefits, log into your MyBenefits account and click 'Renew Benefits'. You'll need proof of income and residency documents. Renewal must be completed 30 days before your current certification ends."},
    {"id": "2", "category": "medicaid", "content": "Medicaid eligibility is based on income level. For a family of 4, the income limit is 138% of the Federal Poverty Level. You can apply online, by mail, or in person at your local DHS office."},
    {"id": "3", "category": "childcare", "content": "The Child Care Assistance Program (CCAP) provides subsidies for families earning below 85% of State Median Income. Applications are processed within 30 business days."},
    {"id": "4", "category": "appeals", "content": "If your benefits were denied or reduced, you have the right to appeal within 90 days. Submit your appeal in writing to the Appeals Division or request an in-person hearing."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading FAQ knowledge base...")
    assistant.load_knowledge_base(SAMPLE_FAQS)
    logger.info("FAQ Assistant ready")
    yield
    logger.info("Shutting down FAQ Assistant")


app = FastAPI(
    title="AI FAQ Assistant",
    description="LangChain + Azure OpenAI powered citizen support assistant",
    version="1.0.0",
    lifespan=lifespan
)


class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str
    response_time_ms: float


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Answer a citizen support question using AI-powered FAQ retrieval."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start = time.time()
    result = assistant.ask(request.question, request.session_id)
    elapsed_ms = round((time.time() - start) * 1000, 2)

    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=result["session_id"],
        response_time_ms=elapsed_ms
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ai-faq-assistant", "model": settings.azure_openai_deployment_name}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=settings.app_env == "development")
