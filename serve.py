# serve.py
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from ask import ConfluenceQA
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load QA system once and keep it warm
logger.info("Loading ConfluenceQA system...")
qa = ConfluenceQA()
logger.info("ConfluenceQA system loaded successfully!")

app = FastAPI(title="Confluence QA API", version="1.0.0")

class Q(BaseModel):
    question: str
    top_k: int = 5

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "confluence-qa"}

@app.post("/ask")
def ask(q: Q):
    logger.info(f"Received question: {q.question}")
    try:
        answer = qa.answer_question(q.question)
        logger.info("Question answered successfully")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return {"answer": f"Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)
