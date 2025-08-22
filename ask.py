"""Question answering CLI for Confluence-QA Bot."""

import argparse
import logging
import sys
from typing import Any
import numpy as np

import openai
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

from config import config

# Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# Disable all logging
import logging
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("sentence_transformers").setLevel(logging.CRITICAL)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


class ConfluenceQA:
    """Question answering system for Confluence content."""

    def __init__(self):
        """Initialize the QA system."""
        # Initialize MongoDB client
        self.mongo_client = MongoClient(config.MONGODB_URI)
        self.database = self.mongo_client[config.MONGODB_DATABASE]
        self.collection = self.database[config.MONGODB_COLLECTION]

        # Initialize embedding model
        # logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

        # Initialize DeepSeek client if API key is available
        self.deepseek_client = None
        if config.DEEPSEEK_API_KEY and config.DEEPSEEK_API_KEY.strip():
            self.deepseek_client = openai.OpenAI(
                api_key=config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"
            )
            # logger.info("DeepSeek API client initialized")
        else:
            pass
            # logger.info("No DeepSeek API key found, will return raw chunks")

    def search_similar_chunks(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search for similar chunks using vector similarity."""
        # Create embedding for the question
        question_embedding = self.embedding_model.encode([question])[0].tolist()

        # Get all documents with embeddings from MongoDB
        documents = list(self.collection.find({"embedding": {"$exists": True}}))
        
        if not documents:
            return []

        # Calculate cosine similarity for each document
        similarities = []
        for doc in documents:
            doc_embedding = np.array(doc['embedding'])
            question_emb = np.array(question_embedding)
            
            # Cosine similarity
            similarity = np.dot(question_emb, doc_embedding) / (
                np.linalg.norm(question_emb) * np.linalg.norm(doc_embedding)
            )
            similarities.append((similarity, doc))

        # Sort by similarity and take top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = similarities[:top_k]

        # Convert to result format
        results = []
        for similarity, doc in top_results:
            result = {
                "score": similarity,
                "page_id": doc["page_id"],
                "page_title": doc["page_title"],
                "chunk_idx": doc["chunk_idx"],
                "text": doc["text"],
                "confluence_url": doc.get("confluence_url", ""),
            }
            results.append(result)

        return results

    def format_context(self, chunks: list[dict[str, Any]]) -> str:
        """Format search results into context string."""
        context_parts = []

        for chunk in chunks:
            part = f"Page: {chunk['page_title']}\n"
            part += f"URL: {chunk['confluence_url']}\n"
            part += f"Content: {chunk['text']}\n"
            part += f"Relevance: {chunk['score']:.3f}\n"
            context_parts.append(part)

        return "\n" + "=" * 50 + "\n".join(context_parts)

    def generate_answer_with_deepseek(self, question: str, context: str) -> str:
        """Generate answer using DeepSeek API."""
        system_prompt = """You are a technical assistant for a Slack channel. Provide concise, direct answers.

Rules:
- Be brief and to the point
- Use bullet points for multiple items
- Include specific values (IPs, URLs, etc.) prominently
- No markdown headers or excessive formatting
- Maximum 3-4 sentences unless listing steps
- Cite the source page briefly

Context: {context}
Question: {question}"""

        try:
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt.format(context=context, question=question),
                    }
                ],
                max_tokens=500,
                temperature=0.1,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            return f"Error generating answer: {e}"

    def answer_question(self, question: str) -> str:
        """Answer a question using the knowledge base."""
        # logger.info(f"Answering question: {question}")

        # Search for relevant chunks
        chunks = self.search_similar_chunks(question, top_k=5)

        if not chunks:
            return "No relevant information found in the knowledge base."

        # Format context
        context = self.format_context(chunks)

        # Generate answer
        if self.deepseek_client:
            # Use DeepSeek for intelligent answering
            answer = self.generate_answer_with_deepseek(question, context)
            return answer
        else:
            # Return concise answer without DeepSeek
            first_chunk = chunks[0]
            answer = f"📄 {first_chunk['text'][:200]}{'...' if len(first_chunk['text']) > 200 else ''}\n\n"
            answer += f"Source: {first_chunk['page_title']}\n{first_chunk['confluence_url']}"

            if len(chunks) > 1:
                answer += f"\n\n(Found {len(chunks)} relevant sections - consider adding DEEPSEEK_API_KEY for better answers)"

            return answer


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Ask questions about Confluence content")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        qa_system = ConfluenceQA()
        answer = qa_system.answer_question(args.question)
        print(answer)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
