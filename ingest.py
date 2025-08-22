"""Confluence page ingestion and indexing pipeline."""

import asyncio
import base64
import logging
import re
import hashlib
from datetime import datetime
from typing import Any

import html2text
import httpx
import tiktoken
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfluenceIngester:
    """Handles Confluence page ingestion and vector indexing."""

    def __init__(self):
        """Initialize the ingester with required clients and models."""
        config.validate()

        # Initialize MongoDB client
        self.mongo_client = MongoClient(config.MONGODB_URI)
        self.database = self.mongo_client[config.MONGODB_DATABASE]
        self.collection = self.database[config.MONGODB_COLLECTION]

        # Initialize embedding model
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

        # Initialize tokenizer for chunk splitting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Initialize HTML to Markdown converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = False

        # Setup HTTP client with auth
        auth_string = f"{config.CONFLUENCE_EMAIL}:{config.CONFLUENCE_API_KEY}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()

        self.http_client = httpx.AsyncClient(
            headers={
                "Authorization": f"Basic {encoded_auth}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.http_client.aclose()
        self.mongo_client.close()

    def setup_collection(self) -> None:
        """Setup MongoDB collection and ensure indexes exist."""
        # Create text index for text search
        try:
            self.collection.create_index([("text", "text")])
            logger.info(f"Created text index on collection: {config.MONGODB_COLLECTION}")
        except Exception as e:
            logger.info(f"Text index already exists or error creating: {e}")

        # Create index on page_id for efficient filtering
        try:
            self.collection.create_index("page_id")
            logger.info(f"Created page_id index on collection: {config.MONGODB_COLLECTION}")
        except Exception as e:
            logger.info(f"Page_id index already exists or error creating: {e}")

    async def fetch_page(self, page_id: str) -> dict[str, Any]:
        """Fetch a single Confluence page via REST API."""
        url = f"{config.CONFLUENCE_BASE}/rest/api/content/{page_id}"
        params = {"expand": "body.storage,title,version"}

        logger.info(f"Fetching page {page_id}")
        response = await self.http_client.get(url, params=params)
        response.raise_for_status()

        return response.json()

    def html_to_markdown(self, html_content: str) -> str:
        """Convert HTML content to clean Markdown."""
        # Clean up some common Confluence HTML artifacts
        html_content = re.sub(r"<ac:.*?</ac:.*?>", "", html_content, flags=re.DOTALL)
        html_content = re.sub(r"<ac:.*?/>", "", html_content)

        # Convert to markdown
        markdown = self.html_converter.handle(html_content)

        # Clean up extra whitespace
        markdown = re.sub(r"\n\s*\n\s*\n", "\n\n", markdown)
        markdown = markdown.strip()

        return markdown

    def chunk_text(self, text: str) -> list[str]:
        """Split text into chunks of approximately CHUNK_SIZE tokens."""
        # Encode the text to tokens
        tokens = self.tokenizer.encode(text)

        chunks = []
        start = 0

        while start < len(tokens):
            # Calculate end position
            end = start + config.CHUNK_SIZE

            # If this is not the last chunk, try to find a good breaking point
            if end < len(tokens):
                # Look for sentence boundaries within the overlap region
                search_start = max(start + config.CHUNK_SIZE - config.CHUNK_OVERLAP, start)
                search_end = min(end + config.CHUNK_OVERLAP, len(tokens))

                # Find the best breaking point (period, newline, etc.)
                best_break = end
                chunk_text = self.tokenizer.decode(tokens[search_start:search_end])

                # Look for sentence endings
                for pattern in [". ", ".\n", ".\r", ". ", "?\n", "!\n"]:
                    matches = list(re.finditer(re.escape(pattern), chunk_text))
                    if matches:
                        # Use the last occurrence
                        last_match = matches[-1]
                        relative_pos = last_match.end()
                        # Convert back to token position
                        prefix = chunk_text[:relative_pos]
                        prefix_tokens = len(self.tokenizer.encode(prefix))
                        best_break = search_start + prefix_tokens
                        break

                end = best_break

            # Extract chunk
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Skip very short chunks
            if len(chunk_text.strip()) > 10:
                chunks.append(chunk_text.strip())

            # Move to next chunk with overlap
            start = end - config.CHUNK_OVERLAP if end < len(tokens) else end

            # Prevent infinite loop
            if start >= len(tokens):
                break

        return chunks

    def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for a list of texts."""
        logger.info(f"Creating embeddings for {len(texts)} chunks")
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()

    async def ingest_page(self, page_id: str) -> int:
        """Ingest a single page and return number of chunks created."""
        try:
            # Fetch page data
            page_data = await self.fetch_page(page_id)

            # Extract content
            title = page_data.get("title", "")
            html_content = page_data.get("body", {}).get("storage", {}).get("value", "")
            updated_at = page_data.get("version", {}).get("when", datetime.now().isoformat())

            if not html_content:
                logger.warning(f"No content found for page {page_id}")
                return 0

            # Convert to markdown and chunk
            markdown_content = self.html_to_markdown(html_content)
            chunks = self.chunk_text(markdown_content)

            if not chunks:
                logger.warning(f"No chunks created for page {page_id}")
                return 0

            logger.info(f"Created {len(chunks)} chunks for page '{title}'")

            # Create embeddings
            embeddings = self.create_embeddings(chunks)

            # Prepare documents for MongoDB
            documents = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
                # Generate consistent document ID from page_id and chunk index
                doc_id_str = f"{page_id}_{idx}"
                doc_id = hashlib.md5(doc_id_str.encode()).hexdigest()
                
                document = {
                    "_id": doc_id,
                    "page_id": page_id,
                    "page_title": title,
                    "chunk_idx": idx,
                    "text": chunk,
                    "embedding": embedding,
                    "updated_at": updated_at,
                    "confluence_url": config.get_confluence_url(page_id),
                }
                documents.append(document)

            # Upsert to MongoDB (replace existing documents)
            for document in documents:
                self.collection.replace_one(
                    {"_id": document["_id"]}, 
                    document, 
                    upsert=True
                )

            logger.info(f"Successfully indexed {len(documents)} chunks for page {page_id}")
            return len(documents)

        except Exception as e:
            logger.error(f"Failed to ingest page {page_id}: {e}")
            raise

    async def ingest_all_pages(self) -> tuple[int, int]:
        """Ingest all configured pages."""
        total_pages = 0
        total_chunks = 0

        # Setup collection first
        self.setup_collection()

        for page_id in config.PAGE_IDS:
            page_id = page_id.strip()
            if not page_id:
                continue

            try:
                chunk_count = await self.ingest_page(page_id)
                total_pages += 1
                total_chunks += chunk_count
            except Exception as e:
                logger.error(f"Failed to process page {page_id}: {e}")
                continue

        return total_pages, total_chunks


async def main():
    """Main ingestion function."""
    async with ConfluenceIngester() as ingester:
        total_pages, total_chunks = await ingester.ingest_all_pages()
        logger.info(f"Ingestion complete: {total_pages} pages, {total_chunks} chunks")


if __name__ == "__main__":
    asyncio.run(main())
