from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime
from typing import List, Optional
from pathlib import Path
import requests
from markitdown import MarkItDown
import hashlib
import trafilatura
import pymupdf4llm
import traceback
import re
import os
from google import genai

app = FastAPI()
GEMINI_API_KEY=""

# CORS configuration
origins = ["*"]  # Allows all origins.  Modify for production.

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
dimension = 384  # Dimension of the embeddings
#ROOT = None
#index = None
#metadata = []
#CACHE_META = {}
#CACHE_FILE = None
#METADATA_FILE = None
#INDEX_FILE = None

OLLAMA_CHAT_URL = "http://localhost:11434/api/generate"
PHI_MODEL = "phi4"

ROOT = Path(__file__).parent.resolve()
    #DOC_PATH = ROOT / "documents"
INDEX_CACHE = ROOT / "faiss_index"
INDEX_CACHE.mkdir(exist_ok=True)
INDEX_FILE = INDEX_CACHE / "index.bin"
METADATA_FILE = INDEX_CACHE / "metadata.json"
    #CACHE_FILE = INDEX_CACHE / "doc_index_cache.json"

    #CACHE_META = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
metadata = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else []
index = faiss.read_index(str(INDEX_FILE)) if INDEX_FILE.exists() else None


#def file_hash(path):
        #return hashlib.md5(Path(path).read_bytes()).hexdigest()

def get_embedding(string):
    return model.encode(string)

async def generate_text(prompt: str) -> str:
    return gemini_generate(prompt)


def gemini_generate(prompt: str) -> str:
    #api_key = os.getenv("GEMINI_API_KEY")
    #client = genai.Client(api_key=api_key)
    client = genai.Client(api_key=GEMINI_API_KEY)
			
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    # ✅ Safely extract response text
    try:
        return response.text.strip()
    except AttributeError:
        try:
            return response.candidates[0].content.parts[0].text.strip()
        except Exception:
            return str(response)



def init_db():
    """Process documents and create FAISS index using unified multimodal strategy."""
    ROOT = Path(__file__).parent.resolve()
    #DOC_PATH = ROOT / "documents"
    INDEX_CACHE = ROOT / "faiss_index"
    INDEX_CACHE.mkdir(exist_ok=True)
    INDEX_FILE = INDEX_CACHE / "index.bin"
    METADATA_FILE = INDEX_CACHE / "metadata.json"
    #CACHE_FILE = INDEX_CACHE / "doc_index_cache.json"

    #CACHE_META = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    metadata = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else []
    index = faiss.read_index(str(INDEX_FILE)) if INDEX_FILE.exists() else None


class ProcessPageRequest(BaseModel):
    url: str
    title: str
    content: str

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5

class SearchResult(BaseModel):
    url: str
    title: str
    content: str
    score: float

class Result(BaseModel):
    title: str
    summary: str
    
@app.on_event("startup")
async def startup_event():
    init_db()


def semantic_merge(text: str) -> list[str]:
    """Splits text semantically using LLM: detects second topic and reuses leftover intelligently."""
    WORD_LIMIT = 512
    words = text.split()
    i = 0
    final_chunks = []

    while i < len(words):
        # 1. Take next chunk of words (and prepend leftovers if any)
        chunk_words = words[i:i + WORD_LIMIT]
        chunk_text = " ".join(chunk_words).strip()

        prompt = f"""
You are a markdown document segmenter.

Here is a portion of a markdown document:

---
{chunk_text}
---

If this chunk clearly contains **more than one distinct topic or section**, reply ONLY with the **second part**, starting from the first sentence or heading of the new topic.

If it's only one topic, reply with NOTHING.

Keep markdown formatting intact.
"""

        try:
            response = requests.post(OLLAMA_CHAT_URL, json={
                "model": PHI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            })
            reply = response.json().get("message", {}).get("content", "").strip()

            if reply:
                # If LLM returned second part, separate it
                split_point = chunk_text.find(reply)
                if split_point != -1:
                    first_part = chunk_text[:split_point].strip()
                    second_part = reply.strip()

                    final_chunks.append(first_part)

                    # Get remaining words from second_part and re-use them in next batch
                    leftover_words = second_part.split()
                    words = leftover_words + words[i + WORD_LIMIT:]
                    i = 0  # restart loop with leftover + remaining
                    continue
                else:
                    # fallback: if split point not found
                    final_chunks.append(chunk_text)
            else:
                final_chunks.append(chunk_text)

        except Exception as e:
            #mcp_log("ERROR", f"Semantic chunking LLM error: {e}")
            final_chunks.append(chunk_text)

        i += WORD_LIMIT

    return final_chunks


def extract_webpage(url:str):
    """Extract and convert webpage content to markdown. Usage: extract_webpage|input={"url": "https://example.com"}"""

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return "Failed to download the webpage."

    markdown = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        include_images=True,
        output_format='markdown'
    ) or ""

    return markdown

@app.post("/process")
async def process_page(data: ProcessPageRequest):
    global index
    try:
        # Generate embedding for the content
        embedding = model.encode(data.content)

        # Add to FAISS index with the correct index
        #if row_id is not None:
            #index.add(embedding.reshape(1, -1))

        url = data.url

        #fhash = file_hash(url)
        #if url in CACHE_META and CACHE_META[url] == fhash:
        #    return {"status": "success"}
        

        markdown = extract_webpage(url)
        #markdown = data.content

        if len(markdown.split()) < 10:               
            chunks = [markdown.strip()]
        else:
            chunks = semantic_merge(markdown)
        #    mcp_log("INFO", f"Running semantic merge on {file.name} with {len(markdown.split())} words")
        
        
        embeddings_for_file = []
        new_metadata = []
        #for i, chunk in enumerate(tqdm(chunks, desc=f"Embedding {file.name}")):
        #embedding = get_embedding(chunk)
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(markdown)
            embeddings_for_file.append(embedding)
            new_metadata.append({
                "url": url,
                "doc": chunk,
                "title": data.title,
                "chunk_id": f"{url}_{i}"
            })

        if embeddings_for_file:
            if index is None:
                dim = len(embeddings_for_file[0])
                index = faiss.IndexFlatL2(dim)
            index.add(np.stack(embeddings_for_file))
            metadata.extend(new_metadata)
            #CACHE_META[url] = fhash

            # ✅ Immediately save index and metadata
            #CACHE_FILE.write_text(json.dumps(CACHE_META, indent=2))
            METADATA_FILE.write_text(json.dumps(metadata, indent=2))
            faiss.write_index(index, str(INDEX_FILE))
                
        return {"status": "success"}

    except Exception as e:
        print(f"Processing error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    

async def extract_data(query:str, title:str, user_input: str) ->dict:
    """
    Uses LLMs to extract structured info:
    - title: Title
    - summary: Summary
    """

    prompt = f"""
You are an AI expert who is very good in generating title and summarizing content

Query: {query}
Input: "{user_input}"

Return the response as a Python dictionary with keys:
- title: Title from the given Input
- summary: Nice Summary of the Input in at most 200 words. If Input has no relation with Query given then generate summary with single word response as: JUNK. Also if the response contains unrelated multiple topics generate summary with single word response as: JUNK

Output only the dictionary on a single line. Do NOT wrap it in ```json or other formatting. 
"""

    try:
        response = await generate_text(prompt)

        # Clean up raw if wrapped in markdown-style ```json
        raw = response.strip()
        if not raw or raw.lower() in ["none", "null", "undefined"]:
            raise ValueError("Empty or null model output")

        # Clean and parse
        clean = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        import json

        try:
            parsed = json.loads(clean.replace("null", "null"))  # Clean up non-Python nulls
        except Exception as json_error:
            print(f"[perception] JSON parsing failed: {json_error}")
            parsed = {}

        # Ensure Keys
        if not isinstance(parsed, dict):
            raise ValueError("Parsed LLM output is not a dict")
        if "title" not in parsed:
            parsed["title"] = title
        if "summary" not in parsed:
            parsed["summary"] = user_input
        #return Result(**parsed)
        return parsed

    except Exception as e:
        print(f"[perception] ⚠️ LLM perception failed: {e}")
        parsed = {}
        parsed["title"] = title
        parsed["summary"] = user_input
        return parsed
        #return Result(title=None, summary="JUNK")

@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    global index
    try:
        # Generate query embedding
        query_embedding = get_embedding(request.query)

        index = faiss.read_index(str(INDEX_FILE))
        metadata = json.loads(METADATA_FILE.read_text())
        query_vec = query_embedding.reshape(1, -1)
        D, I = index.search(query_vec, k=3)
        results = []
        for i, idx in  enumerate(I[0]):
            data = metadata[idx]
            res = await extract_data(request.query, data['title'], data['doc'])
            print("res = " + str(res))
            if res["summary"] == "JUNK":
                continue
            results.append(SearchResult(
                    url=data['url'],
                    #title=data['title'],
                    #title = res.title,
                    title = res["title"],
                    #content=data['doc'][:200] + "...",  # Truncate content for preview

                    #content=data['doc'] + "...",
                    #content = res.summary + '...',
                    content = res["summary"] + '...',
                    score=float(D[0][i])  # Convert numpy.float32 to Python float
                    ))
           # results.append(f"{data['chunk']}\n[Source: {data['doc']}, ID: {data['chunk_id']}]")
        
        print("result=" + str(results) )
        return results

    except Exception as e:
        print(f"Search error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    init_db()
    #  Uvicorn is the recommended way to run FastAPI in production
    #  However, you can't directly run uvicorn from the script like Flask's app.run()
    #  Instead, you would typically use the command line:
    #  uvicorn app:app --reload
    #  This line is just a reminder and won't actually run anything
    print("To run the app, use: uvicorn app:app --reload")