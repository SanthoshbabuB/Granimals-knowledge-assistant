                                                       Granimals Knowledge Assistant 

OVERVIEW : Granimals Knowledge Assistant is a multimodal AI-powered document question-answering system built using FastAPI, Streamlit, Qdrant, Ollama, BM25, LangChain, PyMuPDF, OCR, and RRF hybrid search. It allows users to upload multiple PDF documents, extracts and processes text, tables and visual content, converts document chunks into embeddings, stores them in Qdrant, retrieves relevant information using hybrid search, reranks the results, and generates grounded answers using a local LLM with document/page citations.

                                                               TECH TOOLS
                            [NOTE: My personel lap is below 8GB RAM, so I used ollama for local run and very small model ]
FastAPI – Backend REST API
Streamlit – Interactive chat UI and multiple PDF upload
PyMuPDF – PDF text extraction and document processing
LangChain Recursive Text Splitter – Text chunking
Ollama – Local LLM inference and embeddings
nomic-embed-text – Local text embedding model
Qdrant – Vector database and semantic similarity search
BM25 – Keyword-based sparse retrieval
RRF (Reciprocal Rank Fusion) – Combines dense and sparse search results
Reranking – Improves relevance of retrieved chunks
OCR / Tesseract – Extracts text from scanned/image-based content
Page Rendering – Converts PDF pages into images for visual processing
Table Extraction – Processes tables separately from normal text
Conversation Memory – Maintains conversational context
Structured JSON Logging – Application observability
Pytest – Testing
Docker / Docker Compose – Containerization
Evaluation Framework – Precision, Recall, F1 and groundedness evaluation

WORKFLOW SUMMARY :
                                                               TEXT LOADER
                                        
DOCUMENT UPLOAD
The user can upload one or multiple PDF documents through the Streamlit UI.
Uploaded documents are stored under the data/documents directory.
Each document is processed independently, making the system scalable for multiple PDFs.
Every piece of extracted information maintains document and page-level metadata for source traceability.

                                                              EXTRACTION LAYER
                                         
PDF DOCUMENT PROCESSING
I used PyMuPDF for PDF processing.
Text is extracted page by page rather than treating the entire PDF as one large document.
Page-level information is preserved so that the final answer can provide citations

MULTIMODAL PROCESSING
The knowledge base is not limited to plain text.
PDF documents can contain text, tables, images, charts, graphs and scanned pages.
I created separate processing paths for these different content types.

TEXT CLEANING
Extracted PDF text can contain unnecessary spaces, line breaks and extraction artifacts.
The cleaning step normalizes the extracted content before it is passed to the chunking stage.
This improves the quality of the chunks and therefore the retrieval results.

RECURSIVE TEXT SPLITTER
Large documents cannot be directly passed to the LLM.
Therefore, extracted text is divided into smaller chunks using LangChain RecursiveCharacterTextSplitter.
The splitter attempts to preserve meaningful text boundaries instead of arbitrarily cutting content.
Each chunk also maintains metadata such as document name, page number, content type and chunk ID

TABLE EXTRACTION
Tables are handled separately from normal paragraphs.
This is important for annual reports and other documents containing financial or structured data.
Table content is converted into a searchable representation while preserving its original document and page information.

IMAGE EXTRACTION
Embedded images are extracted from PDF documents and stored as separate assets.
Each extracted image maintains metadata such as:image_id
document_id
document_name
page_number
image_path
image_type
caption
OCR text
bounding box

OCR
Some PDFs contain scanned pages or text inside images.
I used OCR/Tesseract to extract searchable text from image-based content.

PAGE RENDERING
Some visual information such as charts, diagrams and complex page layouts cannot be completely represented by text extraction.
Therefore, PDF pages can be rendered into image files.
These page images are associated with the original document and page number.

                                                                 EMBEDDING LAYER
OLLAMA EMBEDDING
After the text is chunked, each chunk is converted into a vector embedding.
I used the local Ollama model:   "nomic-embed-text:latest"  

                                                                  VECTORDATABASE
QDRANT VECTOR DATABASE
I used Qdrant as the vector database.
The generated embeddings are stored along with document metadata.
Qdrant performs semantic similarity search between the user query and stored document chunks.
Stored information includes:Vector
Document
Page
Content Type
Chunk ID
Image Path

                                                                    RETRIEVER
HYBRID SEARCH
Instead of relying only on vector search, the system combines:

Dense Semantic Search
          +
BM25 Keyword Search
          ↓
     RRF Fusion
     
Dense retrieval helps identify semantically related content.
BM25 helps identify exact keyword matches.
Combining both improves retrieval coverage.

RRF — RECIPROCAL RANK FUSION
Dense and BM25 retrieval produce separate ranked result lists.
I used Reciprocal Rank Fusion (RRF) to combine those rankings into one result set.

                                                                   RERANKING
RERANKING
After hybrid retrieval, the candidate chunks are reranked before being passed to the LLM.
The current implementation uses a lightweight deterministic reranking approach.
This reduces the amount of irrelevant information provided to the generation model.
Dense Search
     +
BM25 Search
     ↓
RRF
     ↓
Reranking
     ↓
Top Relevant Chunks

QUERY EXPANSION
Query processing can improve retrieval by expanding or transforming the user's question before searching.
This helps the retrieval pipeline handle variations in how users phrase questions.

CONFIDENCE CHECK
Before generating an answer, the system evaluates the quality of the retrieved results.
A confidence score is calculated from the retrieval results.
If the system does not have sufficient evidence, it avoids generating an unsupported answer.

                                                                  TEXT GENERATION
OLLAMA LLM
Retrieved document context is passed to a local Ollama LLM for answer generation.
I used: "qwen2.5:0.5b"
The model was selected because it can run locally without requiring a paid API key.
The LLM is instructed to answer only from the retrieved source context.

                                                                      PROMPT
GROUNDED GENERATION

The prompt instructs the LLM to:

Use only retrieved document context
Not use outside knowledge
Not guess
Not invent information
Answer directly
Keep the answer concise
Include citations
Return a safe response when information is unavailable

This helps reduce hallucination.

CITATIONS
Every retrieved chunk contains document and page metadata.
The final answer can therefore reference the original source.
EX: 
Atlas Copco Group's revenue in 2024 was MSEK 176,771.
[AtlasCopca-annual-report-2024.pdf, Page 3]

                                                                  MEMORY
CONVERSATION MEMORY
The system supports conversation IDs.
Previous user and assistant messages are maintained for follow-up questions.

                                                                 RESTAPI
FASTAPI
I used FastAPI as the backend API framework.
The backend exposes endpoints for health checks, document uploads and chat.
The API returns:
Answer
Sources
Confidence
Retrieved Chunk Count
Conversation ID

                                                                 USER INTERFACE
STREAMLIT UI
I used Streamlit to create a simple user interface.
The UI supports:
Multiple PDF Upload
       ↓
Document Processing
       ↓
Chat Interface
       ↓
Question
       ↓
Answer
       ↓
Sources + Confidence

EVALUATION
I created an evaluation framework to measure the RAG pipeline.
Evaluation cases are stored in JSONL format.
Metrics include:
Precision@5
Recall@5
F1@5
Groundedness
Overall Score

                                                                CONTAINERISATION
DOCKER
The project includes Docker and Docker Compose support.
Qdrant runs as a containerized service.
The application can communicate with the local Ollama instance.

                                                                 PROJECT STRUCTURE

Granimals-knowledge-assistant/
│
├── app/
│   ├── ingestion/
│   │   ├── text_chunker.py
│   │   ├── image_extractor.py
│   │   ├── models.py
│   │   ├── ocr.py
│   │   ├── loader.py
│   │   ├── page_renderer.py
│   │   ├── table_extractor.py
│   │   ├── cleaner.py
│   │   ├── storage.py
│   │   └── pipeline.py
│   │
│   ├── retrieval/
│   │   ├── embeddings.py
│   │   ├── qdrant_store.py
│   │   ├── bm25_store.py
│   │   ├── hybrid.py
│   │   ├── query_expander.py
│   │   ├── reranker.py
│   │   └── service.py
│   │
│   ├── rag/
│   ├── llm/
│   ├── memory/
│   └── observability/
│
├── evaluation/
│   ├── dataset.py
│   ├── evaluator.py
│   ├── metrics.py
│   └── rag_eval.jsonl
│
├── scripts/
│   ├── run_ingestion.py
│   ├── index_chunks.py
│   ├── test_rag.py
│   └── run_evaluation.py
│
├── tests/
│
├── ui/
│   └── streamlit_app.py
│
├── data/
│   ├── documents/
│   ├── assets/
│   └── processed/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .dockerignore
├── .gitignore
└── README.md

                                                            HOW TO RUN
1)Install dependencies

pip install -r requirements.txt

2)Start Ollama

ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text
ollama pull llama3.2-vision(OPTIONAL)

3)Start Qdrant

docker compose up -d qdrant

4)Start FastAPI

uvicorn app.main:app --reload --port 8000

5)Start Streamlit

streamlit run ui/streamlit_app.py

6) Upload PDF in UI and start to query.
                                                               






















