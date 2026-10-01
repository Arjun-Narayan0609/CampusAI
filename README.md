# CampusAI

CampusAI is an agentic academic assistant designed to support students and academic workflows through intelligent document retrieval, multi-agent orchestration, and academic tool integrations. In Phase 1, the platform integrates institutional document question answering, synthetic student profile context, deterministic academic utilities (such as SGPA/CGPA calculations and study planning), and an interactive web interface.

## Phase 1 Technology Stack

- **Agent Orchestration**: LangGraph, LangChain
- **LLM Provider**: Groq (`langchain-groq`)
- **Vector Store & Embeddings**: ChromaDB, Sentence-Transformers
- **Document Processing**: PyMuPDF (`pymupdf`)
- **Data & Computation**: Pandas
- **Frontend / Interface**: Streamlit
- **Configuration & Environment**: python-dotenv

## High-Level Folder Structure

```text
CampusAI/
├── app.py                  # Streamlit application entrypoint
├── requirements.txt        # Core project dependencies
├── README.md               # Project documentation
├── .gitignore              # Version control ignore rules
├── .env.example            # Environment variables template
├── data/
│   ├── documents/          # Institutional source documents (.gitkeep)
│   └── students/           # Synthetic student dataset (students.csv)
├── ingestion/              # Document loading, chunking, and metadata pipelines
├── rag/                    # Embeddings, vector store, retrieval, and citations
├── agents/                 # LangGraph state, workflow graph, routers, and agents
├── tools/                  # Deterministic academic calculators & scheduling tools
├── student/                # Student registry and profile state management
├── llm/                    # Groq client and model configuration
└── evaluation/             # RAG and multi-agent evaluation suites
```

## Basic Setup

1. **Clone the repository and enter the directory**:
   ```bash
   git clone <repo-url>
   cd CampusAI
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Unix / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Copy `.env.example` to `.env` and configure your API keys:
   ```bash
   cp .env.example .env
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```
