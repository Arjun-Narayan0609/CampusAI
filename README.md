# 🎓 CampusAI

### Agentic Academic Assistant

CampusAI is a Phase-1 agentic academic assistant that combines **LangGraph orchestration, Retrieval-Augmented Generation (RAG), deterministic academic tools, student context, and response validation** into a single Streamlit application.

Instead of treating every question as a generic LLM prompt, CampusAI first understands the student's request, routes it to the appropriate capability, executes the required operation, validates the result, and then returns a student-readable response.

> **Project status:** Phase 1 completed  
> **Data status:** Uses synthetic/demo student and institutional data

---

## ✨ Overview

Students often need to search through course material, institutional documents, academic rules, schedules, and personal academic information separately.

CampusAI brings these interactions into one assistant.

The system supports:

- 📚 Academic questions grounded in indexed documents
- 🧠 Study-plan generation
- 📊 CGPA / SGPA operations
- 📅 Personal study-session schedules
- 💬 General questions
- 🔎 Source-aware RAG responses
- 🛡️ Response validation and bounded fallbacks

The core design principle is simple:

```text
LLM → understands and reasons about the request
RAG → provides evidence for academic/institutional information
Python tools → perform exact calculations and structured operations
LangGraph → orchestrates the workflow
Reviewer → validates the result before it reaches the student
```

This separation makes the system more reliable than a simple LLM chat wrapper.

---

## 🤖 Why CampusAI Is Agentic

CampusAI uses explicit decision points, state, conditional routing, capability execution, and response validation.

```text
                    Student Question
                           │
                           ▼
                  ┌─────────────────┐
                  │ Question        │
                  │ Analyzer        │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ LangGraph       │
                  │ Conditional     │
                  │ Router          │
                  └────────┬────────┘
                           │
        ┌──────────────────┼───────────────────┐
        │                  │                   │
        ▼                  ▼                   ▼
   Academic Query     Study Plan         CGPA / Schedule
        │                  │                   │
        ▼                  ▼                   ▼
       RAG          Deterministic        Python Tools
                    Planner/Tools
        │                  │                   │
        └──────────────────┼───────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Reviewer   │
                    └──────┬───────┘
                           │
                           ▼
                     Final Answer
```

### Agentic concepts demonstrated

- **State** — student ID, question, intent, context, tool results, answer, and sources are carried through the graph.
- **Intent classification** — the question analyzer identifies the primary operation.
- **Conditional routing** — LangGraph selects the appropriate capability.
- **Capability execution** — RAG or deterministic tools perform the required operation.
- **Grounding** — academic/institutional answers are based on retrieved evidence.
- **Validation** — the reviewer checks whether the response has the required evidence or tool result.

---

## 🏗️ Architecture

```text
┌─────────────────────────────┐
│       Streamlit UI          │
│  Student + Question Input   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ run_agent(student_id,       │
│            question)        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│        LangGraph            │
│                             │
│  Analyzer → Router → Node   │
└──────────────┬──────────────┘
               │
     ┌─────────┼──────────┬──────────┐
     ▼         ▼          ▼          ▼
    RAG     Planner      Tools     General
     │         │        ┌──┴──┐       │
     │         │        │     │       │
     │         │       CGPA Schedule  │
     │         │        │     │       │
     └─────────┴────────┴─────┴───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │   Reviewer   │
                  └──────┬───────┘
                         │
                         ▼
                  Student Response
```

### Supporting components

```text
Academic PDFs
     │
     ▼
PDF ingestion → chunking → embeddings
     │
     ▼
   ChromaDB
     │
     ▼
retrieve_context(...)
     │
     ▼
   RAG node
```

Student information is supplied by the synthetic student registry, while structured academic operations are handled by deterministic Python code.

---

## 🧩 Core Capabilities

| Capability | What it does | Implementation |
|---|---|---|
| 📚 Academic Q&A | Answers academic/institutional questions from indexed documents | RAG + ChromaDB |
| 📝 Study Planning | Creates structured study sessions using student information and constraints | Deterministic planner |
| 📊 CGPA / SGPA | Performs academic calculations | Python tools |
| 📅 Schedule | Views and manages study sessions | Internal scheduler |
| 💬 General | Handles general-purpose questions | Groq LLM |
| 🛡️ Response Review | Checks grounding and tool-result validity | Deterministic reviewer |

---

## 📚 Academic RAG

CampusAI's academic knowledge layer follows a retrieval-first approach.

```text
PDF Documents
     │
     ▼
Document Loading
     │
     ▼
Chunking + Metadata
     │
     ▼
Embeddings
     │
     ▼
ChromaDB
     │
     ▼
Semantic Retrieval
     │
     ▼
Relevant Context + Sources
     │
     ▼
LLM Answer
```

The retriever can use metadata such as:

- department
- semester
- subject
- document category
- page/source information

The RAG interface returns retrieved context together with source metadata.

### Grounding rule

CampusAI does not invent institutional information when supporting evidence is unavailable.

```text
Question
   │
   ▼
Retrieval
   │
   ├── Evidence found ──► Generate grounded answer
   │
   └── No evidence ─────► Bounded fallback
```

For example, if the indexed corpus contains no material explaining a particular academic concept, the system should not fabricate an institutional answer.

### Adding academic documents

Place PDF documents inside:

```text
data/documents/
```

Then rebuild the vector store using the project's ingestion command:

```powershell
.venv\Scripts\python.exe -m rag.ingest
```

After ingestion, the new documents can be retrieved by the RAG pipeline.

> The knowledge available to CampusAI is limited to the documents that have actually been indexed.

---

## 🧠 LangGraph Workflow

CampusAI currently uses five Phase-1 intents:

| Intent | Route | Purpose |
|---|---|---|
| `ACADEMIC_QUERY` | RAG | Academic/institutional information |
| `STUDY_PLAN` | Study Planner | Personal study-plan generation |
| `CGPA` | CGPA tools | CGPA/SGPA operations |
| `SCHEDULE` | Scheduler | Study-session schedule operations |
| `GENERAL` | General | General-purpose questions |

The public backend interface is:

```python
run_agent(student_id, question)
```

It returns a stable response containing:

```python
{
    "answer": ...,
    "intent": ...,
    "sources": ...,
    "tool_result": ...
}
```

### LangGraph vs LangChain

**LangGraph** is responsible for:

- workflow orchestration
- state
- conditional routing
- node execution

**LangChain** provides abstractions used for:

- prompts
- LLM integration
- structured model interaction
- tool integration

---

## 🧮 Deterministic Academic Tools

Exact operations are deliberately kept outside the LLM.

### CGPA / SGPA

The LLM does not perform the actual academic arithmetic.

```text
Student request
      │
      ▼
Intent: CGPA
      │
      ▼
Python calculation
      │
      ▼
Deterministic result
      │
      ▼
Reviewer
      │
      ▼
Student answer
```

### Study Planner

The planner uses structured student information such as subjects, academic dates, available study time, and planning constraints.

The core allocation logic is deterministic rather than relying on an LLM to invent a timetable.

### Scheduler

The internal scheduler manages study sessions and supports structured schedule operations without requiring an external calendar service.

This demonstrates an important agentic pattern:

> **The LLM understands the request; trusted code performs the exact operation.**

---

## 👨‍🎓 Student Context

CampusAI uses a synthetic student registry as the source of student-specific context.

Student profiles can provide information such as:

- Student ID
- Name
- Department
- Semester
- Year
- CGPA
- Subjects
- Exam information
- Study availability/preferences

The student data is **synthetic/demo data** and is not intended to represent real student records.

---

## 🛡️ Response Reviewer

The reviewer acts as the final quality checkpoint before the response is displayed.

### For RAG responses

It verifies that the response has the required:

- answer
- retrieved context
- source information

### For tool-based responses

It verifies that the required tool result exists before allowing the response to claim a successful operation.

If required information is missing or an operation fails, CampusAI produces a bounded fallback instead of silently presenting an unsupported answer.

---

## 🖥️ Streamlit Interface

The current Streamlit application provides:

- Student selection from the registry
- Natural-language question input
- CampusAI response rendering
- Intent display
- CGPA result presentation
- Schedule table
- Study-plan table
- Per-subject study-plan summary
- RAG source display
- Error/failure handling

The UI communicates with the backend through the stable:

```python
run_agent(student_id, question)
```

interface.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Agent orchestration | LangGraph |
| LLM / prompt abstractions | LangChain |
| LLM provider | Groq |
| RAG vector store | ChromaDB |
| Embeddings | Sentence Transformers |
| PDF processing | PyMuPDF |
| Backend / tools | Python |
| Student data | CSV |
| Internal scheduling | SQLite / project scheduler implementation |
| Testing | pytest |

---

## 📁 Project Structure

```text
CampusAI/
├── agents/
│   ├── graph.py
│   ├── question_analyzer.py
│   ├── reviewer.py
│   ├── router.py
│   ├── state.py
│   └── study_planner.py
│
├── data/
│   ├── documents/
│   └── students/
│
├── ingestion/
│   ├── chunker.py
│   ├── ingest.py
│   ├── loader.py
│   └── metadata.py
│
├── llm/
│   └── groq.py
│
├── rag/
│   ├── citations.py
│   ├── embeddings.py
│   ├── prompts.py
│   ├── retriever.py
│   └── vectorstore.py
│
├── student/
│   ├── registry.py
│   └── state.py
│
├── tools/
│   ├── cgpa.py
│   └── scheduler.py
│
├── evaluation/
│   ├── agent_tests.py
│   └── rag_evaluation.py
│
├── docs/
├── app.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

### 2. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file using `.env.example` as the template.

For example:

```text
GROQ_API_KEY=your_groq_api_key
```

Never commit the real API key.

### 5. Prepare the knowledge base

Place academic/institutional PDFs in:

```text
data/documents/
```

Then run:

```powershell
.venv\Scripts\python.exe -m rag.ingest
```

---

## ▶️ Running CampusAI

Start the Streamlit application:

```powershell
streamlit run app.py
```

The application opens in the browser.

The runtime flow is:

```text
Browser
   ↓
Streamlit
   ↓
run_agent()
   ↓
LangGraph
   ↓
Analyzer
   ↓
Router
   ↓
RAG / Planner / Tools / General
   ↓
Reviewer
   ↓
Streamlit Response
```

---

## 🧪 Testing

Run the complete test suite:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

The latest verified Phase-1 backend run completed with:

```text
180 passed
37 subtests passed
```

The test suite covers major parts of the system including:

- Question Analyzer
- Intent routing
- LangGraph workflow
- RAG integration
- Response reviewer
- CGPA
- Study planner
- Scheduler
- Public `run_agent()` interface

---

## 💬 Example Queries

| Example question | Route |
|---|---|
| `What is normalization in DBMS?` | `ACADEMIC_QUERY → RAG` |
| `Explain process in Operating Systems.` | `ACADEMIC_QUERY → RAG` |
| `What is my CGPA?` | `CGPA` |
| `Show my schedule.` | `SCHEDULE` |
| `Create a study plan for me.` | `STUDY_PLAN` |
| `What is Docker?` | `GENERAL` |

The exact answer to an academic query depends on whether relevant material exists in the indexed document corpus.

---

## 🎯 Design Principles

CampusAI follows several deliberate engineering principles:

- **Ground academic/institutional answers in retrieved evidence**
- **Do not fabricate unsupported institutional information**
- **Use deterministic Python code for exact calculations**
- **Keep orchestration separate from the UI**
- **Carry student context explicitly through the workflow**
- **Validate responses before presenting them**
- **Keep Phase-1 scope manageable**
- **Prefer stable interfaces between project modules**

---

## 🚧 Phase-1 Scope & Limitations

CampusAI intentionally keeps the first version lightweight.

The current implementation does not attempt to provide:

- External calendar OAuth
- Complex authentication
- Mobile applications
- Voice interfaces
- Microservice infrastructure
- Model fine-tuning
- Production-scale distributed infrastructure

The architecture is intentionally built around:

```text
Streamlit
   +
Python
   +
LangGraph
   +
LangChain
   +
Groq
   +
ChromaDB
   +
Local embeddings
   +
SQLite / local structured storage
```

This keeps the system practical for a demonstrable academic project while still showing meaningful agentic architecture.

---

## 🎬 Recommended Demo Flow

A concise demonstration can follow this sequence:

### 1. Academic RAG

Ask:

```text
What is normalization in DBMS?
```

Show:

- `ACADEMIC_QUERY`
- generated answer
- retrieved source

### 2. CGPA

Ask:

```text
What is my CGPA?
```

Show that the calculation comes from deterministic backend data/tools.

### 3. Schedule

Ask:

```text
Show my schedule.
```

Show the structured schedule returned by the scheduler.

### 4. Study Plan

Ask:

```text
Create a study plan for me.
```

Show:

- sessions
- dates
- subjects
- duration
- per-subject summary

### 5. General Question

Ask:

```text
What is Docker?
```

Show the `GENERAL` route.

### 6. Explain the Architecture

Demonstrate that the same natural-language interface can route different requests through different capabilities.

This is the key point of the CampusAI demo:

> **One assistant interface, multiple specialized execution paths.**

---

## 📌 Project Status

### Phase 1 — Completed ✅

Implemented and integrated:

- Student context
- Question analysis
- LangGraph state and routing
- Academic RAG
- ChromaDB knowledge retrieval
- Deterministic CGPA/SGPA operations
- Deterministic study planning
- Internal scheduling
- Response reviewer
- Stable `run_agent()` interface
- Streamlit application
- Academic source display
- Phase-1 test coverage

CampusAI is an **academic/demo application using synthetic institutional and student data**. It is not intended to be treated as a production university information system.

---

## 👥 Architecture Ownership

The project was developed using separated module responsibilities:

| Area | Responsibility |
|---|---|
| RAG / Knowledge | Document ingestion, embeddings, retrieval and sources |
| Agent / Orchestration | LangGraph, analyzer, routing, reviewer and integration |
| Academic Tools | Student registry, CGPA, planner and scheduler |
| Application | Streamlit student-facing interface |

The modules communicate through explicit function contracts rather than tightly coupling the entire application.

---

## 🔐 Security Note

Never commit:

```text
.env
```

or any real API key.

Use:

```text
.env.example
```

for sharing the required environment-variable names.

The included student and institutional information is synthetic/demo data.

---

## 📄 Project Documentation

The repository also contains the project's architecture and implementation documentation under:

```text
docs/
```

These documents describe the intended solution, architecture, responsibilities, and Phase-1 scope.

---

## ⭐ Key Takeaway

CampusAI demonstrates how an academic assistant can combine:

```text
Natural Language
      +
Intent Understanding
      +
LangGraph Orchestration
      +
RAG
      +
Deterministic Tools
      +
Student Context
      +
Response Validation
      =
Agentic Academic Assistant
```

The important idea is not simply using an LLM to answer questions.

It is **deciding what kind of task the student is asking for, selecting the correct capability, executing it with the appropriate technology, validating the result, and then presenting it clearly.**
