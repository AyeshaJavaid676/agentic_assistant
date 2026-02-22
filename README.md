# 🔮 CodeOracle AI - Multi-Agent Intelligent Document Assistant

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-339933.svg)
![Groq](https://img.shields.io/badge/Groq-LLM-FF6B6B.svg)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-FFD21E.svg)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-9B59B6.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents
* [🌟 Overview](#-overview)
* [✨ Key Features](#-key-features)
* [🏗️ System Architecture](#️-system-architecture)
* [🤖 Multi-Agent Architecture](#-multi-agent-architecture)
* [📁 Project Structure](#-project-structure)
* [⚙️ Installation](#️-installation)
* [🔧 Configuration](#-configuration)
* [🚀 Usage Guide](#-usage-guide)
* [📊 Component Details](#-component-details)
* [🔍 Agent Interaction Flow](#-agent-interaction-flow)
* [💾 Data Persistence](#-data-persistence)
* [🎯 Performance Optimization](#-performance-optimization)
* [🧪 Testing](#-testing)
* [📈 Future Enhancements](#-future-enhancements)
* [👥 Contributing](#-contributing)
* [📄 License](#-license)
* [🙏 Acknowledgments](#-acknowledgments)

## 🌟 Overview
**CodeOracle AI** is a cutting-edge multi-agent RAG (Retrieval-Augmented Generation) system designed to intelligently process, understand, and answer questions about coding documents, technical PDFs, and software documentation. Leveraging a sophisticated multi-agent architecture, it provides context-aware responses by combining document analysis, web research, and code understanding capabilities.

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **📄 Multi-Document Processing** | Handle multiple PDFs simultaneously with automatic chunking |
| **🖼️ Vision Understanding** | Qwen 3.5 Vision integration for charts, graphs, and images |
| **🤖 5 Specialized Agents** | Code Expert, Code Explainer, Code Debugger, Web Researcher, Synthesizer |
| **🔍 Semantic Search** | FAISS vector database with HuggingFace embeddings |
| **🌐 Web Integration** | Real-time web research via DuckDuckGo |
| **💬 Conversational Memory** | Full chat history with context retention |
| **🔊 Text-to-Speech** | gTTS integration for audio responses |
| **📊 Source Citation** | Page-level references for all answers |
| **⚡ Real-time Processing** | Async document processing with progress tracking |
| **🎨 Modern UI** | Streamlit interface with falling snow animation |

## 🏗️ System Architecture
*(Section content to be added)*

## 🤖 Agent Responsibilities Matrix

| Agent | Primary Role | Tools Used | Response Type |
| :--- | :--- | :--- | :--- |
| **Supervisor** | Orchestration, routing | All agents | Decision making |
| **Code Expert** | Code search & analysis | FAISS, VectorDB | Code snippets, functions |
| **Code Explainer** | Concept explanation | Groq LLM | Tutorials, explanations |
| **Code Debugger** | Error diagnosis | Groq LLM | Bug fixes, solutions |
| **Web Researcher** | External data | DuckDuckGo | Current info, docs |
| **Synthesizer** | Final compilation | Groq LLM | Comprehensive answers |

---

## 📁 Project Structure

```text
📦 agentic_assistant/
├── 📂 config/
│   └── settings.py                 # Environment & API configuration
├── 📂 modules/
│   ├── pdf_extractor.py            # PDF text & image extraction
│   ├── multimodal_processor.py      # Qwen 3.5 Vision integration
│   ├── vector_store.py              # FAISS vector database management
│   ├── multi_agent.py               # Multi-agent system implementation
│   ├── web_search.py                # DuckDuckGo integration
│   └── tts_service.py               # Text-to-speech functionality
├── 📂 utils/
│   └── helpers.py                   # Utility functions
├── 📂 data/
│   ├── 📂 pdfs/                     # Uploaded PDF storage
│   └── 📂 vectorstore/                # FAISS index & metadata
├── 📂 tests/
│   ├── test_multiagent.py            # Agent system tests
│   └── test_vision.py                 # Vision model tests
├── main.py                            # Streamlit application
├── preload_vectorstore.py              # Document pre-processing
├── .env                                # Environment variables
├── .gitignore                          # Git ignore rules
├── Pipfile                             # Pipenv dependencies
├── Pipfile.lock                         # Locked dependencies
├── requirements.txt                     # pip requirements
└── README.md                            # This file
```

## ⚙️ Installation

### Prerequisites
* **Python 3.13+**
* **Git**
* **Groq API Key**
* **Hugging Face Token**

## ⚙️ Setup Instructions

```bash
# 1. Clone Repository
git clone [https://github.com/AyeshaJavaid676/CodeOracle-AI.git](https://github.com/AyeshaJavaid676/CodeOracle-AI.git)
cd CodeOracle-AI

# 2. Create Virtual Environment
pip install pipenv
pipenv install
pipenv shell

# 3. Configure Environment Variables
echo "GROQ_API_KEY=your_groq_key_here" > .env
echo "HF_TOKEN=your_huggingface_token_here" >> .env

# 4. Install Dependencies
pip install -r requirements.txt

# 5. Run Pre-load Script (Optional)
python preload_vectorstore.py

# 6. Launch Application
streamlit run main.py
```
## 🔧 Configuration
### Environment Variables (.env)
env
```bash
GROQ_API_KEY=gsk_your_groq_api_key_here
HF_TOKEN=hf_your_huggingface_token_here
```

## Settings (config/settings.py)
```bash
# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


# Model Selection
LLM_MODEL = "llama-3.1-8b-instant"  # Optimized for token limits
VISION_MODEL = "Qwen/Qwen3.5-397B-A17B:together"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

## Paths
** PDF_UPLOAD_FOLDER = "data/pdfs"
** VECTOR_STORE_PATH = "data/vectorstore"

## Processing Parameters
** CHUNK_SIZE = 500
** CHUNK_OVERLAP = 50
** MAX_IMAGES_PER_PDF = 100
** SEARCH_RESULTS_COUNT = 10
## 🚀 Usage Guide
### Starting the Application
bash ```
streamlit run main.py
```
## Workflow
Document Upload

Place PDFs in data/pdfs/ folder

Or upload via sidebar interface

Automatic text extraction & chunking

Image Processing

Qwen 3.5 Vision analyzes charts/graphs

Generates detailed descriptions

Stores with page references

Ask Questions

text
🔍 Example Queries:
- "What outliers were found in my EDA project?"
- "Explain the function calculate_mean() in my code"
- "Find all classes related to data visualization"
- "Debug this error: KeyError in dictionary"
- "Show me examples of list comprehension"
Get Answers with Sources

text
📊 WHAT YOUR DOCUMENT SAYS
According to your EDA report (Page 8):
- **19 outliers** detected in Income variable
- Premium "High-Income Customers" segment identified

🔍 DETAILED ANALYSIS
The Z-score method used shows significance at p < 0.05...

📚 Sources (3)
• From EDA_Portfolio.pdf, Page 8
• From Python_Guide.pdf, Page 23
📊 Component Details
1. PDF Extractor
python
class PDFExtractor:
    - extract_text(): Page-level text extraction
    - extract_images(): Base64 image extraction
    - metadata preservation: page numbers, document names
2. Multimodal Processor
python
class MultimodalProcessor:
    - generate_image_description(): Qwen 3.5 Vision integration
    - create_text_embedding(): Vector embeddings
    - process_image_for_rag(): Complete pipeline
3. Vector Store
python
class VectorStore:
    - create_from_documents(): FAISS index creation
    - add_documents(): Incremental updates
    - search(): Semantic similarity search
    - load_existing(): Persistent storage
4. Multi-Agent System
python
class SupervisorAgent:
    - run(): Orchestrates all agents
    - prioritizes document info over web
    - synthesizes final answers

