# 🔮 CodeOracle AI - Multi-Agent Intelligent Document Assistant

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-339933.svg)
![Groq](https://img.shields.io/badge/Groq-LLM-FF6B6B.svg)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-FFD21E.svg)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-9B59B6.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Overview
**CodeOracle AI** is a cutting-edge multi-agent RAG (Retrieval-Augmented Generation) system designed to intelligently process, understand, and answer questions about coding documents, technical PDFs, and software documentation. Leveraging a sophisticated multi-agent architecture, it provides context-aware responses by combining document analysis, web research, and code understanding capabilities.

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **📄 Multi-Document Processing** | Handle multiple PDFs simultaneously with automatic chunking |
| **🖼️ Vision Understanding** | Qwen 3.5 Vision integration for charts, graphs, and images |
| **🤖 5 Specialized Agents** | Code Expert, Code Explainer, Code Debugger, Web Researcher, Synthesizer |
| **🔍 Semantic Search** | FAISS vector database with HuggingFace embeddings |
| **🌐 Web Integration** | Real-time web research via Tavily |
| **💬 Conversational Memory** | Full chat history with context retention |
| **🔊 Text-to-Speech** | gTTS integration for audio responses |
| **📊 Source Citation** | Page-level references for all answers |
| **⚡ Real-time Processing** | Async document processing with progress tracking |
| **🎨 Modern UI** | Streamlit interface with falling snow animation |

## 🏗️ System Architecture
![Architecture Diagram](https://raw.githubusercontent.com/AyeshaJavaid676/agentic_assistant/master/System%20Architecture.jpeg)

## 🤖 Agent Responsibilities Matrix

| Agent | Primary Role | Tools Used | Response Type |
| :--- | :--- | :--- | :--- |
| **Supervisor** | Orchestration, routing | All agents | Decision making |
| **Code Expert** | Code search & analysis | FAISS, VectorDB | Code snippets, functions |
| **Code Explainer** | Concept explanation | Groq LLM | Tutorials, explanations |
| **Code Debugger** | Error diagnosis | Groq LLM | Bug fixes, solutions |
| **Web Researcher** | External data | Tavily | Current info, docs |
| **Synthesizer** | Final compilation | Groq LLM | Comprehensive answers |

---

## 📁 Project Structure

```text
📦 agentic_assistant/
├── 📂 config/
│   └── settings.py                  # Environment & API configuration
├── 📂 modules/
│   ├── pdf_extractor.py             # PDF text & image extraction
│   ├── multimodal_processor.py      # Qwen 3.5 Vision integration
│   ├── vector_store.py              # FAISS vector database management
│   ├── multi_agent.py               # Multi-agent system implementation
│   ├── web_search.py                # Tavily integration
│   └── tts_service.py               # Text-to-speech functionality
├── 📂 utils/
│   └── helpers.py                   # Utility functions
├── 📂 data/
│   ├── 📂 pdfs/                     # Uploaded PDF storage
│   └── 📂 vectorstore/              # FAISS index & metadata
├── 📂 tests/
│   ├── test_multiagent.py           # Agent system tests
│   ├── test_vision.py               # Vision model tests
│   ├── test_working_Hf.py           # HuggingFace connection test
│   ├── test_modules.py              # Core modules functionality test
│   ├── test_preload.py              # Vectorstore preloading test
│   ├── test_setup.py                # Environment setup validation
│   ├── test_tts_simple.py           # Basic TTS output test
│   ├── test_search.py               # Web search/Tavily test
│   └── test_env.py                  # .env variable verification
├── main.py                          # Streamlit application
├── preload_vectorstore.py            # Document pre-processing
├── .env                             # Environment variables
├── .gitignore                       # Git ignore rules
├── Pipfile                          # Pipenv dependencies
├── Pipfile.lock                     # Locked dependencies
├── requirements.txt                 # pip requirements
└── README.md                        # Project documentation
```

## ⚙️ Installation

### Prerequisites
* **Python 3.13+**
* **Git**
* **Groq API Key**
* **Hugging Face Token**

## Output Example

![Output Example ](https://raw.githubusercontent.com/AyeshaJavaid676/agentic_assistant/master/Output.jpg)
![](https://raw.githubusercontent.com/AyeshaJavaid676/agentic_assistant/master/Output1.jpg)
![](https://raw.githubusercontent.com/AyeshaJavaid676/agentic_assistant/master/Output2.jpg)
![Sequnece Flow Diagram:](https://raw.githubusercontent.com/AyeshaJavaid676/agentic_assistant/master/Agents.jpg)
## Sequnece Flow Diagram:





