# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Running the Applications
```bash
# Run the Flask API server (production-ready with cascading logic)
python api.py

# Run the Streamlit application (alternative interface)
streamlit run app.py

# Test the API
python test_api.py
```

### Environment Setup
- Set `GROQ_API_KEY` environment variable for the Groq API
- For Streamlit version, configure `.streamlit/secrets.toml` with `GROQ_API_KEY`

## Architecture Overview

This is an AI assistant system with multiple interfaces designed to answer technical questions using a cascading knowledge retrieval approach.

### Core Components

**api.py** - Primary Flask API server with intelligent cascading logic:
- First attempts to answer questions using the local manual (`manual_limpo.txt`)
- If no relevant answer is found in the manual, automatically falls back to web search
- Uses Groq's LLaMA model for response generation
- Implements conversation history support
- CORS enabled for web integration

**app.py** - Alternative Streamlit interface:
- Interactive web app with chat interface
- Uses FAISS vector search for local knowledge retrieval
- Implements intelligent decision-making about information sources
- Creates embeddings using sentence-transformers
- More sophisticated web search with multiple article extraction

**chatbot.html** - Frontend chat widget:
- Embeddable chat interface for websites
- Connects to the Flask API
- Real-time messaging with typing indicators
- Responsive design with floating chat bubble

### Key Technical Details

**Knowledge Sources:**
- Local manual: `manual_limpo.txt` (loaded into memory)
- Web search: DuckDuckGo integration with content extraction via trafilatura
- Knowledge base: `conhecimento/` directory contains the source manual

**AI Integration:**
- Primary LLM: Groq API with LLaMA 3.1-8B Instant model (`llama-3.1-8b-instant`)
- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` for Portuguese content
- Vector database: FAISS for similarity search (app.py only)

**Response Strategy:**
- Manual-first approach: Always check local knowledge base first
- Automatic fallback: Web search triggered when manual doesn't contain relevant information
- Context-aware: Maintains conversation history for follow-up questions
- Portuguese-focused: All responses generated in Portuguese

### File Structure
- `api.py` - Main Flask API with cascading logic
- `app.py` - Streamlit alternative interface
- `chatbot.html` - Frontend chat widget
- `test_api.py` - API testing script
- `manual_limpo.txt` - Core knowledge base
- `conhecimento/` - Source knowledge directory
- `requirements.txt` - Python dependencies

### Development Workflow
1. Ensure `GROQ_API_KEY` is set in environment
2. Install dependencies with `pip install -r requirements.txt`
3. Run `python api.py` for the main API server
4. Use `python test_api.py` to test API functionality
5. Open `chatbot.html` in browser to test the frontend interface

The system is designed to be deployed as a Flask API with the HTML widget embedded in external websites for technical support purposes.