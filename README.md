# LangGraph Playground

A repository for experimenting with LangGraph, a framework for building stateful, multi-actor applications with LLMs.

## Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/)

### Setup

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd langgraph-playground
   ```

2. **Install dependencies using uv (recommended)**

   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:

   ```bash
   touch .env
   ```

   Add your Google AI API key:

   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

   You can get a Google AI API key from the [Google AI Studio](https://aistudio.google.com/).

## Available Graphs

### Basic Chatbot (`graphs/basic_chatbot/`)

A simple conversational chatbot built with LangGraph that demonstrates the basic concepts of state management and message handling.

**How to run:**

```bash
cd
uv run graphs/basic_chatbot/main.py
```

## Project Structure

```
langgraph-playground/
├── graphs/
│   └── basic_chatbot/
│       └── main.py          # Basic chatbot implementation
├── pyproject.toml           # Project dependencies and metadata
├── uv.lock                  # Lock file for reproducible builds
└── README.md               # This file
```
