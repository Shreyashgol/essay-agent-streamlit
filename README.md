# AI Essay Agent

A Streamlit-based autonomous essay writing application that takes a topic, performs web research, creates a first draft, critiques it, and then produces a polished final essay.

The project combines an LLM, a web search tool, and a LangGraph workflow so the essay is generated through multiple focused steps instead of a single prompt.

## Project Description

This app is designed to simulate a small multi-step writing agent. Given a topic, it:

1. Generates targeted search queries.
2. Searches the web for relevant information.
3. Synthesizes research notes.
4. Writes a first draft.
5. Reviews the draft against the research.
6. Produces a refined final version.

The UI is built with Streamlit, so the whole workflow can be run from a simple browser interface.

## Tech Stack

### 1. Streamlit
- Used for building the frontend and running the app locally with minimal setup.
- Chosen because it is fast for prototyping AI applications and makes it easy to display workflow progress and outputs.

### 2. LangChain
- Used for prompt composition and LLM interaction.
- Chosen because it provides clean abstractions for chat models, prompt templates, and tool integration.

### 3. LangGraph
- Used to define the agent workflow as a stateful graph.
- Chosen because the project is naturally multi-step: each stage depends on the output of the previous stage.

### 4. Groq + `langchain-groq`
- Used as the LLM provider for keyword generation, research synthesis, drafting, critique, and final rewriting.
- Chosen because it offers fast inference and integrates well with LangChain.

### 5. DuckDuckGo Search
- Used to fetch public web search results for topic research.
- Chosen because it adds lightweight external context without requiring a separate search API setup.

### 6. Python Dotenv
- Used to load environment variables from a local `.env` file.
- Chosen to keep API keys out of source code and simplify local development.

## Why This Architecture

This project uses a staged agent workflow instead of a single prompt because essay writing is easier to control when the work is split into smaller responsibilities:

- Research is separated from writing.
- Drafting is separated from critique.
- The final version is generated only after review.

This improves structure, reduces unsupported claims, and makes the app easier to extend later with better tools, memory, or citation support.

## Project Structure

```text
essay-agent-streamlit/
├── app.py
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── secrets.toml.example
└── README.md
```

### File Overview

- `app.py`: Main Streamlit app, agent state, workflow nodes, graph compilation, and UI.
- `requirements.txt`: Python dependencies required to run the project.
- `.env.example`: Example environment variable file for the Groq API key.
- `.streamlit/secrets.toml.example`: Example Streamlit secrets file included in the repo.
- `README.md`: Project documentation and setup guide.

## Workflow Overview

The agent in `app.py` is built as a LangGraph pipeline with the following nodes:

1. `keyword_generator`
2. `web_researcher`
3. `first_drafter`
4. `reviewer`
5. `final_writer`

Each node updates a shared state object containing:

- Topic
- Search terms
- Research data
- First draft
- Critique
- Final draft

## Local Setup

### Prerequisites

- Python installed locally
- `pip` available
- A Groq API key

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd essay-agent-streamlit
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Then set your Groq API key:

```env
groq_api=your_groq_api_key_here
```

Note: The current app loads the API key from the environment using `python-dotenv`, so the `.env` file is the setup path that matches the code as it exists today.

### 5. Run the app

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, enter an essay topic, and click `Generate Essay`.

## How the App Works

When the user submits a topic:

- The app generates focused search keywords.
- It performs web searches for those keywords.
- It synthesizes raw search output into usable research notes.
- It writes a first draft grounded in those notes.
- It critiques the draft for structure, flow, and factual alignment.
- It rewrites the essay into a cleaner final draft.

The Streamlit interface also exposes the internal research notes and critique so the workflow is easier to inspect.

## Dependencies

The current `requirements.txt` includes:

- `streamlit`
- `langchain`
- `langgraph`
- `langchain-groq`
- `duckduckgo-search`
- `dotenv`

## Possible Improvements

- Add citations or source links in the final essay
- Improve error handling for failed searches or malformed model output
- Store intermediate results more explicitly in the UI
- Add support for essay tone, length, or audience controls
- Replace generic dependency pins with versioned requirements for reproducibility

## Notes

- Keep `.env` and `.streamlit/secrets.toml` out of version control.
- The app currently depends on external web search and LLM responses, so output quality may vary by topic and API availability.
