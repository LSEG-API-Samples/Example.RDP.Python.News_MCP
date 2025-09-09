# News MCP Server

This repository demonstrates an example implementation of a **Model Context Protocol (MCP) server** that exposes tools for querying financial news data from Refinitiv Data Platform (RDP). The project showcases how to integrate news search capabilities with an agentic system built with LangGraph.

## What is MCP?

The Model Context Protocol (MCP) is a standardized way to connect AI assistants with data sources and tools. This implementation creates a news server that LLMs can query to get real-time financial news information, enabling AI applications to provide up-to-date market insights.


## Project Structure

```
├── llm.py                    # LLM Configuration
├── chat_app.py               # FastAPI chat interface
├── mcp-servers/
│   ├── news-server.py        # MCP server implementation
│   └── rdp_auth.py          # RDP authentication utilities
├── evals/
│   └── trajectory_llm_as_judge.py  # Evaluation framework
├── pyproject.toml            # Project dependencies
└── langgraph.json           # LangGraph configuration
```

## Configuration

### RDP (Refinitiv Data Platform) Credentials

You need valid Refinitiv Data Platform (RDP) credentials with entitlements for news data:

```bash
export RDP_USERNAME="your-username"
export RDP_PASSWORD="your-password"  
export RDP_CLIENT_ID="your-client-id"
```

### LangSmith (Optional - for evaluations)

```bash
export LANGSMITH_API_KEY="your-langsmith-api-key"
export LANGSMITH_TRACING="true"
```

**Important**: Please check your license agreement regarding the use of News data with LLMs before running this application.

For more information, see: https://www.lseg.com/en/data-analytics/financial-news-service/machine-readable-news

## Usage

### 1. MCP Inspector (Development & Testing)

Use the MCP Inspector to test and explore the news MCP server tools:

```bash
uv run mcp dev mcp-servers/news-server.py
```

This opens a web interface where you can:
- Test news search queries
- Explore available tools and their parameters
- View response formats and sample data

### 2. LangGraph Studio

Launch the interactive LangGraph development environment:

```bash
uv run langgraph dev
```

This provides a visual interface for:
- Building and testing conversation flows
- Debugging agent behavior
- Monitoring tool usage

### 3. Custom Chat Interface

Start the FastAPI-based chat application:

```bash
uv run chat_app.py
```

Access the chat interface at `http://localhost:8000` to interact with the news-enabled AI assistant.

## Sample Queries

Try these example queries in any of the interfaces:

```
"Find recent news about Apple"
"What's the latest on Tesla?"
```

## Evaluation

Run automated evaluations to assess the system's performance. 
This requires a LANGSMITH account.

```bash
uv run pytest evals/trajectory_llm_as_judge.py --langsmith-output
```
