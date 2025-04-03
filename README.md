# LangChain Query System (langchain-qs)

A Multi-Agent Retrieval Augmented Generation (RAG) system built with LangChain and LangGraph.

## Architecture

This repository implements a graph-based multi-agent RAG system with the following components:

- **Document Processing Pipeline**: Handles document ingestion, chunking, and indexing
- **Agent-based Graph System**: Utilizes LangGraph for orchestrating multiple agents
- **Streamlit Web Interface**: Provides an interactive UI for querying the system
- **Integration with LLMs**: Uses various large language models to power the agents

![Architecture](/static/agents_nodes_graph.png)

## Prerequisites

- Python 3.9+
- Git
- Access to LLM APIs (Fireworks, OpenAI, etc.)
- Sufficient disk space for document storage

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/langchain-qs.git
cd langchain-qs
```

2. Use the Makefile to set up your environment and install dependencies:
```bash
# Create virtual environment and install dependencies
make install

# Create .env file with placeholder values
make env
```

3. Configure your `.env` file with your API keys:
```
TAVILY_API_KEY=
FIREWORKS_API_KEY=
API_PUBLIC_KEY=
API_PRIVATE_KEY=
GROUP_ID=
# Add other API keys as needed
```

## Running the Application

Start the Streamlit app:
```bash
make all
```

The application will be available at `http://localhost:8501` by default.

## Docker Support

You can also run the application using Docker:

```bash
# Build Docker image
make docker-build

# Run Docker container
make docker-run
```

## Usage

1. Open the application in your web browser
2. Select your preferred LLM model from the sidebar
3. Enter your query in the input field
4. The system will:
    - Process your query
    - Retrieve relevant information
    - Generate a response using the Multi-Agent framework

## Features

- **Multiple LLM Support**: Choose between different language models
- **Graph-based Agent Workflow**: Agents collaborate to improve response quality
- **Document Retrieval**: Automatically fetches relevant information from indexed documents
- **Query Refinement**: Rewrite queries to improve search results when needed
- **One-Click Deployments**: Deploy easily to various cloud platforms
- **GitHub Integration**: Seamless version control and collaboration support
- **Expanded LLM Providers**: Support for newer models from various providers
- **Enhanced Caching**: Improved response times through optimized caching

## Makefile Commands

For convenience, the following make commands are available:

```bash
make help        # Show all available commands
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)
