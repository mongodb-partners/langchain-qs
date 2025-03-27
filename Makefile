# Makefile for LangChain Query System (langchain-qs)

# Variables
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit
DOCKER := docker
APP := app.py
REQUIREMENTS := requirements.txt
ONE_CLICK_SCRIPT := one-click.ksh

# Default target
.PHONY: all
all: install run

# Create virtual environment and install dependencies
.PHONY: install
install: $(VENV)/bin/activate

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r $(REQUIREMENTS)
	touch $(VENV)/bin/activate

# Run the streamlit application
.PHONY: run
run:
	$(STREAMLIT) run $(APP)

# Build the search index
.PHONY: build-index
build-index:
	$(PYTHON) -c "from rag.utils import create_search_index; create_search_index()"

# Create .env file if it doesn't exist
.PHONY: env
env:
	@if [ ! -f .env ]; then \
		echo "Creating .env file with placeholder values..."; \
		echo "TAVILY_API_KEY=" >> .env; \
		echo "FIREWORKS_API_KEY=" >> .env; \
		echo "MONGODB_URI=" >> .env; \
		echo "MONGODB_COLLECTION=rag" >> .env; \
		echo "AWS_ACCESS_KEY_ID=" >> .env; \
		echo "AWS_SECRET_ACCESS_KEY=" >> .env; \
		echo "AWS_SESSION_TOKEN=" >> .env; \
		echo ".env file created. Please fill in your API keys."; \
	else \
		echo ".env file already exists."; \
	fi

# Build Docker image
.PHONY: docker-build
docker-build:
	$(DOCKER) build -t langchain-qs .

# Run Docker container
.PHONY: docker-run
docker-run:
	$(DOCKER) run -p 8501:8501 --env-file .env langchain-qs

# Clean generated files
.PHONY: clean
clean:
	rm -rf __pycache__
	rm -rf **/__pycache__
	rm -rf .pytest_cache
	rm -rf .coverage

# Full clean including virtual environment
.PHONY: clean-all
clean-all: clean
	rm -rf $(VENV)

# Help command
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all          : Install dependencies and run the application"
	@echo "  install      : Create virtual environment and install dependencies"
	@echo "  run          : Run the Streamlit application"
	@echo "  build-index  : Build the search index"
	@echo "  env          : Create .env file with placeholder values if it doesn't exist"
	@echo "  docker-build : Build Docker image"
	@echo "  docker-run   : Run Docker container with the application"
	@echo "  clean        : Remove generated files"
	@echo "  clean-all    : Remove generated files and virtual environment"
	@echo "  help         : Show this help message"