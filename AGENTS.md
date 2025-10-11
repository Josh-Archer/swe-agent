# Agent Instructions

This document provides instructions for AI agents working on this codebase.

## General Principles

*   **Goal**: The primary goal of this project is to create a deployable AI agent that can be improved over time through feedback.
*   **Technology Stack**: The backend is Python with FastAPI, containerized with Docker, and deployed on Kubernetes (k3s). The agent interacts with an LLM via Ollama.
*   **Code Style**: Follow PEP 8 for Python code. Keep code clean, modular, and well-documented.

## Development Workflow

1.  **Understand the Task**: Before writing any code, ensure you understand the requirements. Ask for clarification if anything is unclear.
2.  **Write Tests**: For any new feature or bug fix, write or update tests in the `tests/` directory.
3.  **Implement the Change**: Write the application code in the `src/` directory.
4.  **Update Documentation**: If you add or change a feature, update the `README.md` and any relevant documentation.
5.  **Run Tests**: Ensure all tests pass before submitting your work.
6.  **Verify Deployment**: When making changes to the deployment configuration, verify that the application can be deployed successfully.

## CI/CD

The CI/CD pipeline is defined in `.github/workflows/ci.yaml`. It automates testing and container image publishing. When modifying the workflow, ensure it remains robust and efficient. The image is pushed to a container registry, and the Kubernetes manifests should be updated accordingly.