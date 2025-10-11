# K3s Deployable AI Agent

This project provides a template for creating a k3s-deployable AI agent that wraps a local or remote Large Language Model (LLM). It includes a feedback mechanism to allow for fine-tuning and improvement of the agent's coding abilities.

## Features

*   **FastAPI Backend**: A Python-based backend using FastAPI to serve the agent's API.
*   **Ollama Integration**: Pre-configured to connect to an Ollama instance for LLM inference.
*   **Dockerized**: Comes with a `Dockerfile` for easy containerization.
*   **Kubernetes Ready**: Includes k3s-compatible deployment and service manifests.
*   **GitHub Actions CI/CD**: An automated workflow to test the application, build, and push the Docker image to a container registry.

## Getting Started

### Prerequisites

*   Docker
*   A Kubernetes cluster (like k3s or minikube)
*   `kubectl` configured to connect to your cluster

### CI/CD Setup

The GitHub Actions workflow in `.github/workflows/ci.yaml` is configured to build and push a Docker image to Docker Hub. To make it work, you need to add the following secrets to your GitHub repository settings:

*   `DOCKERHUB_USERNAME`: Your Docker Hub username (`jarcher1200`).
*   `DOCKERHUB_TOKEN`: A personal access token for Docker Hub with permissions to push images.

The image will be tagged and pushed to `jarcher1200/<your-repo-name>:latest`. The `k8s/deployment.yaml` file is already configured to pull this image.

### Local Development

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the application:**
    ```bash
    uvicorn src.main:app --reload
    ```
    The application will be available at `http://127.0.0.1:8000`.

## API Endpoints

*   `POST /api/generate`: Takes a prompt and returns a response from the LLM.
*   `POST /api/feedback`: Allows submitting feedback on the agent's responses to be used for future training.

## Deployment

1.  **Apply the Kubernetes manifests:**
    ```bash
    kubectl apply -f k8s/
    ```

2.  **Verify the deployment:**
    ```bash
    kubectl get pods
    kubectl get services
    ```