# K3s Deployable AI Agent

This project provides a template for creating a k3s-deployable AI agent that wraps a local or remote Large Language Model (LLM). It includes a feedback mechanism to allow for fine-tuning and improvement of the agent's coding abilities.

## Features

*   **FastAPI Backend**: A Python-based backend using FastAPI to serve the agent's API.
*   **Ollama Integration**: Pre-configured to connect to an Ollama instance for LLM inference.
*   **Durable Feedback Storage**: Feedback is persisted via pluggable backends (`sqlite` or `file`).
*   **Dockerized**: Comes with a `Dockerfile` for easy containerization.
*   **Kubernetes Ready**: Includes k3s-compatible deployment and service manifests.
*   **GitHub Actions CI/CD**: An automated workflow to test the application, build, and push the Docker image to a container registry.

## Getting Started

### Prerequisites

*   Docker
*   A Kubernetes cluster (like k3s or minikube)
*   `kubectl` configured to connect to your cluster

### CI/CD Setup

The GitHub Actions workflow in `.github/workflows/ci.yaml` is configured to build and push a Docker image to the GitHub Container Registry (GHCR). To make it work, you need to add the following secrets to your GitHub repository settings:

*   `DOCKERHUB_USERNAME`: Your Docker Hub username (or other registry username).
*   `DOCKERHUB_TOKEN`: A personal access token with permissions to push images to the registry.

The image will be tagged and pushed as `ghcr.io/YOUR_USERNAME/YOUR_REPONAME:latest`. You will need to update the `k8s/deployment.yaml` file to use this image path.

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

    By default feedback is stored in `data/feedback.db` (SQLite). Override with env vars (see below).

## API Endpoints

*   `POST /api/generate`: Takes a prompt and returns a response from the LLM.
*   `POST /api/feedback`: Submits feedback on the agent's responses; persists the record and returns a `feedback_id`.
*   `GET /api/feedback/{feedback_id}`: Retrieves a previously stored feedback record by ID.

## Feedback storage

Feedback is written through a pluggable storage layer (`src/storage.py`).

| Backend | Env value | `FEEDBACK_STORAGE_PATH` meaning | Default path |
|---------|-----------|----------------------------------|--------------|
| SQLite  | `sqlite` (default) | Path to a `.db` file | `data/feedback.db` |
| File    | `file` | Directory of one JSON file per record | `data/feedback` |

Environment variables:

*   `FEEDBACK_STORAGE_BACKEND` — `sqlite` or `file` (default: `sqlite`)
*   `FEEDBACK_STORAGE_PATH` — backend-specific path (see table)

Example (file backend):

```bash
export FEEDBACK_STORAGE_BACKEND=file
export FEEDBACK_STORAGE_PATH=./data/feedback
uvicorn src.main:app --reload
```

### Volume mounts (Docker)

Persist feedback across container restarts by mounting a host directory (or named volume) at `/app/data`:

```bash
# SQLite (default path inside the container: /app/data/feedback.db)
docker run -d \
  -p 8000:8000 \
  -v swe-agent-data:/app/data \
  -e FEEDBACK_STORAGE_BACKEND=sqlite \
  -e FEEDBACK_STORAGE_PATH=/app/data/feedback.db \
  ghcr.io/YOUR_USERNAME/YOUR_REPONAME:latest

# File backend (JSON files under /app/data/feedback)
docker run -d \
  -p 8000:8000 \
  -v swe-agent-data:/app/data \
  -e FEEDBACK_STORAGE_BACKEND=file \
  -e FEEDBACK_STORAGE_PATH=/app/data/feedback \
  ghcr.io/YOUR_USERNAME/YOUR_REPONAME:latest
```

Bind-mount a host path instead of a named volume if preferred:

```bash
docker run -d -p 8000:8000 -v /var/lib/swe-agent:/app/data ...
```

### Volume mounts (Kubernetes)

The manifests under `k8s/` mount a PersistentVolumeClaim at `/app/data` and set storage env vars. Apply as usual:

```bash
kubectl apply -f k8s/
```

Key pieces in `k8s/deployment.yaml`:

*   PVC `ai-agent-data` mounted at `/app/data`
*   `FEEDBACK_STORAGE_BACKEND=sqlite`
*   `FEEDBACK_STORAGE_PATH=/app/data/feedback.db`

Adjust the PVC size/storage class in `k8s/pvc.yaml` for your cluster.

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
