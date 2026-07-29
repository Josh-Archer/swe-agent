# K3s Deployable AI Agent

This project provides a template for creating a k3s-deployable AI agent that wraps a local or remote Large Language Model (LLM). It includes a feedback mechanism to allow for fine-tuning and improvement of the agent's coding abilities.

## Features

*   **FastAPI Backend**: A Python-based backend using FastAPI to serve the agent's API.
*   **Ollama Integration**: Pre-configured to connect to an Ollama instance for LLM inference.
*   **Streaming Responses**: Optional Server-Sent Events (SSE) streaming from `/api/generate`.
*   **API Authentication**: Optional API-key auth (`X-API-Key` or `Authorization: Bearer`), or network-policy-only access for in-cluster clients.
*   **Dockerized**: Comes with a `Dockerfile` for easy containerization.
*   **Kubernetes Ready**: Includes k3s-compatible deployment, service, and NetworkPolicy manifests.
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

2.  **(Optional) Enable API key auth:**
    ```bash
    # Windows PowerShell
    $env:API_KEY = "dev-secret"

    # bash
    export API_KEY=dev-secret
    ```
    When `API_KEY` is unset, protected endpoints are open (handy for local work and NetworkPolicy-only clusters).

3.  **Run the application:**
    ```bash
    uvicorn src.main:app --reload
    ```
    The application will be available at `http://127.0.0.1:8000`. OpenAPI docs: `http://127.0.0.1:8000/docs`.

## API Endpoints

| Method | Path            | Auth        | Description                                      |
|--------|-----------------|-------------|--------------------------------------------------|
| `GET`  | `/`             | Public      | Health check (liveness/readiness)                |
| `POST` | `/api/generate` | If configured | Generate a model response (JSON or SSE stream) |
| `POST` | `/api/feedback` | If configured | Submit feedback for future training            |

### Authentication

When the `API_KEY` environment variable is set, `/api/generate` and `/api/feedback` require one of:

*   Header: `X-API-Key: <key>`
*   Header: `Authorization: Bearer <key>`

`/` always remains public so Kubernetes probes keep working.

**Network-policy-friendly option:** leave `API_KEY` unset, expose the service as `ClusterIP` only, and apply `k8s/networkpolicy.yaml` so only in-cluster pods can reach the agent.

### Client examples

#### Non-streaming generate

```bash
curl -sS -X POST "http://127.0.0.1:8000/api/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-secret" \
  -d '{"prompt": "Explain Kubernetes in one sentence.", "model": "llama2"}'
```

Response:

```json
{"response": "..."}
```

#### Streaming generate (SSE)

Set `"stream": true`. The response is `text/event-stream`:

```bash
curl -N -X POST "http://127.0.0.1:8000/api/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev-secret" \
  -d '{"prompt": "Count to five.", "stream": true}'
```

Example event stream:

```text
data: {"content": "One"}

data: {"content": ", two"}

event: done
data: {"done": true}
```

| Event   | Payload example              | Meaning                          |
|---------|------------------------------|----------------------------------|
| (data)  | `{"content": "token"}`       | Incremental text chunk           |
| `done`  | `{"done": true}`             | Stream finished successfully     |
| `error` | `{"error": "..."}`           | Upstream/Ollama failure mid-stream |

Python client sketch:

```python
import json
import httpx

url = "http://127.0.0.1:8000/api/generate"
headers = {"X-API-Key": "dev-secret"}
payload = {"prompt": "Hello", "stream": True}

with httpx.stream("POST", url, json=payload, headers=headers, timeout=None) as r:
    r.raise_for_status()
    for line in r.iter_lines():
        if line.startswith("data: "):
            print(json.loads(line[6:]))
```

#### Feedback

```bash
curl -sS -X POST "http://127.0.0.1:8000/api/feedback" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-secret" \
  -d '{
    "prompt": "What is the capital of France?",
    "response": "Paris",
    "is_correct": true
  }'
```

### Generate error responses

`/api/generate` maps Ollama failures to clear HTTP status codes (instead of opaque 500s):

| Status | When |
|--------|------|
| **400** | Invalid request or model not found on the Ollama server |
| **502** | Ollama returned an upstream error |
| **503** | Ollama is unreachable (connection failure / timeout) |
| **500** | Unexpected internal error |

Errors are also written as structured log records (`error_type`, `model`, `http_status`, etc.).

## Deployment

1.  **Apply the Kubernetes manifests:**
    ```bash
    kubectl apply -f k8s/
    ```

2.  **(Optional) Enable API key via Secret:**
    ```bash
    kubectl create secret generic ai-agent-secrets --from-literal=api-key='YOUR_KEY'
    ```
    Then uncomment the `API_KEY` env block in `k8s/deployment.yaml` and re-apply.

3.  **(Optional) Restrict ingress with NetworkPolicy:**
    ```bash
    kubectl apply -f k8s/networkpolicy.yaml
    ```
    Prefer `ClusterIP` in `k8s/service.yaml` when relying on network policy alone.

4.  **Verify the deployment:**
    ```bash
    kubectl get pods
    kubectl get services
    ```
