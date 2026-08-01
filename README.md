# K3s Deployable AI Agent

This project provides a template for creating a k3s-deployable AI agent that wraps a local or remote Large Language Model (LLM). It includes a feedback mechanism to allow for fine-tuning and improvement of the agent's coding abilities.

## Features

*   **FastAPI Backend**: A Python-based backend using FastAPI to serve the agent's API.
*   **Ollama Integration**: Pre-configured to connect to an Ollama instance for LLM inference.
*   **Readiness checks**: Kubernetes readiness/startup probes verify the configured model is pulled before the agent receives traffic.
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

    Ensure Ollama is running and the model is pulled (see [Pre-pulling models](#pre-pulling-models)). Set `OLLAMA_HOST` if Ollama is not on the default URL.

## API Endpoints

*   `GET /`: Liveness check (process is up; does not require the model).
*   `GET /ready`: Readiness check (returns 200 only when `OLLAMA_MODEL` is available on Ollama; 503 otherwise).
*   `POST /api/generate`: Takes a prompt and returns a response from the LLM.
*   `POST /api/feedback`: Allows submitting feedback on the agent's responses to be used for future training.

## Deployment

1.  **Apply the Kubernetes manifests:**
    ```bash
    kubectl apply -f k8s/ollama-deployment.yaml
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    ```

2.  **Pre-pull the Ollama model** (required before the agent becomes Ready):
    ```bash
    kubectl apply -f k8s/ollama-model-pull-job.yaml
    ```
    Or pull manually:
    ```bash
    kubectl exec -it deployment/ollama-deployment -- ollama pull llama2
    ```
    Keep the model name in sync with `OLLAMA_MODEL` in `k8s/deployment.yaml` and the pull Job.

3.  **Verify the deployment:**
    ```bash
    kubectl get pods
    kubectl get services
    kubectl get job ollama-model-pull
    ```
    The AI agent pod remains `0/1 Ready` until `/ready` succeeds (model is present).

### Health probes

| Probe    | Path    | Purpose |
|----------|---------|---------|
| Liveness | `GET /` | Restart the container if the process is dead. Independent of Ollama. |
| Startup  | `GET /ready` | Delay readiness until the model is pulled (long failure window). |
| Readiness| `GET /ready` | Stop traffic if the model disappears or Ollama becomes unreachable. |

## Pre-pulling models

Ollama starts without any models. If the agent is marked Ready before the model exists, the first `/api/generate` call fails. This project avoids that by:

1. **Agent `/ready` endpoint** – calls `ollama.show(OLLAMA_MODEL)` and returns 503 until the model exists.
2. **Kubernetes probes** – `startupProbe` and `readinessProbe` hit `/ready` on the agent Deployment.
3. **Pull Job** – `k8s/ollama-model-pull-job.yaml` pulls the model into the Ollama service.

### Recommended cluster workflow

```bash
# 1. Start Ollama
kubectl apply -f k8s/ollama-deployment.yaml

# 2. Wait until Ollama is Ready
kubectl wait --for=condition=available deployment/ollama-deployment --timeout=120s

# 3. Pull the model (must match OLLAMA_MODEL on the agent)
kubectl apply -f k8s/ollama-model-pull-job.yaml
kubectl wait --for=condition=complete job/ollama-model-pull --timeout=600s

# 4. Deploy the agent (probes will pass once the model is listed)
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Local Ollama

```bash
ollama pull llama2
export OLLAMA_MODEL=llama2
# optional if not default:
# export OLLAMA_HOST=http://127.0.0.1:11434
uvicorn src.main:app --reload
curl http://127.0.0.1:8000/ready
```

### Changing the model

Update **all** of the following to the same name:

* `OLLAMA_MODEL` env in `k8s/deployment.yaml`
* `OLLAMA_MODEL` env in `k8s/ollama-model-pull-job.yaml`
* `ENV OLLAMA_MODEL` in the `Dockerfile` (image default)

Then re-run the pull Job (delete the old Job first if it already completed):

```bash
kubectl delete job ollama-model-pull --ignore-not-found
kubectl apply -f k8s/ollama-model-pull-job.yaml
```

For durable model storage across Ollama pod restarts, replace the `emptyDir` volume in `k8s/ollama-deployment.yaml` with a `PersistentVolumeClaim`.
