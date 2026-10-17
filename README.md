# K3s Deployable AI Agent

This project provides a template for creating a k3s-deployable AI agent that wraps a local or remote Large Language Model (LLM). It includes a feedback mechanism to allow for fine-tuning and improvement of the agent's coding abilities, plus a **tool-using agent loop** that can checkout code, run tests, and produce patches inside an isolated sandbox.

## Features

*   **FastAPI Backend**: A Python-based backend using FastAPI to serve the agent's API.
*   **API authentication**: `POST /api/resolve` and `POST /api/generate` require `API_KEY` (`X-API-Key` or `Authorization: Bearer`). Requests without a matching key are rejected, including when `API_KEY` is unset.
*   **Ollama Integration**: Pre-configured to connect to an Ollama instance for LLM inference.
*   **Tool-using agent loop**: Sandboxed `run_command`, `read_file`, and `write_file` tools with an iterative resolve flow (checkout → edit → test → patch).
*   **Issue resolution API**: Synchronous or background job mode via `POST /api/resolve`, with resolve jobs persisted in SQLite and recovered after restarts.
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

2.  **Set the API key** (required for `/api/resolve` and `/api/generate`):
    ```bash
    export API_KEY=dev-secret
    ```
    Do not commit this value. When `API_KEY` is unset or blank, those two routes return `401`.

3.  **Run the application:**
    ```bash
    uvicorn src.main:app --reload
    ```
    The application will be available at `http://127.0.0.1:8000`.

4.  **Run tests:**
    ```bash
    pytest
    ```

## API Endpoints

*   `GET /`: Health check. Public, so Kubernetes probes keep working.
*   `GET /ready`: Readiness check. Public.
*   `GET /api/tools`: List the minimal tool interface and sandbox safety notes.
*   `POST /api/generate`: Takes a prompt and returns a response from the LLM. **Requires API key.**
*   `POST /api/feedback`: Allows submitting feedback on the agent's responses to be used for future training.
*   `POST /api/resolve`: Run the tool-using agent loop against an issue (sync or async job). **Requires API key.**
*   `GET /api/jobs`: List persisted resolve jobs.
*   `GET /api/jobs/{job_id}`: Fetch status/result for a resolve job.

### Authentication

`POST /api/resolve` and `POST /api/generate` require the `API_KEY` environment variable. Send either header:

*   `X-API-Key: <API_KEY>`
*   `Authorization: Bearer <API_KEY>`

If `X-API-Key` is present it is the credential that is checked. Query parameters are ignored. A missing key, a wrong key, and a server with no `API_KEY` configured all receive `401` with `WWW-Authenticate: Bearer`. The comparison is constant-time. `/`, `/ready`, feedback, tool listing, and job status stay reachable without the key so probes and local inspection keep working.

```bash
curl -sS -X POST "http://127.0.0.1:8000/api/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-secret" \
  -d '{"prompt": "Explain Kubernetes in one sentence.", "model": "llama2"}'
```

### Resolve example

Synchronous (waits for the agent to finish):

```bash
curl -s -X POST http://127.0.0.1:8000/api/resolve \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev-secret" \
  -d '{
    "issue": "Add a function add(a, b) in mathutil.py and make a simple test pass",
    "max_steps": 10,
    "test_command": "pytest",
    "async_mode": false
  }'
```

Background job mode:

```bash
curl -s -X POST http://127.0.0.1:8000/api/resolve \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-secret" \
  -d '{"issue":"Fix the failing unit test","repo_url":"https://github.com/example/repo.git","async_mode":true}'

# then poll
curl -s http://127.0.0.1:8000/api/jobs/<job_id>
```

Optional body fields: `repo_url`, `git_ref`, `max_steps`, `test_command`, `model`, `async_mode`.

## Tool-using agent loop

The resolve flow:

1. Create a **sandbox workspace** (temp dir under the sandbox root; optional `git clone` of `repo_url`).
2. Establish a git baseline so later changes can be exported as a patch.
3. Loop (up to `max_steps`): ask the LLM for the next JSON action → execute a tool → feed the observation back.
4. On `finish` (or max steps), collect `git diff` and return status, summary, step log, and patch.

### Minimal tool interface

| Tool | Purpose |
|------|---------|
| `run_command` | Run a shell command with `cwd` = workspace and a hard timeout |
| `read_file` | Read a UTF-8 text file **inside** the workspace |
| `write_file` | Create/overwrite a UTF-8 text file **inside** the workspace |

Path arguments are confined to the workspace; `../` traversal is rejected.

## Sandbox & safety defaults

**There is no host mount by default.** Workspaces are created under:

* `AGENT_SANDBOX_ROOT` if set, otherwise
* `<system temp>/swe-agent-sandboxes/`

Important environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `API_KEY` | unset (protected routes reject) | Shared secret required on `/api/resolve` and `/api/generate` |
| `AGENT_SANDBOX_ROOT` | system temp subdir | Root directory for all agent workspaces |
| `JOB_STORAGE_PATH` | `data/jobs.db` | SQLite database for durable resolve job requests, statuses, and results |
| `AGENT_ALLOW_HOST_MOUNT` | `false` | If `true`, allow binding an arbitrary host path as a workspace |
| `AGENT_COMMAND_TIMEOUT` | `60` | Default command timeout (seconds) |
| `AGENT_MAX_OUTPUT_BYTES` | `262144` | Truncate tool stdout/stderr to this many bytes |
| `OLLAMA_MODEL` / `OLLAMA_HOST` | `llama2` / ollama default | LLM configuration |

### Sandbox notes (read carefully)

* Isolation is **application-level** (path confinement + timeouts + minimal env). It is **not** a full OS sandbox (no seccomp/user namespaces by itself).
* Prefer deploying the agent **in Kubernetes without `hostPath` volumes** so the container filesystem boundary is the primary isolation layer.
* Do **not** set `AGENT_ALLOW_HOST_MOUNT=true` in shared/multi-tenant environments.
* Child processes inherit a reduced environment and set `SWE_AGENT_SANDBOX=1`.
* Resolve jobs are stored in SQLite at `JOB_STORAGE_PATH` with a private database file. The Kubernetes Deployment mounts the `ai-agent-data` PVC at `/app/data` so jobs survive pod restarts, and uses the `Recreate` strategy to keep pods from overlapping as writers. A process atomically claims queued jobs and renews its owner lease while working. Startup requeues only jobs whose heartbeat lease expired; a live process keeps its job. If a process dies after making partial changes, that job is retried from scratch, so resolve execution is at-least-once across crashes. Credentials in repository URL userinfo are stripped before the URL is persisted or returned.

## Deployment

1.  **Create the API key secret** (the Deployment will not start without it):
    ```bash
    kubectl create secret generic ai-agent-secrets \
      --from-literal=api-key="$(openssl rand -hex 32)"
    ```
    The value is injected as `API_KEY`. Do not store it in the image or the git repo.

2.  **Apply the Kubernetes manifests:**
    ```bash
    kubectl apply -f k8s/
    ```
    `k8s/service.yaml` is a LoadBalancer. `/api/resolve` and `/api/generate` still require the key. For in-cluster clients only, set the Service to `ClusterIP` and apply `k8s/networkpolicy.yaml`. That policy does not replace the key check.

3.  **Verify the deployment:**
    ```bash
    kubectl get pods
    kubectl get services
    ```

The sample `k8s/deployment.yaml` does **not** mount host paths into the agent container, matching the safety default of no host mount.
