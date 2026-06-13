# acestep-serverless

RunPod serverless worker for ACE-Step XL turbo (4B). Built on top of the
official [ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5) Dockerfile
(CUDA 12.8 + uv sync frozen).

- Image: `ghcr.io/micmaree/acestep-serverless:latest` (built by GH Actions)
- Models: live on attached RunPod network volume at `/runpod-volume/models/acestep/`
- Handler: `handler.py` spawns `acestep-api` once per worker, proxies each job
  through `/release_task` → `/query_result`, returns base64 MP3.

## Local test

```bash
docker build -t acestep-serverless .
docker run --gpus all -v ./checkpoints:/runpod-volume/models/acestep \
  -e RUNPOD_TEST_INPUT='{"input":{"prompt":"deep house","lyrics":"[instrumental]","audio_duration":60}}' \
  acestep-serverless
```
