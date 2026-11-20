# Hugging Face Inference Endpoints (custom container)

Inference Endpoints run a custom image with the selected model repository downloaded to `/repository`. arc-serve
looks there (and in `/repository/mini`), so the **runtime** image with the model repo `WhissleAI/arc-v1.1` serves Arc
Mini without baking weights into the image.

1. Push the image to a registry the endpoint can pull from (Docker Hub, ECR, ...):
   `docker buildx build --platform linux/amd64 -f deploy/Dockerfile --target runtime -t <registry>/arc-transcriber:0.1.0 --push .`
2. Create the endpoint (UI: "Deploy" -> Inference Endpoints -> "Custom container"; or the API with
   [`endpoint.json`](endpoint.json)):

| setting | value |
|---|---|
| model repository | `WhissleAI/arc-v1.1` (gated with automatic approval: the endpoint's owner must have accepted its terms) |
| container type | Custom, image `<registry>/arc-transcriber:0.1.0` |
| container port | `8080` |
| health route | `/v1/health/ready` |
| hardware | CPU (Arc Mini is CPU-only; a GPU instance would sit idle). 2-4 vCPU per replica is plenty for one or two concurrent clips. |
| environment | `ARC_WORKERS=2` (concurrent requests), optional `ARC_THREADS`, `ARC_DOMAIN`, `ARC_MODEL_ID=arc-mini` |
| task | custom (the server defines its own routes) |

3. Call it with the endpoint URL and a Hugging Face token:

```bash
curl -s https://<endpoint>.endpoints.huggingface.cloud/v1/audio/transcriptions -H "Authorization: Bearer $HF_TOKEN" \
     -F file=@clip.wav -F response_format=verbose_json
```

OpenAI SDKs: `OpenAI(base_url="https://<endpoint>.endpoints.huggingface.cloud/v1", api_key=HF_TOKEN)`.

Notes:
- The endpoint proxies HTTP; check WebSocket (`/v1/realtime`) support on your endpoint type before relying on it.
- Request size and timeout limits are the platform's; long files are better sent to SageMaker batch transform.
- A future Pro model: a private model repository in the same Arc Mini folder format, same image.
- `endpoint.json` follows the Inference Endpoints API's custom-image fields as documented at the time of writing;
  check field names against the current API reference before scripting it.
