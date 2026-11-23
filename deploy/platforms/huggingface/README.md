# Hugging Face Inference Endpoints

Inference Endpoints run a custom image with the model repository downloaded to `/repository`. arc-serve looks there
(and in `/repository/mini`), so the **runtime** image serves Arc Mini with no weights baked in.

1. Push the image to a registry the endpoint can pull from:

   ```bash
   docker buildx build --platform linux/amd64 -f deploy/Dockerfile --target runtime -t <registry>/arc-transcriber:0.1.0 --push .
   ```

2. Create the endpoint ("Deploy" -> Inference Endpoints -> "Custom container", or the API with
   [`endpoint.json`](endpoint.json)):

   | setting | value |
   |---|---|
   | model repository | `WhissleAI/arc-v1.1` (gated: the endpoint's owner must accept its terms) |
   | container | custom, `<registry>/arc-transcriber:0.1.0`, port `8080`, health route `/v1/health/ready` |
   | hardware | CPU, 2-4 vCPU per replica (this image is libarc, CPU-only; for a GPU endpoint use `Dockerfile.cuda`) |
   | environment | `ARC_WORKERS=2`; optional `ARC_THREADS`, `ARC_DOMAIN`, `ARC_MODEL_ID` |

3. Call it:

   ```bash
   curl -s https://<endpoint>.endpoints.huggingface.cloud/v1/audio/transcriptions -H "Authorization: Bearer $HF_TOKEN" \
        -F file=@clip.wav -F response_format=verbose_json
   ```

Request size and timeout limits are the platform's; send long files to SageMaker batch transform. Open items:
[platforms/README.md](../README.md#hugging-face).
