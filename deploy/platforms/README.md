# Platform adapters

One image, one contract ([`../openapi.yaml`](../openapi.yaml)); each platform only differs in where the weights come
from, which port and health route it probes, and how requests reach the container.

| platform | weights | port / health | requests | status |
|---|---|---|---|---|
| [Hugging Face Inference Endpoints](huggingface/README.md) | the model repo, mounted at `/repository` (runtime image) | 8080, `/v1/health/ready` | `/v1/audio/transcriptions`, `/v1/realtime` | ready to try |
| [Amazon SageMaker / AWS Marketplace](sagemaker/README.md) | `model.tar.gz` at `/opt/ml/model`, or inside the image (bundled) | 8080, `/ping` | `/invocations` (real-time, batch transform) | ready to try; Marketplace listing needs the checklist |
| [Snowflake SPCS](snowflake/README.md) | inside the image (bundled) | 8080, `/v1/health/ready` | service functions -> `/v1/snowflake/transcribe` | ready to try (amd64) |
| [NVIDIA NIM](nim/README.md) | | NIM-style health / models routes exist | | blocked: no GPU path |

The server never reaches the network, so every platform's network isolation works as is.
