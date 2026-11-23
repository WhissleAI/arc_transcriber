# Platforms and marketplace listings

One contract ([`../openapi.yaml`](../openapi.yaml)): the CPU image (libarc, Arc Mini) for every platform below except
NIM, which uses the CUDA image (Python, Arc Transcriber full). Platforms differ only in where the weights come from,
which health route they probe, and how requests reach the container. The server never reaches the network, so every
platform's network isolation works as is.

This page is the single place for marketplace status and listing checklists. Each folder has the how-to.

## Status

| platform | guide | weights | health | status |
|---|---|---|---|---|
| Hugging Face Inference Endpoints | [huggingface/](huggingface/README.md) | model repo mounted at `/repository` | `/v1/health/ready` | ready to try |
| Amazon SageMaker / AWS Marketplace | [sagemaker/](sagemaker/README.md), [combined](sagemaker/combined/README.md) | `model.tar.gz` or bundled | `/ping` | packages for Arc Mini and Mini + Tagger, [sample notebook](sagemaker/sample_notebook.ipynb); not yet validated in an AWS account; not listed |
| Snowflake (SPCS, Native App) | [snowflake/](snowflake/README.md), [native_app](snowflake/native_app/README.md) | bundled | `/v1/health/ready` | service ready to try (amd64); Native App package not yet run in an account; not listed |
| NVIDIA NIM | [nim/](nim/README.md) | Hugging Face cache or bundled | `/v1/health/ready` | GPU path available via the Python package ([CUDA image](../README.md#gpu-image-nvidia-cuda), untested on a GPU); NIM packaging not yet validated |

## Checklist for every paid listing

- [ ] **Licence**: code and weights are CC BY-NC 4.0. A paid or commercial listing needs Whissle's commercial EULA as
      its terms, not CC BY-NC.
- [ ] **Image** in the platform's registry, immutable version tags, scanned (critical / high findings cleared),
      rebuilt on base-image security updates.
- [ ] **No network at run time**: the conformance run checks readiness under `--network none`.
- [ ] **Listing copy**: description and model card (below), usage (the platform guide), sample input and output,
      support contact, limits.

## AWS Marketplace (SageMaker model package)

- [ ] Image in ECR in the seller account, in each listed region.
- [ ] Model package from [`sagemaker/model-package.json`](sagemaker/model-package.json) (or
      [`combined/`](sagemaker/combined/model-package.json)): content types, CPU instance types, `CertifyForMarketplace`,
      and a validation batch transform on `examples/example_synth.wav` in S3.
- [ ] Run the test endpoint and the validation transform in an AWS account before submitting.
- [ ] Price per instance-hour in the listing; SageMaker meters, the container does not.
- [ ] CPU instance types only (`ml.c6i.*`, `ml.m6i.*`); add Graviton only after testing the arm64 image.
- [ ] Buyer notebook: [`sagemaker/sample_notebook.ipynb`](sagemaker/sample_notebook.ipynb).

## Snowflake Marketplace (Native App)

Steps in [snowflake/native_app/](snowflake/native_app/README.md#provider-steps).

- [ ] First `snow app run` in a provider account: the service reaches `READY`, `services.transcribe` returns a
      transcript and `arc_tags`, and the app owns no external access integration.
- [ ] Confirm the parts written from Snowflake's docs but not yet run: manifest v2 keys and grant callback, the app
      stage mounted as a volume, compute pool limits on `CPU_X64_S`.
- [ ] Version registered, `DISTRIBUTION = EXTERNAL`, security review `APPROVED`, release directive set.
- [ ] Provider profile approved; listing with `quickstart.sql` as the usage example; tested privately from a second
      account.

## Hugging Face

- [ ] Check WebSocket support on the endpoint type before advertising `/v1/realtime`.
- [ ] Check `endpoint.json` field names against the current Inference Endpoints API.

## NVIDIA NIM

- [ ] Run the [CUDA image](../README.md#gpu-image-nvidia-cuda) on an NVIDIA GPU (L4 or A10G): readiness, a
      transcript matching the CPU run, latency and memory.
- [ ] Then the NIM steps in [nim/](nim/README.md).

## Listing copy (model card)

**Model**: Arc Mini (`WhissleAI/arc-v1.1`, `mini/`), served by libarc 1.4.1 (ABI 2). About 71 MB of weights.

**Does**: English transcripts, punctuated and cased, with inline tags and word times; names linked to a caller's list;
numbers with values (scores, durations, times, money, dates); device-directed, intent and dialogue-act read-outs;
sound and channel events; speakers and roles; n-best alternatives; streaming with partials and finals. The combined
image adds Arc Tagger: PII spans and redaction, safety, intent, act, affect.

**Does not**: languages other than English; names and numbers linking inside streams; Arc Transcriber (full);
compressed audio (send WAV or PCM).

**Checked**: libarc reproduces arc_audio's transcripts, tags, linked names and number values on the golden clip and
609 clip runs; the server's conformance run checks parity again on its own clips.

**Speed and memory** (Apple M3 Pro, 4 threads): real-time factor 0.026 (transcript only) to 0.046 (everything);
loads in 30-90 ms; about 180 MB for a 30 s clip. Raspberry Pi 5: RTF 0.24-0.37 on 4 threads.

**Accuracy**: on the [model card](https://huggingface.co/WhissleAI/arc-v1.1).

**Licence**: CC BY-NC 4.0 with the Whissle terms ([LICENSE](../../LICENSE)); Moonshine-streaming-tiny inside `mini/`
is MIT. Commercial use: ksingla@whissle.ai.
