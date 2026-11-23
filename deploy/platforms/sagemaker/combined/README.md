# SageMaker / AWS Marketplace: transcriber + Arc Tagger

The combined image ([`Dockerfile.combined`](../../../Dockerfile.combined)) is one `arc-serve-combined` binary: audio in,
the Arc Mini transcript (names, numbers, speakers, utterance tags) plus `arc_tags` from Arc Tagger (PII spans and
`redacted_text`, safety, intent, act, role, events) out. Text requests go straight to the tagger. Same container
contract as the [single package](../README.md).

| file | what |
|---|---|
| [`model-package.json`](model-package.json) | `create-model-package` input with a validation transform on `examples/example_synth.wav` |
| [`endpoint.cfn.yaml`](endpoint.cfn.yaml) | test endpoint (network isolation on) |
| [`../sample_notebook.ipynb`](../sample_notebook.ipynb) | buyer notebook (set `PRODUCT = "combined"`) |

## Image

Build context is the folder holding both checkouts (`arc_transcriber/`, `arc-tagger-c/`):

```bash
cd ..
HF_TOKEN=hf_... arc_transcriber/deploy/fetch_weights.sh mini tagger     # deploy/weights/mini, deploy/weights/arc-tagger
aws ecr create-repository --repository-name arc-transcriber-tagger --image-tag-mutability IMMUTABLE
docker buildx build --platform linux/amd64 -f arc_transcriber/deploy/Dockerfile.combined --target bundled \
  -t <account>.dkr.ecr.<region>.amazonaws.com/arc-transcriber-tagger:0.1.0 --push .
```

`--target bundled` puts both weight folders in the image; leave `ModelDataUrl` out. For `--target runtime`, put both
in one archive (`mini/` and `arc-tagger/`, the latter from `WhissleAI/arc-tagger` `onnx/`):

```bash
tar -czf model.tar.gz -C deploy/weights mini arc-tagger
aws s3 cp model.tar.gz s3://<bucket>/arc-combined/model.tar.gz
```

`ARC_MODEL_DIR` / `ARCTAG_MODEL_DIR` override the folders.

## Test and use

```bash
aws cloudformation deploy --stack-name arc-combined-test --template-file deploy/platforms/sagemaker/combined/endpoint.cfn.yaml \
  --parameter-overrides ImageUri=<image> ExecutionRoleArn=arn:aws:iam::<account>:role/<sagemaker-role>
aws sagemaker-runtime invoke-endpoint --endpoint-name arc-combined-test --content-type audio/wav \
  --custom-attributes 'domain=nba;tagger_domain=talk' --body fileb://examples/example_synth.wav out.json
```

JSON request with a names list and redaction:

```json
{"audio": "<base64 WAV>", "domain": "nba",
 "names": [{"name": "Theo Brandt", "kind": "person", "aliases": ["Brandt"]}],
 "tagger": {"domain": "talk", "redact": "mask"}}
```

- The response is `verbose_json` plus `arc_tags` (`spans`, `events`, `utterances`, `redacted_text`).
- `"tags": false` or `response_format: json` skips the tagger.
- Text only: `{"text": ...}`, `{"words": [...]}`, `{"segments": [...]}` or `text/plain` returns tagger output alone.
- Batch transform works as for the single package; transcripts in JSON Lines use `SplitType=Line`, `MultiRecord`.

## Model package

```bash
aws s3 cp examples/example_synth.wav s3://<bucket>/arc-combined/validation/example_synth.wav
aws sagemaker create-model-package --cli-input-json file://deploy/platforms/sagemaker/combined/model-package.json
aws sagemaker describe-model-package --model-package-name arc-transcriber-tagger   # wait for ValidationStatus Completed
```

Sizing: Arc Mini (about 180 MB per 30 s clip) plus about 72 MB per tagger handle. Default real-time `ml.c6i.xlarge`
(`ARC_WORKERS=2`, `ARCTAG_WORKERS=2`). Limits: English only, WAV or raw PCM16, CPU only, 6 MB real-time payloads,
clips up to 900 s. Redaction is not a compliance guarantee.

Not yet run in an AWS account: the validation job, instance availability per region, timings. Checklist:
[platforms/README.md](../../README.md#aws-marketplace-sagemaker-model-package).
