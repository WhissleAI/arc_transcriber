# SageMaker / AWS Marketplace: Arc Transcriber + Tagger (combined)

The combined image ([`Dockerfile.combined`](../../../Dockerfile.combined)) is one `arc-serve-combined` binary: audio in,
the Arc Mini transcript (names, numbers, speakers, utterance tags) plus `arc_tags` from Arc Tagger (PII spans and
`redacted_text`, safety flags, intent, dialogue act, speaker role, per-word events) out. Text requests go straight to
the tagger. Same container contract as the single packages ([`../README.md`](../README.md)): `serve`, port 8080,
`/ping`, `/invocations`, `/execution-parameters`, no outbound calls, so it runs with `EnableNetworkIsolation: true`.

| file | what |
|---|---|
| [`model-package.json`](model-package.json) | `create-model-package` input: inference specification (image, content types, `ml.c6i` / `ml.m6i` real-time and transform types, as in the single packages) and a validation transform on `examples/example_synth.wav` |
| [`endpoint.cfn.yaml`](endpoint.cfn.yaml) | test endpoint: model (network isolation on), endpoint config, endpoint |
| [`../sample_notebook.ipynb`](../sample_notebook.ipynb) | buyer notebook: model package -> endpoint -> invoke -> batch transform -> clean up (set `PRODUCT = "combined"`) |

## Image

Build context is the folder holding both checkouts (`arc_transcriber/` and `arc-tagger-c/`):

```bash
cd ..
HF_TOKEN=hf_... arc_transcriber/deploy/fetch_weights.sh mini tagger      # deploy/weights/mini, deploy/weights/arc-tagger
aws ecr create-repository --repository-name arc-transcriber-tagger --image-tag-mutability IMMUTABLE
docker buildx build --platform linux/amd64 -f arc_transcriber/deploy/Dockerfile.combined --target bundled \
  -t <account>.dkr.ecr.<region>.amazonaws.com/arc-transcriber-tagger:0.1.0 --push .
```

`--target bundled` puts both weight folders in the image (`/opt/arc/model/mini`, `/opt/arc/model/arc-tagger`), which is
what a Marketplace buyer wants: one model package, no S3 step. Leave `ModelDataUrl` out.

## Model artifacts (`model.tar.gz`) for the runtime image

With `--target runtime`, SageMaker unpacks `ModelDataUrl` into `/opt/ml/model`; the server looks for Arc Mini in
`/opt/ml/model/mini` and Arc Tagger in `/opt/ml/model/arc-tagger` (`arctag_meta.json` marks it). Both weights go in one
archive:

```
model.tar.gz
├── mini/                         Arc Mini (WhissleAI/arc-v1.1, mini/)
│   ├── config.json
│   ├── tags.json
│   ├── model.q4.safetensors
│   ├── heads.safetensors
│   └── moonshine-streaming-tiny/
└── arc-tagger/                   Arc Tagger (WhissleAI/arc-tagger @ v0.4.1, onnx/)
    ├── arctag_meta.json
    ├── tokenizer.json
    ├── arctag.onnx
    ├── arctag_int8.onnx
    └── SHA256SUMS
```

```bash
tar -czf model.tar.gz -C deploy/weights mini arc-tagger
tar -tzf model.tar.gz | grep -E '^(mini/config.json|arc-tagger/arctag_meta.json)$'    # both must print
aws s3 cp model.tar.gz s3://<bucket>/arc-combined/model.tar.gz
```

The env vars `ARC_MODEL_DIR` / `ARCTAG_MODEL_DIR` override the folders if you lay the archive out differently.

## Test endpoint

```bash
aws cloudformation deploy --stack-name arc-combined-test --template-file deploy/platforms/sagemaker/combined/endpoint.cfn.yaml \
  --parameter-overrides ImageUri=<account>.dkr.ecr.<region>.amazonaws.com/arc-transcriber-tagger:0.1.0 \
                        ExecutionRoleArn=arn:aws:iam::<account>:role/<sagemaker-role>
```

## Usage: audio -> transcript + tags

```bash
# WAV in; verbose_json with arc_tags out. Tagger options ride in the custom attributes too.
aws sagemaker-runtime invoke-endpoint --endpoint-name arc-combined-test --content-type audio/wav \
  --custom-attributes 'domain=nba;tagger_domain=talk' --body fileb://examples/example_synth.wav out.json
jq '{text, names: [.names[]? | select(.state == "linked") | .entity], spans: [.arc_tags.spans[]? | {type, text}]}' out.json
```

JSON with base64 audio, a names list and tagger options (redaction):

```json
{
  "audio": "<base64 WAV>",
  "domain": "nba",
  "names": [{"name": "Theo Brandt", "kind": "person", "group": "Harbor City Hawks", "aliases": ["Brandt"]}],
  "tagger": {"domain": "talk", "redact": "mask"}
}
```

The response is the transcriber's `verbose_json` (`text`, `segments`, `words`, `names`, `numbers`, `events`,
`speakers`, `diarization`, `usage`) plus `arc_tags`: the tagger's output on the transcript's words (`spans`, `events`,
`utterances`, `redacted_text` when redaction is on). Speakers from diarization feed the tagger's role read-out.
`"tags": false` skips the tagger for a request; `response_format: json` (text only) skips it too.

Text only (no audio): `Content-Type: application/json` with `{"text": ...}`, `{"words": [...]}` or `{"segments": [...]}`
(a transcriber `verbose_json` works as `segments`), or `text/plain`, returns the tagger output alone.

## Batch transform

As the single package: one WAV per S3 object -> one JSON per object.

```bash
aws sagemaker create-transform-job --transform-job-name arc-combined-batch-1 --model-name <model> \
  --transform-input DataSource={S3DataSource={S3DataType=S3Prefix,S3Uri=s3://<bucket>/calls/}},ContentType=audio/wav,SplitType=None \
  --transform-output S3OutputPath=s3://<bucket>/calls-out/,Accept=application/json \
  --transform-resources InstanceType=ml.c6i.2xlarge,InstanceCount=1 \
  --batch-strategy SingleRecord --max-payload-in-mb 100 --max-concurrent-transforms 2
```

Transcripts already in JSON Lines (`{"text": ...}` per line): `ContentType=application/jsonlines,SplitType=Line`,
`--batch-strategy MultiRecord`, `Accept=application/jsonlines,AssembleWith=Line`.

## Model package

Fill `<account>`, `<region>`, the role and the bucket, upload `examples/example_synth.wav` to the validation prefix, then:

```bash
aws s3 cp examples/example_synth.wav s3://<bucket>/arc-combined/validation/example_synth.wav
aws sagemaker create-model-package --cli-input-json file://deploy/platforms/sagemaker/combined/model-package.json
aws sagemaker describe-model-package --model-package-name arc-transcriber-tagger   # wait for ValidationStatus Completed
```

Sizing: as Arc Mini (about 180 MB per 30 s clip, about 1 GB at 8 minutes) plus about 72 MB per tagger handle (fp32).
Default real-time `ml.c6i.xlarge` (`ARC_WORKERS=2`, `ARCTAG_WORKERS=2`); batch `ml.c6i.2xlarge` or `ml.m6i.2xlarge` for
long files. Limits are the union of both products: English only, WAV / raw PCM16 input, CPU only, real-time payloads at
most 6 MB with a 60 s timeout, clips up to 900 s (`ARC_MAX_AUDIO_S`); redaction is not a compliance guarantee.

Not verified here (no AWS account was used): the validation job, the instance types' availability per region, and
timings on these instances. Run the test endpoint and the validation transform before submitting.
