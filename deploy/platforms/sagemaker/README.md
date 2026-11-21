# Amazon SageMaker and AWS Marketplace

arc-serve implements SageMaker's inference container contract: started as `docker run IMAGE serve`, listens on 8080,
`GET /ping` (200 once the weights are loaded), `POST /invocations`, and `GET /execution-parameters` for batch
transform. It reads the model from `/opt/ml/model` and makes no network calls, so it runs with
`EnableNetworkIsolation: true` (required for Marketplace model packages).

Also here: [`sample_notebook.ipynb`](sample_notebook.ipynb), the buyer notebook Marketplace asks for (subscribe, model
from the model package ARN, real-time endpoint, invoke with a WAV, batch transform, clean up), and
[`combined/`](combined/README.md), the model package for the transcriber + tagger image (`Dockerfile.combined`).

## Model artifacts (`model.tar.gz`)

With the **runtime** image, SageMaker downloads `ModelDataUrl` and unpacks it into `/opt/ml/model`. Either layout works:

```
model.tar.gz                      model.tar.gz
├── config.json                   └── mini/
├── tags.json                         ├── config.json
├── model.q4.safetensors              ├── ...
├── heads.safetensors                 └── moonshine-streaming-tiny/
└── moonshine-streaming-tiny/
```

```bash
HF_TOKEN=hf_... deploy/fetch_weights.sh mini
tar -czf model.tar.gz -C deploy/weights mini
aws s3 cp model.tar.gz s3://<bucket>/arc/mini/model.tar.gz
```

With the **bundled** image (`--target bundled`) the weights are in the image at `/opt/arc/model`; leave
`ModelDataUrl` out. Marketplace sellers usually bundle, so buyers deploy one model package with no S3 step.

## Image

```bash
aws ecr create-repository --repository-name arc-transcriber
docker buildx build --platform linux/amd64 -f deploy/Dockerfile --target bundled \
  -t <account>.dkr.ecr.<region>.amazonaws.com/arc-transcriber:0.1.0-mini --push .
```

amd64 for `ml.c*/m*` instances; add `linux/arm64` for Graviton (`ml.c7g`, `ml.m7g`) real-time endpoints.

## A test endpoint

CloudFormation: [`endpoint.cfn.yaml`](endpoint.cfn.yaml) (model with network isolation, endpoint config, endpoint).

```bash
aws cloudformation deploy --stack-name arc-mini-test --template-file deploy/platforms/sagemaker/endpoint.cfn.yaml \
  --parameter-overrides ImageUri=<account>.dkr.ecr.<region>.amazonaws.com/arc-transcriber:0.1.0-mini \
                        ExecutionRoleArn=arn:aws:iam::<account>:role/<sagemaker-role>
aws sagemaker-runtime invoke-endpoint --endpoint-name arc-mini-test --content-type audio/wav \
  --custom-attributes 'domain=nba' --body fileb://examples/example_synth.wav out.json && cat out.json
```

Or with the CLI only:

```bash
aws sagemaker create-model --model-name arc-mini --execution-role-arn <role> --enable-network-isolation \
  --primary-container Image=<image>,Environment={ARC_WORKERS=2}            # + ,ModelDataUrl=s3://... for the runtime image
aws sagemaker create-endpoint-config --endpoint-config-name arc-mini \
  --production-variants VariantName=AllTraffic,ModelName=arc-mini,InitialInstanceCount=1,InstanceType=ml.c6i.xlarge
aws sagemaker create-endpoint --endpoint-name arc-mini --endpoint-config-name arc-mini
```

`/invocations` routing (see `../../openapi.yaml`): `audio/wav` / `audio/pcm` / `multipart/form-data` transcribe
(`verbose_json`; `Accept: text/plain` for text); `application/json` with base64 `audio` (and options) transcribes; a
JSON batch (`{"instances": [...]}`) returns `{"results": [...]}`; `application/jsonlines` is one request per line.
Options also come in `X-Amzn-SageMaker-Custom-Attributes` (`domain=nba;response_format=text`).

Limits to plan around: real-time `InvokeEndpoint` payloads are at most 6 MB (about 3 minutes of 16 kHz PCM16 WAV) with
a 60 s timeout; use asynchronous inference or batch transform for longer audio.

## Batch transform

```bash
aws sagemaker create-transform-job --transform-job-name arc-mini-batch-1 --model-name arc-mini \
  --transform-input DataSource={S3DataSource={S3DataType=S3Prefix,S3Uri=s3://<bucket>/calls/}},ContentType=audio/wav,SplitType=None \
  --transform-output S3OutputPath=s3://<bucket>/calls-out/,Accept=application/json \
  --transform-resources InstanceType=ml.c6i.2xlarge,InstanceCount=1 \
  --batch-strategy SingleRecord --max-payload-in-mb 100 --max-concurrent-transforms 2
```

One WAV per object -> one `verbose_json` per object. For many small base64 clips in JSON lines files use
`ContentType=application/jsonlines,SplitType=Line` and `--batch-strategy MultiRecord`. `GET /execution-parameters`
gives SageMaker these defaults (`MaxConcurrentTransforms` = `ARC_WORKERS`).

## AWS Marketplace model package checklist

- [ ] **Licence first**: the code and weights are CC BY-NC 4.0. A paid listing needs Whissle's commercial EULA as the
      listing's terms (Whissle owns the rights; the CC BY-NC grant is not a commercial licence for buyers).
- [ ] **Image in ECR** in the seller account, in each region the listing supports; immutable tags per version.
- [ ] **Image scan**: enable ECR enhanced scanning (Inspector) and clear critical / high findings before submitting;
      Marketplace scans again. The runtime base is distroless (glibc only, no shell or package manager) to keep this
      short. Rebuild on base-image security updates.
- [ ] **No network at run time**: the container must work with network isolation. arc-serve makes no outbound calls,
      downloads nothing, and does not phone home (conformance `run.sh` checks readiness under `--network none`).
- [ ] **Model package**: `aws sagemaker create-model-package` with [`model-package.json`](model-package.json):
      inference specification (image, content types `audio/wav`, `audio/pcm`, `application/json`, `application/jsonlines`;
      response types `application/json`, `text/plain`; real-time and transform instance types), `CertifyForMarketplace`,
      and a **validation specification**: a batch transform job SageMaker runs on sample input in S3 to prove the package
      works (use `examples/example_synth.wav`).
- [ ] **Metering by instance-hour**: set an hourly price per instance type in the Marketplace listing. SageMaker and
      Marketplace meter the buyer's instance-hours; the container does no metering (it could not reach a metering API
      under network isolation anyway).
- [ ] **Instance types**: list only CPU types (Arc Mini has no GPU path), e.g. `ml.c6i.large`-`ml.c6i.4xlarge`,
      `ml.m6i.*`; add Graviton types only after testing the arm64 image on them.
- [ ] **Listing content**: description and the model card (`../../README.md`), usage instructions (this file), sample
      input and output, the sample notebook ([`sample_notebook.ipynb`](sample_notebook.ipynb): model package, endpoint,
      invoke, batch transform, clean up), support contact, and the known
      limits (English only, WAV / PCM input, 6 MB real-time payloads).
- [ ] **Pro tier later**: a second model package from the same Dockerfile with the Pro weights bundled and
      `MODEL_ID=arc-pro`, listed as a separate product or version.
