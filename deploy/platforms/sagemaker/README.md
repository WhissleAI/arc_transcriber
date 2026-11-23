# Amazon SageMaker and AWS Marketplace

arc-serve implements SageMaker's inference container contract: `docker run IMAGE serve`, port 8080, `GET /ping`,
`POST /invocations`, `GET /execution-parameters`. It reads the model from `/opt/ml/model` and makes no network calls,
so it runs with `EnableNetworkIsolation: true` (required for Marketplace).

| file | what |
|---|---|
| [`endpoint.cfn.yaml`](endpoint.cfn.yaml) | test endpoint: model (network isolation), endpoint config, endpoint |
| [`model-package.json`](model-package.json) | `create-model-package` input for the Marketplace |
| [`sample_notebook.ipynb`](sample_notebook.ipynb) | buyer notebook: subscribe, endpoint, invoke, batch transform, clean up |
| [`combined/`](combined/README.md) | the same for the transcriber + Arc Tagger image |

Marketplace checklist and status: [platforms/README.md](../README.md#aws-marketplace-sagemaker-model-package).

## Image and weights

```bash
HF_TOKEN=hf_... deploy/fetch_weights.sh mini
aws ecr create-repository --repository-name arc-transcriber
docker buildx build --platform linux/amd64 -f deploy/Dockerfile --target bundled \
  -t <account>.dkr.ecr.<region>.amazonaws.com/arc-transcriber:0.1.0-mini --push .
```

With `--target bundled` the weights are in the image: leave `ModelDataUrl` out (what Marketplace sellers usually do).
With `--target runtime`, upload the weights instead; files at the archive root or in a `mini/` folder both work:

```bash
tar -czf model.tar.gz -C deploy/weights mini
aws s3 cp model.tar.gz s3://<bucket>/arc/mini/model.tar.gz
```

Use amd64 for `ml.c*` / `ml.m*` instances; add `linux/arm64` for Graviton.

## Test endpoint

```bash
aws cloudformation deploy --stack-name arc-mini-test --template-file deploy/platforms/sagemaker/endpoint.cfn.yaml \
  --parameter-overrides ImageUri=<image> ExecutionRoleArn=arn:aws:iam::<account>:role/<sagemaker-role>
aws sagemaker-runtime invoke-endpoint --endpoint-name arc-mini-test --content-type audio/wav \
  --custom-attributes 'domain=nba' --body fileb://examples/example_synth.wav out.json && cat out.json
```

`/invocations` accepts `audio/wav`, `audio/pcm` and `multipart/form-data` (returns `verbose_json`; `Accept: text/plain`
for text), `application/json` with base64 `audio`, a JSON batch `{"instances": [...]}`, and `application/jsonlines`.
Options go in `X-Amzn-SageMaker-Custom-Attributes`.

Real-time payloads are at most 6 MB (about 3 minutes of 16 kHz WAV) with a 60 s timeout. Use asynchronous inference
or batch transform for longer audio.

## Batch transform

```bash
aws sagemaker create-transform-job --transform-job-name arc-mini-batch-1 --model-name arc-mini \
  --transform-input DataSource={S3DataSource={S3DataType=S3Prefix,S3Uri=s3://<bucket>/calls/}},ContentType=audio/wav,SplitType=None \
  --transform-output S3OutputPath=s3://<bucket>/calls-out/,Accept=application/json \
  --transform-resources InstanceType=ml.c6i.2xlarge,InstanceCount=1 \
  --batch-strategy SingleRecord --max-payload-in-mb 100 --max-concurrent-transforms 2
```

One WAV per object gives one `verbose_json` per object. For many small base64 clips, use JSON Lines with
`ContentType=application/jsonlines,SplitType=Line` and `--batch-strategy MultiRecord`.
