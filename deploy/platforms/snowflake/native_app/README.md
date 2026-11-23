# Snowflake Native App (Arc Mini + Arc Tagger)

Wraps the [SPCS service](../README.md) as a **Native App** for the Snowflake Marketplace. The consumer installs it;
the app creates a compute pool, a service running the **combined** image
([`Dockerfile.combined`](../../../Dockerfile.combined), bundled weights) and SQL functions, all in the consumer's
account. No external access integration, no public endpoint.

| file | what |
|---|---|
| [`manifest.yml`](manifest.yml) | manifest v2: image, `CREATE COMPUTE POOL` + `CREATE WAREHOUSE` privileges, callbacks |
| [`setup_script.sql`](setup_script.sql) | roles `app_user` / `app_admin`, schema `core`, stage `data.audio`, `core.start_app()` |
| [`services/arc.yaml`](services/arc.yaml) | service spec (combined image) |
| [`snowflake.yml`](snowflake.yml) | Snowflake CLI project (package `arc_app_pkg`, app `arc_app`) |
| [`app_readme.md`](app_readme.md) | consumer readme, shipped as the app's `README.md` |
| [`quickstart.sql`](quickstart.sql) | consumer quick start |

| SQL object | does |
|---|---|
| `services.transcribe(stage_path[, options])` | transcript + `arc_tags`; `stage_path` relative to `@data.audio` |
| `services.tags(text)`, `services.tags_opts(text, options)` | Arc Tagger on text |
| `services.transcribe_raw(audio)`, `services.transcribe_opts(audio, options)` | raw service functions (path or base64) |
| `core.start_app()`, `service_status()`, `service_logs(n)`, `suspend_app()`, `resume_app()` | lifecycle (`app_admin`) |
| `core.refresh_audio()` | refresh the stage directory after uploads (`app_user`) |

## 1. Test the container locally

From the folder holding both checkouts:

```bash
HF_TOKEN=hf_... arc_transcriber/deploy/fetch_weights.sh mini tagger
docker buildx build --platform linux/amd64 -f arc_transcriber/deploy/Dockerfile.combined --target bundled \
  -t arc-transcriber-tagger:0.1.0 --load .
mkdir -p /tmp/stage/demo && cp arc_transcriber/examples/example_synth.wav /tmp/stage/demo/
docker run --rm -d --name arc-na -p 8080:8080 -e ARC_FILE_ROOT=/stage -v /tmp/stage:/stage:ro arc-transcriber-tagger:0.1.0 serve
curl -s localhost:8080/v1/snowflake/transcribe -H 'content-type: application/json' \
  -d '{"data": [[0, "/stage/demo/example_synth.wav"], [1, "/stage/../etc/passwd"]]}' | head -c 600
docker rm -f arc-na
```

Row 1 must come back as `{"error": ...}` without failing row 0.

## 2. Run the app in your own account

Needs the Snowflake CLI (`snow`) and a provider account with SPCS.

```bash
snow sql -q "CREATE DATABASE IF NOT EXISTS arc_db; CREATE SCHEMA IF NOT EXISTS arc_db.arc;
             CREATE IMAGE REPOSITORY IF NOT EXISTS arc_db.arc.images;"
snow spcs image-registry login
REPO=$(snow spcs image-repository url arc_db.arc.images | tail -1)
docker tag arc-transcriber-tagger:0.1.0 "$REPO/arc-transcriber-tagger:0.1.0" && docker push "$REPO/arc-transcriber-tagger:0.1.0"

cd arc_transcriber/deploy/platforms/snowflake/native_app
snow app run
snow sql -q "GRANT CREATE COMPUTE POOL ON ACCOUNT TO APPLICATION arc_app;
             GRANT CREATE WAREHOUSE ON ACCOUNT TO APPLICATION arc_app;
             CALL arc_app.core.start_app();"
```

Then walk [`quickstart.sql`](quickstart.sql). `snow app teardown --cascade` removes the app and what it created.

## Provider steps

1. **Version**: `snow app version create v0_1_0 --skip-git-check`. A new image needs a new tag in `manifest.yml` and
   `services/arc.yaml`, and a new version or patch.
2. **Security review**: `ALTER APPLICATION PACKAGE arc_app_pkg SET DISTRIBUTION = EXTERNAL;` then wait for
   `review_status = APPROVED` in `SHOW VERSIONS IN APPLICATION PACKAGE arc_app_pkg`.
3. **Release directive**: `ALTER APPLICATION PACKAGE arc_app_pkg SET DEFAULT RELEASE DIRECTIVE VERSION = v0_1_0 PATCH = 0;`
4. **Provider profile** in Snowsight (Data Products -> Provider Studio).
5. **Listing**: attach `arc_app_pkg`, add the listing copy and `quickstart.sql`, choose regions and pricing (paid needs
   Whissle's commercial EULA), submit.
6. **Test as a consumer** from a second account before going public.

Status and open items: [platforms/README.md](../../README.md#snowflake-marketplace-native-app). The YAML and SQL were
checked offline (parse, balance, paths, routes); nothing has been run in a Snowflake account yet. `snow --help` is
authoritative for CLI flags.
