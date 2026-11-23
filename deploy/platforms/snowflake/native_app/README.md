# Snowflake Native App package (Arc Mini + Arc Tagger on SPCS)

This folder turns the plain SPCS service in [`..`](../README.md) into a **Native App** a provider can list on the
Snowflake Marketplace. The consumer installs it; the app creates a compute pool, a service running the **combined**
image ([`Dockerfile.combined`](../../../Dockerfile.combined), bundled weights) and SQL functions, all in the consumer's
account. It requests no external access integration and no public endpoint, so it makes no outbound calls.

| file | what |
|---|---|
| [`manifest.yml`](manifest.yml) | manifest v2: version, container image, `CREATE COMPUTE POOL` + `CREATE WAREHOUSE` privileges, grant callback, version initializer |
| [`setup_script.sql`](setup_script.sql) | application roles `app_user` / `app_admin`, versioned schema `core` (procedures), stage `data.audio`, `core.start_app()` (pool, warehouse, service, functions, grants) |
| [`services/arc.yaml`](services/arc.yaml) | service spec: [`../service.yaml`](../service.yaml) with the combined image, the app's own stage and `ARCTAG_WORKERS` |
| [`snowflake.yml`](snowflake.yml) | Snowflake CLI project definition (application package `arc_app_pkg`, application `arc_app`) |
| [`app_readme.md`](app_readme.md) | consumer-facing readme (shipped as the app's `README.md`) |
| [`quickstart.sql`](quickstart.sql) | consumer quick start, SQL only |

SQL surface in the consumer account (functions in the `services` schema, because they are bound to the stateful
service; procedures in the versioned `core` schema):

| object | route / does |
|---|---|
| `services.transcribe(stage_path)`, `services.transcribe(stage_path, options)` | `/v1/snowflake/transcribe`; `stage_path` relative to `@data.audio`, turned into `/stage/...` by `services.to_stage_path` |
| `services.tags(text)`, `services.tags_opts(text, options)` | `/v1/snowflake/tags` |
| `services.transcribe_raw(audio)`, `services.transcribe_opts(audio, options)` | the raw service functions (`/stage/...` path or base64) |
| `core.start_app()`, `core.service_status()`, `core.service_logs(n)`, `core.suspend_app()`, `core.resume_app()` | lifecycle (`app_admin`) |
| `core.refresh_audio()` | refresh `DIRECTORY(@data.audio)` after uploads (`app_user`) |

## 1. Test the container locally (no Snowflake account)

The service functions only POST `{"data": [[row, arg, ...]]}` to the container, so the app's whole data path can be
exercised with Docker. Build the combined bundled image from the folder holding both checkouts:

```bash
cd ..   # parent of arc_transcriber/ and arc-tagger-c/
HF_TOKEN=hf_... arc_transcriber/deploy/fetch_weights.sh mini tagger
nice -n 19 docker buildx build --platform linux/amd64 -f arc_transcriber/deploy/Dockerfile.combined --target bundled \
  -t arc-transcriber-tagger:0.1.0 --load .

mkdir -p /tmp/stage/demo && cp arc_transcriber/examples/example_synth.wav /tmp/stage/demo/
docker run --rm -d --name arc-na -p 8080:8080 -e ARC_FILE_ROOT=/stage -v /tmp/stage:/stage:ro \
  arc-transcriber-tagger:0.1.0 serve
# (no egress: `deploy/conformance/run.sh ... --combined ...` already checks readiness under --network none)
curl -s localhost:8080/v1/health/ready
curl -s localhost:8080/v1/snowflake/transcribe -H 'content-type: application/json' \
  -d '{"data": [[0, "/stage/demo/example_synth.wav"], [1, "/stage/../etc/passwd"]]}' | head -c 600; echo
curl -s localhost:8080/v1/snowflake/tags -H 'content-type: application/json' \
  -d '{"data": [[0, "Hi, this is Maria Lopez. My number is 555 867 5309.", "{\"domain\": \"talk\", \"redact\": \"mask\"}"]]}'
docker rm -f arc-na
```

Row 1 must come back as `{"error": ...}` (outside the stage) without failing row 0.

## 2. Build and run the app in your own account (`snow app run`)

Needs the [Snowflake CLI](https://docs.snowflake.com/en/developer-guide/snowflake-cli/index) (`snow`) with a
connection to a provider account where SPCS is available, and a role that can create databases, image repositories and
application packages.

```bash
# Image repository for the provider (path must match manifest.yml and services/arc.yaml)
snow sql -q "CREATE DATABASE IF NOT EXISTS arc_db; CREATE SCHEMA IF NOT EXISTS arc_db.arc;
             CREATE IMAGE REPOSITORY IF NOT EXISTS arc_db.arc.images;"
snow spcs image-registry login
REPO=$(snow spcs image-repository url arc_db.arc.images | tail -1)    # <org>-<acct>.registry.snowflakecomputing.com/arc_db/arc/images
docker tag arc-transcriber-tagger:0.1.0 "$REPO/arc-transcriber-tagger:0.1.0"
docker push "$REPO/arc-transcriber-tagger:0.1.0"

cd arc_transcriber/deploy/platforms/snowflake/native_app
snow app run                     # creates arc_app_pkg (stage + files) and installs arc_app in the same account
snow sql -q "GRANT CREATE COMPUTE POOL ON ACCOUNT TO APPLICATION arc_app;
             GRANT CREATE WAREHOUSE ON ACCOUNT TO APPLICATION arc_app;
             CALL arc_app.core.start_app();"
```

Then walk [`quickstart.sql`](quickstart.sql) (with `PUT` from `snow sql` or SnowSQL). Iterate with `snow app run`
again (redeploys files and upgrades the dev install); `snow app teardown --cascade` drops the app, package and the
objects the app created (compute pool, warehouse).

Things to check on the first run (these could not be verified offline, see the end):
- `core.service_status()` reaches `READY` and `service_logs` shows `arc-serve` listening with the tagger loaded.
- `services.transcribe('demo/example_synth.wav')` returns a transcript and `arc_tags`.
- The account shows **no** external access integration owned by the app (`SHOW EXTERNAL ACCESS INTEGRATIONS`).

## 3. Provider steps: application package, version, listing

1. **Versions.** `snow app version create v0_1_0 --skip-git-check` (adds version `V0_1_0` with patch 0 to
   `arc_app_pkg`), or in SQL `ALTER APPLICATION PACKAGE arc_app_pkg REGISTER VERSION v0_1_0 USING '@<stage>'`.
   A new image needs a new tag in `manifest.yml` + `services/arc.yaml`, and a new version or patch.
2. **External distribution + security review.** `ALTER APPLICATION PACKAGE arc_app_pkg SET DISTRIBUTION = EXTERNAL;`
   Snowflake then scans each version / patch (including the container image). Wait for
   `SHOW VERSIONS IN APPLICATION PACKAGE arc_app_pkg` to report `review_status = APPROVED`.
3. **Release directive.** `ALTER APPLICATION PACKAGE arc_app_pkg SET DEFAULT RELEASE DIRECTIVE VERSION = v0_1_0 PATCH = 0;`
   (or `snow app release-directive set default --version v0_1_0 --patch 0`; packages with release channels use
   `snow app release-channel ...` instead).
4. **Provider profile.** In Snowsight, *Data Products → Provider Studio*: accept the provider terms and create the
   provider profile (Marketplace listings need an approved profile).
5. **Listing.** *Provider Studio → Listings → + Create listing*: Marketplace, attach `arc_app_pkg` as the data product,
   paste the listing copy (title, description, use cases, limits, support contact), add `quickstart.sql` as the usage
   example, choose regions (cross-region auto-fulfilment must be on for regions other than the package's), pick free /
   limited trial / paid (paid needs Whissle's commercial EULA in place of CC BY-NC 4.0), and submit for review.
6. **Test as a consumer.** Before going public, share the listing privately with a second account you control,
   install from the listing, run `quickstart.sql`, then uninstall.

## What was and was not verified

Verified offline in this repo: the YAML files parse; `setup_script.sql` and `quickstart.sql` pass a statement / quote /
`$$`-block balance check; every path the manifest, `snowflake.yml` and the spec name exists here; the routes
`/v1/snowflake/transcribe` and `/v1/snowflake/tags`, `ARC_FILE_ROOT`, the row / option shape and per-row errors match
`deploy/server/main.c`.

Not verified (no Snowflake account was used; written from Snowflake's Native App + SPCS documentation as of late 2026,
check these on the first `snow app run`):
- `manifest_version: 2` keys as written (`configuration.grant_callback`, `lifecycle_callbacks.version_initializer`,
  `privileges` with `CREATE COMPUTE POOL` / `CREATE WAREHOUSE`) and whether automated granting under v2 invokes the
  grant callback at install (if not, `core.start_app()` covers it).
- That a Native App service may mount the app-owned stage `@data.audio` as a volume, and that consumers holding
  `app_user` can `PUT` into it (fallback: send base64 rows to `services.transcribe_raw`).
- `EXECUTE IMMEDIATE :stmt` for the dynamically named pool / warehouse, `ALTER SERVICE ... SET QUERY_WAREHOUSE`, and SQL
  UDFs wrapping service functions (`services.transcribe`).
- Exact Snowflake CLI subcommands and flags (`snow app version create`, `release-directive`, `image-repository url`)
  for the CLI version you have; `snow --help` is authoritative.
- Compute pool sizing: the spec requests 2 CPU / 2 GiB with limits of 4 CPU / 4 GiB (copied from `../service.yaml`);
  confirm SPCS accepts those limits on a `CPU_X64_S` node and that memory holds for your call lengths.
