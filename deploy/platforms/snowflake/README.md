# Snowflake Snowpark Container Services (SPCS)

The **bundled** image runs as an SPCS service; SQL reaches it through **service functions**, which POST batches of rows
as `{"data": [[row, arg1, ...], ...]}` and expect `{"data": [[row, result], ...]}`. arc-serve answers that shape at
`/v1/snowflake/transcribe`. Audio comes either as a path inside a stage mounted into the container (`ARC_FILE_ROOT`) or
as base64 in the row. The service needs no external access integration (it makes no network calls).

```bash
# SPCS runs linux/amd64 images
HF_TOKEN=hf_... deploy/fetch_weights.sh mini
docker buildx build --platform linux/amd64 -f deploy/Dockerfile --target bundled -t arc-transcriber:0.1.0-mini --load .
snow spcs image-registry login        # or: docker login <org>-<account>.registry.snowflakecomputing.com
docker tag arc-transcriber:0.1.0-mini <org>-<account>.registry.snowflakecomputing.com/arc_db/arc/images/arc-transcriber:0.1.0-mini
docker push <org>-<account>.registry.snowflakecomputing.com/arc_db/arc/images/arc-transcriber:0.1.0-mini
```

Then [`setup.sql`](setup.sql) (compute pool, stage, service from [`service.yaml`](service.yaml), functions) and:

```sql
SELECT relative_path, arc_transcribe('/stage/' || relative_path) AS t
FROM DIRECTORY(@arc_db.arc.audio) WHERE relative_path ILIKE '%.wav';

SELECT t:text::string, f.value:entity::string AS linked_name
FROM (SELECT arc_transcribe_opts('/stage/game1.wav', '{"domain": "nba", "names": ["Theo Brandt", "Harbor City Hawks"]}') AS t),
     LATERAL FLATTEN(t:names) f WHERE f.value:state = 'linked';
```

Notes:
- Paths must resolve inside `ARC_FILE_ROOT` (symlinks and `..` are resolved and checked); other rows get
  `{"error": ...}` as their value instead of failing the whole batch.
- Keep `MAX_BATCH_ROWS` small for audio (each row is a whole clip); one replica handles `ARC_WORKERS` clips at a time.
- `CPU_X64_S` / `CPU_X64_M` pools fit Arc Mini (about 180 MB for a 30 s clip with every read-out; more for long files).

To list on the Snowflake Marketplace, [`native_app/`](native_app/README.md) wraps the combined image (transcriber +
tagger) as a **Native App**: manifest, setup script (application roles, versioned schema, compute pool, service,
`TRANSCRIBE` / `TAGS` service functions), a SQL quick start and the provider steps.
