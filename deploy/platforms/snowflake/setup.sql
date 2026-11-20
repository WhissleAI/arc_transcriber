-- Arc Transcriber on Snowpark Container Services. Run as a role that can create these objects.
CREATE DATABASE IF NOT EXISTS arc_db;
CREATE SCHEMA IF NOT EXISTS arc_db.arc;
CREATE IMAGE REPOSITORY IF NOT EXISTS arc_db.arc.images;
CREATE STAGE IF NOT EXISTS arc_db.arc.audio DIRECTORY = (ENABLE = TRUE) ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
CREATE STAGE IF NOT EXISTS arc_db.arc.specs;
-- PUT file://deploy/platforms/snowflake/service.yaml @arc_db.arc.specs AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

CREATE COMPUTE POOL IF NOT EXISTS arc_pool MIN_NODES = 1 MAX_NODES = 2 INSTANCE_FAMILY = CPU_X64_S;

CREATE SERVICE IF NOT EXISTS arc_db.arc.arc_svc
  IN COMPUTE POOL arc_pool
  FROM @arc_db.arc.specs SPECIFICATION_FILE = 'service.yaml'
  MIN_INSTANCES = 1 MAX_INSTANCES = 2;

-- SELECT SYSTEM$GET_SERVICE_STATUS('arc_db.arc.arc_svc');
-- SELECT SYSTEM$GET_SERVICE_LOGS('arc_db.arc.arc_svc', 0, 'arc');

-- audio: a path under /stage (the mounted stage) or base64 WAV / PCM16; returns verbose_json as a VARIANT
CREATE OR REPLACE FUNCTION arc_db.arc.arc_transcribe(audio VARCHAR)
  RETURNS VARIANT
  SERVICE = arc_db.arc.arc_svc ENDPOINT = api MAX_BATCH_ROWS = 4
  AS '/v1/snowflake/transcribe';

-- with options: '{"domain": "nba", "names": [...], "response_format": "verbose_json"}'
CREATE OR REPLACE FUNCTION arc_db.arc.arc_transcribe_opts(audio VARCHAR, options VARCHAR)
  RETURNS VARIANT
  SERVICE = arc_db.arc.arc_svc ENDPOINT = api MAX_BATCH_ROWS = 4
  AS '/v1/snowflake/transcribe';

-- combined image only (Dockerfile.combined): tags on text
-- CREATE OR REPLACE FUNCTION arc_db.arc.arc_tags(text VARCHAR) RETURNS VARIANT
--   SERVICE = arc_db.arc.arc_svc ENDPOINT = api MAX_BATCH_ROWS = 64 AS '/v1/snowflake/tags';
