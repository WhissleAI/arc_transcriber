-- Setup script for the Whissle Arc Snowflake Native App (runs in the consumer account on install and on every upgrade).
-- Objects:
--   application roles  app_admin (start / suspend / logs; includes app_user) and app_user (functions, audio stage)
--   core      VERSIONED schema: procedures, recreated with each version
--   data      stage data.audio, which the service mounts at /stage; consumers upload WAV / PCM files here
--   services  the SPCS service and its service functions (stateful, so not in the versioned schema)
-- The compute pool, warehouse, service and functions are created by core.start_app(), which the grant callback calls
-- once the app holds CREATE COMPUTE POOL (and CREATE WAREHOUSE). No external access integration exists or is requested.
-- Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0 with the Whissle terms; see LICENSE.

CREATE APPLICATION ROLE IF NOT EXISTS app_user;
CREATE APPLICATION ROLE IF NOT EXISTS app_admin;
GRANT APPLICATION ROLE app_user TO APPLICATION ROLE app_admin;

CREATE OR ALTER VERSIONED SCHEMA core;
GRANT USAGE ON SCHEMA core TO APPLICATION ROLE app_user;

CREATE SCHEMA IF NOT EXISTS data;
GRANT USAGE ON SCHEMA data TO APPLICATION ROLE app_user;
CREATE STAGE IF NOT EXISTS data.audio
  DIRECTORY = (ENABLE = TRUE)
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
  COMMENT = 'Audio for Arc: WAV or raw PCM16. Mounted into the service at /stage.';
GRANT READ, WRITE ON STAGE data.audio TO APPLICATION ROLE app_user;

CREATE SCHEMA IF NOT EXISTS services;
GRANT USAGE ON SCHEMA services TO APPLICATION ROLE app_user;

-- A path the server accepts (under /stage, the mount of @data.audio) from 'calls/a.wav', '@data.audio/calls/a.wav' or
-- '/stage/calls/a.wav'. Base64 audio (its alphabet has no '.') passes through unchanged.
CREATE OR REPLACE FUNCTION services.to_stage_path(p VARCHAR)
  RETURNS VARCHAR
  AS
$$
  CASE
    WHEN STARTSWITH(p, '/stage/') THEN p
    WHEN STARTSWITH(LOWER(p), '@data.audio/') THEN '/stage/' || SUBSTR(p, 13)
    WHEN CONTAINS(p, '.') THEN '/stage/' || LTRIM(p, '/')
    ELSE p
  END
$$;
GRANT USAGE ON FUNCTION services.to_stage_path(VARCHAR) TO APPLICATION ROLE app_user;

-- Create whatever the app's privileges allow: compute pool + service + functions (CREATE COMPUTE POOL), warehouse
-- (CREATE WAREHOUSE). Idempotent; safe to call again.
CREATE OR REPLACE PROCEDURE core.start_app()
  RETURNS STRING
  LANGUAGE SQL
  EXECUTE AS OWNER
AS
$$
DECLARE
  pool VARCHAR DEFAULT CURRENT_DATABASE() || '_ARC_POOL';
  wh   VARCHAR DEFAULT CURRENT_DATABASE() || '_ARC_WH';
  msg  VARCHAR DEFAULT '';
  stmt VARCHAR;
BEGIN
  IF (SYSTEM$HOLD_PRIVILEGE_ON_ACCOUNT('CREATE WAREHOUSE')) THEN
    stmt := 'CREATE WAREHOUSE IF NOT EXISTS ' || wh ||
      ' WAREHOUSE_SIZE = XSMALL AUTO_SUSPEND = 60 AUTO_RESUME = TRUE INITIALLY_SUSPENDED = TRUE';
    EXECUTE IMMEDIATE :stmt;
    msg := msg || 'warehouse ' || wh || '; ';
  END IF;

  IF (NOT SYSTEM$HOLD_PRIVILEGE_ON_ACCOUNT('CREATE COMPUTE POOL')) THEN
    RETURN msg || 'waiting for CREATE COMPUTE POOL';
  END IF;

  -- one CPU_X64_S node; services/arc.yaml requests 2 CPU / 2 GiB (limits 4 CPU / 4 GiB, as ../service.yaml)
  stmt := 'CREATE COMPUTE POOL IF NOT EXISTS ' || pool ||
    ' MIN_NODES = 1 MAX_NODES = 1 INSTANCE_FAMILY = CPU_X64_S AUTO_RESUME = TRUE AUTO_SUSPEND_SECS = 3600';
  EXECUTE IMMEDIATE :stmt;

  stmt := 'CREATE SERVICE IF NOT EXISTS services.arc_svc IN COMPUTE POOL ' || pool ||
    ' FROM SPECIFICATION_FILE = ''services/arc.yaml'' MIN_INSTANCES = 1 MAX_INSTANCES = 1';
  EXECUTE IMMEDIATE :stmt;
  IF (SYSTEM$HOLD_PRIVILEGE_ON_ACCOUNT('CREATE WAREHOUSE')) THEN
    stmt := 'ALTER SERVICE services.arc_svc SET QUERY_WAREHOUSE = ' || wh;
    EXECUTE IMMEDIATE :stmt;
  END IF;

  -- Service functions: SPCS POSTs {"data": [[row, arg1, arg2?], ...]} to the route and reads {"data": [[row, value]]}.
  -- audio: a path under /stage (the mounted data.audio stage) or base64 WAV / PCM16. Returns verbose_json (+ arc_tags).
  CREATE OR REPLACE FUNCTION services.transcribe_raw(audio VARCHAR)
    RETURNS VARIANT
    SERVICE = services.arc_svc ENDPOINT = api MAX_BATCH_ROWS = 4
    AS '/v1/snowflake/transcribe';
  -- options: '{"domain": "nba", "names": [...], "tags": false, "tagger": {"redact": "mask"}}'
  CREATE OR REPLACE FUNCTION services.transcribe_opts(audio VARCHAR, options VARCHAR)
    RETURNS VARIANT
    SERVICE = services.arc_svc ENDPOINT = api MAX_BATCH_ROWS = 4
    AS '/v1/snowflake/transcribe';
  -- text -> Arc Tagger output (spans, events, utterances, redacted_text)
  CREATE OR REPLACE FUNCTION services.tags(txt VARCHAR)
    RETURNS VARIANT
    SERVICE = services.arc_svc ENDPOINT = api MAX_BATCH_ROWS = 64
    AS '/v1/snowflake/tags';
  -- options: '{"domain": "talk", "redact": "mask"}'
  CREATE OR REPLACE FUNCTION services.tags_opts(txt VARCHAR, options VARCHAR)
    RETURNS VARIANT
    SERVICE = services.arc_svc ENDPOINT = api MAX_BATCH_ROWS = 64
    AS '/v1/snowflake/tags';
  -- TRANSCRIBE(stage_path [, options]): stage_path relative to @data.audio; see services.to_stage_path (top of this script)
  CREATE OR REPLACE FUNCTION services.transcribe(stage_path VARCHAR)
    RETURNS VARIANT
    AS 'services.transcribe_raw(services.to_stage_path(stage_path))';
  CREATE OR REPLACE FUNCTION services.transcribe(stage_path VARCHAR, options VARCHAR)
    RETURNS VARIANT
    AS 'services.transcribe_opts(services.to_stage_path(stage_path), options)';

  GRANT USAGE ON FUNCTION services.transcribe(VARCHAR) TO APPLICATION ROLE app_user;
  GRANT USAGE ON FUNCTION services.transcribe(VARCHAR, VARCHAR) TO APPLICATION ROLE app_user;
  GRANT USAGE ON FUNCTION services.transcribe_raw(VARCHAR) TO APPLICATION ROLE app_user;
  GRANT USAGE ON FUNCTION services.transcribe_opts(VARCHAR, VARCHAR) TO APPLICATION ROLE app_user;
  GRANT USAGE ON FUNCTION services.tags(VARCHAR) TO APPLICATION ROLE app_user;
  GRANT USAGE ON FUNCTION services.tags_opts(VARCHAR, VARCHAR) TO APPLICATION ROLE app_user;
  GRANT MONITOR ON SERVICE services.arc_svc TO APPLICATION ROLE app_admin;

  RETURN msg || 'compute pool ' || pool || '; service services.arc_svc; functions services.transcribe / services.tags';
END;
$$;
GRANT USAGE ON PROCEDURE core.start_app() TO APPLICATION ROLE app_admin;

-- Grant callback (manifest configuration.grant_callback): called with the privileges just granted.
CREATE OR REPLACE PROCEDURE core.grant_callback(privileges ARRAY)
  RETURNS STRING
  LANGUAGE SQL
  EXECUTE AS OWNER
AS
$$
BEGIN
  CALL core.start_app();
  RETURN 'start_app ran for ' || ARRAY_TO_STRING(privileges, ', ');
END;
$$;

-- Version initializer (manifest lifecycle_callbacks.version_initializer): on upgrade, move the running service to
-- this version's spec (and so its image). Before the first start there is no service; nothing to do.
CREATE OR REPLACE PROCEDURE core.version_init()
  RETURNS STRING
  LANGUAGE SQL
  EXECUTE AS OWNER
AS
$$
BEGIN
  ALTER SERVICE IF EXISTS services.arc_svc FROM SPECIFICATION_FILE = 'services/arc.yaml';
  RETURN 'service spec updated';
EXCEPTION
  WHEN OTHER THEN
    RETURN 'no service to update: ' || SQLERRM;
END;
$$;

CREATE OR REPLACE PROCEDURE core.service_status()
  RETURNS VARCHAR
  LANGUAGE SQL
  EXECUTE AS OWNER
AS
$$
BEGIN
  RETURN SYSTEM$GET_SERVICE_STATUS('services.arc_svc');
END;
$$;
GRANT USAGE ON PROCEDURE core.service_status() TO APPLICATION ROLE app_admin;

CREATE OR REPLACE PROCEDURE core.service_logs(lines INTEGER)
  RETURNS VARCHAR
  LANGUAGE SQL
  EXECUTE AS OWNER
AS
$$
BEGIN
  RETURN SYSTEM$GET_SERVICE_LOGS('services.arc_svc', 0, 'arc', lines);
END;
$$;
GRANT USAGE ON PROCEDURE core.service_logs(INTEGER) TO APPLICATION ROLE app_admin;

-- Pick up files uploaded to @data.audio in DIRECTORY(@data.audio) (internal stages do not refresh on PUT).
CREATE OR REPLACE PROCEDURE core.refresh_audio()
  RETURNS VARCHAR
  LANGUAGE SQL
  EXECUTE AS OWNER
AS
$$
BEGIN
  ALTER STAGE data.audio REFRESH;
  RETURN 'refreshed';
END;
$$;
GRANT USAGE ON PROCEDURE core.refresh_audio() TO APPLICATION ROLE app_user;

-- Stop paying for the compute pool when idle; core.resume_app() brings it back.
CREATE OR REPLACE PROCEDURE core.suspend_app()
  RETURNS VARCHAR
  LANGUAGE SQL
  EXECUTE AS OWNER
AS
$$
DECLARE
  stmt VARCHAR;
BEGIN
  ALTER SERVICE IF EXISTS services.arc_svc SUSPEND;
  stmt := 'ALTER COMPUTE POOL IF EXISTS ' || CURRENT_DATABASE() || '_ARC_POOL SUSPEND';
  EXECUTE IMMEDIATE :stmt;
  RETURN 'suspended';
END;
$$;
GRANT USAGE ON PROCEDURE core.suspend_app() TO APPLICATION ROLE app_admin;

CREATE OR REPLACE PROCEDURE core.resume_app()
  RETURNS VARCHAR
  LANGUAGE SQL
  EXECUTE AS OWNER
AS
$$
DECLARE
  stmt VARCHAR;
BEGIN
  stmt := 'ALTER COMPUTE POOL IF EXISTS ' || CURRENT_DATABASE() || '_ARC_POOL RESUME';
  EXECUTE IMMEDIATE :stmt;
  ALTER SERVICE IF EXISTS services.arc_svc RESUME;
  RETURN 'resumed';
END;
$$;
GRANT USAGE ON PROCEDURE core.resume_app() TO APPLICATION ROLE app_admin;
