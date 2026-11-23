-- Whissle Arc Native App: consumer quick start (SQL only). Replace ARC_APP with the installed application's name.
-- Run as ACCOUNTADMIN, or a role that can grant account privileges to the app and that holds app_user.

-- 1. Let the app create its compute pool and warehouse, then start it (the grant callback normally starts it).
GRANT CREATE COMPUTE POOL ON ACCOUNT TO APPLICATION arc_app;
GRANT CREATE WAREHOUSE ON ACCOUNT TO APPLICATION arc_app;
CALL arc_app.core.start_app();
CALL arc_app.core.service_status();           -- repeat until the status is READY
-- CALL arc_app.core.service_logs(100);

-- 2. Who may call it.
GRANT APPLICATION ROLE arc_app.app_user TO ROLE SYSADMIN;

-- 3. Audio in. From SnowSQL / Snowflake CLI (`snow sql`), on the machine holding the files:
--    PUT file://examples/example_synth.wav @arc_app.data.audio/demo/ AUTO_COMPRESS = FALSE OVERWRITE = TRUE;
CALL arc_app.core.refresh_audio();
SELECT relative_path, size FROM DIRECTORY(@arc_app.data.audio);

-- 4. Transcribe (verbose_json + arc_tags as a VARIANT).
CREATE OR REPLACE TEMPORARY TABLE arc_demo AS
SELECT relative_path, arc_app.services.transcribe(relative_path) AS t
FROM DIRECTORY(@arc_app.data.audio)
WHERE relative_path ILIKE '%.wav';

SELECT relative_path, t:text::string AS transcript, t:duration::float AS seconds, t:error AS error FROM arc_demo;

-- 5. Names linked to your list, with options.
SELECT n.value:heard::string AS heard, n.value:entity::string AS linked_to
FROM (SELECT arc_app.services.transcribe('demo/example_synth.wav',
        '{"domain": "nba", "names": ["Marcus Delaney", "Theo Brandt", "Harbor City Hawks", "Riverside Comets"]}') AS t),
     LATERAL FLATTEN(t:names) n
WHERE n.value:state = 'linked';

-- 6. Tagger read-outs on the transcript: PII and safety spans per call.
SELECT relative_path, s.value:type::string AS span_type, s.value:text::string AS span_text, s.value:score::float AS score
FROM arc_demo, LATERAL FLATTEN(t:arc_tags:spans) s;

-- 7. Tag any text directly, with redaction.
SELECT arc_app.services.tags_opts('Hi, this is Maria Lopez. My number is 555 867 5309.',
                                  '{"domain": "talk", "redact": "mask"}'):redacted_text::string AS redacted;

-- 8. Idle: stop paying for the pool.
-- CALL arc_app.core.suspend_app();   -- later: CALL arc_app.core.resume_app();
