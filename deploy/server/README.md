# arc-serve source

The C source of the hosted service. Usage, routes and configuration: [deploy/README.md](../README.md).

| file | what |
|---|---|
| `main.c` | routes, request options, engine pool, SageMaker / Snowflake shapes, realtime sessions |
| `http.c`, `http.h` | HTTP/1.1 and WebSocket server (keep-alive, limits, access log) |
| `audio.c`, `audio.h` | WAV / PCM decoding and resampling to 16 kHz mono |

```bash
make -C deploy -j2                              # deploy/build/arc-serve
make -C deploy -j2 ARCTAG=/path/to/arc-tagger-c  # deploy/build/arc-serve-combined (with Arc Tagger)
deploy/build/arc-serve --port 8080 --model /path/to/arc-v1.1/mini
```

JSON parsing uses the bundled cJSON in `deploy/third_party/cjson` (MIT).
