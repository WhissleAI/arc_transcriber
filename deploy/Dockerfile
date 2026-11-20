# syntax=docker/dockerfile:1
# Arc Transcriber (Arc Mini, libarc) as a service: arc-serve, one static-ish C binary on a distroless base.
# CPU only: libarc has no GPU path (see deploy/README.md), so there is no CUDA variant.
#
#   # bring-your-own weights (SageMaker / Hugging Face mount them at /opt/ml/model or /repository)
#   docker buildx build --platform linux/amd64,linux/arm64 -f deploy/Dockerfile --target runtime -t arc-transcriber:0.1.0 .
#   # weights inside the image (Marketplace model package, Snowflake, NIM-style): deploy/fetch_weights.sh first
#   docker buildx build --platform linux/amd64,linux/arm64 -f deploy/Dockerfile --target bundled \
#       --build-arg WEIGHTS=deploy/weights/mini --build-arg MODEL_ID=arc-mini -t arc-transcriber:0.1.0-mini .
#
# A later "Pro" image is the same Dockerfile with other weights and id: --build-arg WEIGHTS=deploy/weights/pro
# --build-arg MODEL_ID=arc-pro (the binary reads whatever Arc Mini-format folder it is given; nothing else changes).
# Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0 with the Whissle terms; see LICENSE.

ARG BUILD_IMAGE=debian:bookworm-slim
ARG RUNTIME_IMAGE=gcr.io/distroless/cc-debian12:nonroot

FROM ${BUILD_IMAGE} AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc make libc6-dev binutils && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY c/include c/include
COPY c/src c/src
COPY c/Makefile c/Makefile
COPY deploy/Makefile deploy/Makefile
COPY deploy/server deploy/server
COPY deploy/third_party deploy/third_party
ARG VERSION=0.1.0
# CFLAGS / LDFLAGS from the environment: the Makefiles append their own flags (-std, -I, ...) to them. -O2 generic
# code: libarc picks its AVX2 / NEON kernels at run time (x86-64 without AVX2 falls back to portable C)
ENV CFLAGS="-O2 -fstack-protector-strong -D_FORTIFY_SOURCE=2" LDFLAGS="-Wl,-z,relro,-z,now"
RUN make -C c -j2 build/libarc.a \
 && make -C deploy -j2 VERSION="${VERSION}" \
 && strip deploy/build/arc-serve \
 && ./deploy/build/arc-serve --version

FROM ${RUNTIME_IMAGE} AS runtime
ARG VERSION=0.1.0
LABEL org.opencontainers.image.title="arc-transcriber" \
      org.opencontainers.image.description="Arc Mini speech-to-text with linked names, number values, inline tags and utterance tags (libarc, CPU)" \
      org.opencontainers.image.vendor="Whissle Inc." \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.licenses="CC-BY-NC-4.0" \
      org.opencontainers.image.source="https://github.com/WhissleAI/arc_transcriber" \
      com.amazonaws.sagemaker.capabilities.accept-bind-to-port="true"
COPY --from=build /src/deploy/build/arc-serve /opt/arc/bin/serve
COPY deploy/openapi.yaml /opt/arc/openapi.yaml
COPY LICENSE NOTICE /opt/arc/
ENV PORT=8080 HOST=0.0.0.0 ARC_WORKERS=2 ARC_ACCESS_LOG=1
EXPOSE 8080
# distroless :nonroot runs as uid 65532; no shell, no package manager, no curl (the binary is its own health check)
USER 65532:65532
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD ["/opt/arc/bin/serve", "--healthcheck"]
ENTRYPOINT ["/opt/arc/bin/serve"]

FROM runtime AS bundled
ARG WEIGHTS=deploy/weights/mini
ARG MODEL_ID=arc-mini
# read-only weights owned by root; found at /opt/arc/model unless a platform mounts others at /opt/ml/model or /repository
COPY --chown=0:0 ${WEIGHTS} /opt/arc/model
ENV ARC_MODEL_ID=${MODEL_ID}
