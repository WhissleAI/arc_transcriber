# arc-serve: Arc Mini (libarc) as an HTTP + WebSocket service. C11, libc, libm, pthreads; no Python, no ONNX Runtime.
#   make -C deploy                                  -> deploy/build/arc-serve           (transcriber)
#   make -C deploy ARCTAG=/path/to/arc-tagger-c     -> deploy/build/arc-serve-combined  (transcriber + Arc Tagger, native backend)
#   make -C deploy run MODEL=/path/arc-v1.1/mini
# Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0 with the Whissle terms; see LICENSE.
CC ?= cc
CFLAGS ?= -O2
B ?= build
C := ../c
VERSION ?= 0.1.0+$(shell git -C .. rev-parse --short HEAD 2>/dev/null || echo dev)
CFLAGS += -std=gnu11 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -pthread -I$(C)/include -Iserver
CFLAGS += -DARC_DEPLOY_VERSION=\"$(VERSION)\"
LDLIBS += -lm -pthread
LIBARC := $(C)/build/libarc.a
SRV_SRC := server/http.c server/audio.c server/main.c
ARCTAG ?=

ifeq ($(ARCTAG),)
TARGET := $(B)/arc-serve
CJSON_DIR := third_party/cjson
OBJDIR := $(B)/obj
EXTRA_OBJ :=
else
TARGET := $(B)/arc-serve-combined
CJSON_DIR := $(ARCTAG)/third_party/cjson
OBJDIR := $(B)/obj-combined
CFLAGS += -DARC_SERVE_TAGGER -I$(ARCTAG)/include -I$(ARCTAG)/deploy/server
TAG_SRC := $(ARCTAG)/src/arctag.c $(ARCTAG)/src/guard.c $(ARCTAG)/src/native.c $(ARCTAG)/src/onnx_file.c $(ARCTAG)/src/tokenizer.c $(ARCTAG)/deploy/server/tagsrv.c
EXTRA_OBJ := $(patsubst $(ARCTAG)/%.c,$(OBJDIR)/arctag/%.o,$(TAG_SRC))
ifeq ($(shell uname -s),Darwin)
LDLIBS += -framework Accelerate
endif
endif
CFLAGS += -I$(CJSON_DIR)
SRV_OBJ := $(patsubst server/%.c,$(OBJDIR)/%.o,$(SRV_SRC)) $(OBJDIR)/cJSON.o

all: $(TARGET)

$(LIBARC):
	$(MAKE) -C $(C) build/libarc.a

$(OBJDIR)/%.o: server/%.c server/http.h server/audio.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJDIR)/cJSON.o: $(CJSON_DIR)/cJSON.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJDIR)/arctag/%.o: $(ARCTAG)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(ARCTAG)/src -Wno-sign-compare -Wno-missing-field-initializers -c $< -o $@
$(TARGET): $(SRV_OBJ) $(EXTRA_OBJ) $(LIBARC)
	$(CC) $(CFLAGS) $(LDFLAGS) $(SRV_OBJ) $(EXTRA_OBJ) $(LIBARC) $(LDLIBS) -o $@

MODEL ?= /opt/ml/model
PORT ?= 8080
run: $(TARGET)
	ARC_MODEL_DIR=$(MODEL) PORT=$(PORT) $(TARGET)

clean:
	rm -rf $(B)
.PHONY: all run clean
