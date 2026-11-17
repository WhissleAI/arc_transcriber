# libarc: `make` builds build/libarc.a, the tools (arc-transcribe, arc-stream, arc-score, arc-parity, arc-trim) and the tests.
# `make test` runs the unit tests that need no weights. C11; libc, libm and threads only.
# Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
CC ?= cc
CFLAGS ?= -O2
# Raspberry Pi presets: make PI=5 (Cortex-A76) or PI=4 (Cortex-A72); CPU=... for another -mcpu
ifeq ($(PI),5)
CPU ?= cortex-a76
endif
ifeq ($(PI),4)
CPU ?= cortex-a72
endif
ifneq ($(CPU),)
CFLAGS += -mcpu=$(CPU)
endif
CFLAGS += -std=c11 -Iinclude -Isrc -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -pthread
LDLIBS += -lm
# NFT2 (Arc Transcriber Space) encoder + RNNT backend: make ARC_ORT=/path/to/onnxruntime (its include/ and lib/) links
# ONNX Runtime and builds into build/nft2/; without it the library stays dependency-free and arc_nft2_create reports that no backend is built in.
ARC_ORT ?=
ifneq ($(ARC_ORT),)
NFT2_BE := src/arc_nft2_ort.c
B ?= build/nft2
CFLAGS += -I$(ARC_ORT)/include -DARC_HAVE_ORT
LDLIBS += -L$(ARC_ORT)/lib -lonnxruntime -Wl,-rpath,$(abspath $(ARC_ORT))/lib
else
NFT2_BE := src/arc_nft2_none.c
B ?= build
endif
# Arc Tagger on each final (arc_nft2_set_tagger): make ARC_WITH_ARCTAG=/path/to/arc-tagger-c (its include/ and a built
# build/libarctag.a; the tagger's native backend needs nothing else)
ARC_WITH_ARCTAG ?=
ifneq ($(ARC_WITH_ARCTAG),)
CFLAGS += -DARC_WITH_ARCTAG -I$(ARC_WITH_ARCTAG)/include
LDLIBS := $(ARC_WITH_ARCTAG)/build/libarctag.a $(LDLIBS)
endif
SRC := $(filter-out src/arc_nft2_ort.c src/arc_nft2_none.c,$(wildcard src/*.c)) $(NFT2_BE)
HDR := $(wildcard src/*.h include/*.h)
OBJ := $(patsubst src/%.c,$(B)/%.o,$(SRC))
UNIT := test_num test_ntext test_tidy test_post test_echo
TESTS := $(UNIT) test_moon test_tok test_transcribe test_full store_hash test_whole_long test_score test_stream

all: $(B)/libarc.a $(B)/arc-transcribe $(B)/arc-stream $(B)/arc-score $(B)/arc-parity $(B)/arc-trim $(B)/arc-nft2 $(B)/nft2-app-example $(addprefix $(B)/,$(TESTS))

$(B)/%.o: src/%.c $(HDR) | $(B)
	$(CC) $(CFLAGS) -c $< -o $@
$(B)/libarc.a: $(OBJ)
	rm -f $@ && ar rcs $@ $^
$(B)/arc-transcribe: tools/arc_transcribe.c tools/json_out.h $(B)/libarc.a
	$(CC) $(CFLAGS) $< $(B)/libarc.a $(LDLIBS) -o $@
$(B)/arc-stream: tools/arc_stream.c tools/json_out.h $(B)/libarc.a
	$(CC) $(CFLAGS) $< $(B)/libarc.a $(LDLIBS) -o $@
$(B)/arc-score: tools/arc_score.c $(B)/libarc.a
	$(CC) $(CFLAGS) $< $(B)/libarc.a $(LDLIBS) -o $@
$(B)/arc-trim: tools/arc_trim.c $(B)/libarc.a
	$(CC) $(CFLAGS) $< $(B)/libarc.a $(LDLIBS) -o $@
$(B)/arc-parity: tools/arc_parity.c tools/json_out.h $(B)/libarc.a
	$(CC) $(CFLAGS) $< $(B)/libarc.a $(LDLIBS) -o $@
$(B)/arc-nft2: tools/arc_nft2.c tools/json_out.h $(B)/libarc.a
	$(CC) $(CFLAGS) $< $(B)/libarc.a $(LDLIBS) -o $@
$(B)/nft2-app-example: examples/nft2_app.c include/arc.h $(B)/libarc.a
	$(CC) $(CFLAGS) $< $(B)/libarc.a $(LDLIBS) -o $@
$(B)/test_%: tests/test_%.c $(B)/libarc.a
	$(CC) $(CFLAGS) $< $(B)/libarc.a $(LDLIBS) -o $@
$(B)/store_hash: tests/store_hash.c $(B)/libarc.a
	$(CC) $(CFLAGS) $< $(B)/libarc.a $(LDLIBS) -o $@
$(B):
	mkdir -p $(B)

# unit tests against arc_audio's own outputs (fixtures in tests/data, made by tools/dump_{num,ntext,tidy,post,echo}.py)
test: $(addprefix $(B)/,$(UNIT))
	$(B)/test_num tests/data/num.jsonl
	$(B)/test_ntext tests/data/ntext.jsonl
	$(B)/test_tidy tests/data/tidy.jsonl
	$(B)/test_post tests/data/post.jsonl
	$(B)/test_echo tests/data/echo.json ../examples/example_synth.wav

# the release's golden check (needs the weights): make parity MINI=/path/arc-v1.1/mini [ARC=..] [PY=python]
ARC ?= ..
PY ?= python3
parity: $(B)/arc-parity
	$(B)/arc-parity $(MINI) $(ARC)/examples/example_synth.wav $(ARC)/examples/example_names.json > $(B)/candidate.json
	cd $(ARC) && $(PY) scripts/check_parity.py --backbone mini --candidate $(CURDIR)/$(B)/candidate.json
# install: PREFIX/lib/libarc.a, PREFIX/include/arc.h, PREFIX/bin/arc-transcribe, arc-stream
PREFIX ?= /usr/local
install: $(B)/libarc.a $(B)/arc-transcribe $(B)/arc-stream
	mkdir -p $(PREFIX)/lib $(PREFIX)/include $(PREFIX)/bin
	cp $(B)/libarc.a $(PREFIX)/lib/
	cp include/arc.h $(PREFIX)/include/
	cp $(B)/arc-transcribe $(B)/arc-stream $(PREFIX)/bin/
clean:
	rm -rf build
.PHONY: all clean test parity install
