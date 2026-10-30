# libarc: `make` builds build/libarc.a, the tools (arc-transcribe, arc-score, arc-parity, arc-trim) and the tests.
# `make test` runs the unit tests that need no weights. C11; libc, libm and threads only.
# Part of arc_transcriber (Whissle Inc.), CC BY-NC 4.0; see LICENSE.
CC ?= cc
CFLAGS ?= -O2
CFLAGS += -std=c11 -Iinclude -Isrc -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -pthread
LDLIBS += -lm
SRC := $(wildcard src/*.c)
HDR := $(wildcard src/*.h include/*.h)
OBJ := $(patsubst src/%.c,build/%.o,$(SRC))
UNIT := test_num test_ntext test_tidy
TESTS := $(UNIT) test_moon test_tok test_transcribe test_full store_hash test_whole_long test_score

all: build/libarc.a build/arc-transcribe build/arc-score build/arc-parity build/arc-trim $(addprefix build/,$(TESTS))

build/%.o: src/%.c $(HDR) | build
	$(CC) $(CFLAGS) -c $< -o $@
build/libarc.a: $(OBJ)
	ar rcs $@ $^
build/arc-transcribe: tools/arc_transcribe.c tools/json_out.h build/libarc.a
	$(CC) $(CFLAGS) $< build/libarc.a $(LDLIBS) -o $@
build/arc-score: tools/arc_score.c build/libarc.a
	$(CC) $(CFLAGS) $< build/libarc.a $(LDLIBS) -o $@
build/arc-trim: tools/arc_trim.c build/libarc.a
	$(CC) $(CFLAGS) $< build/libarc.a $(LDLIBS) -o $@
build/arc-parity: tools/arc_parity.c tools/json_out.h build/libarc.a
	$(CC) $(CFLAGS) $< build/libarc.a $(LDLIBS) -o $@
build/test_%: tests/test_%.c build/libarc.a
	$(CC) $(CFLAGS) $< build/libarc.a $(LDLIBS) -o $@
build/store_hash: tests/store_hash.c build/libarc.a
	$(CC) $(CFLAGS) $< build/libarc.a $(LDLIBS) -o $@
build:
	mkdir -p build

# unit tests against arc_audio's own outputs (fixtures in tests/data, made by tools/dump_{num,ntext,tidy}.py)
test: $(addprefix build/,$(UNIT))
	build/test_num tests/data/num.jsonl
	build/test_ntext tests/data/ntext.jsonl
	build/test_tidy tests/data/tidy.jsonl

# the release's golden check (needs the weights): make parity MINI=/path/arc-v1.1/mini [ARC=..] [PY=python]
ARC ?= ..
PY ?= python3
parity: build/arc-parity
	build/arc-parity $(MINI) $(ARC)/examples/example_synth.wav $(ARC)/examples/example_names.json > build/candidate.json
	cd $(ARC) && $(PY) scripts/check_parity.py --backbone mini --candidate $(CURDIR)/build/candidate.json
clean:
	rm -rf build
.PHONY: all clean test parity
