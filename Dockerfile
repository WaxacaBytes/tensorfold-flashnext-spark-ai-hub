# TensorFold's Zig engine for Qwen3.8-Flash-Next, for the Spark AI Hub
# (DGX Spark: aarch64, GB10 sm_121)
#
# Mia's AI Lab's single-Spark recipe builds this engine on every machine that
# runs it (scripts/prepare.sh); their published image stops at an older patch
# set. This repo builds exactly their prepare.sh Dockerfile, at their commit
# 656879f (patches 0001-0015, NVIDIA FP4 support), once, in CI, so the Spark AI
# Hub recipes pull it instead of compiling it on the user's Spark:
#   https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
#
# NVIDIA's PyTorch 26.07 container (arm64, CUDA 13), pinned by digest;
# Zig 0.17.0; TensorFold's zig-flashnext branch at db281878 with Mia's
# patches; the TP=1 kernel set compiled ahead of time for sm_121 (no GPU
# needed), source mtimes pinned so the cubins match theirs. The tag
# zig-db28187-c0a2df4c680d is the one Mia's scripts/config.sh computes for
# this build (TensorFold commit + hash of the patches and Dockerfile).
#
# No weights here: each Spark AI Hub recipe bakes its checkpoint in on top.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c

FROM ${BASE_IMAGE} AS build
ARG TF_REF=db281878ddb836fd0df510d8771ecb7e0fe47d26
ARG MIA_REF=656879fd752749bbd11a6e1a065a326c00940911
RUN curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz" && \
    echo "9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz && \
    /opt/zig/zig version
RUN git init -q /tensorfold && cd /tensorfold && git remote add origin https://github.com/ashhart/TensorFold.git && \
    git fetch -q --depth 1 origin "$TF_REF" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "$TF_REF"
RUN git init -q /mia && cd /mia && git remote add origin https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold.git && \
    git fetch -q --depth 1 origin "$MIA_REF" && git checkout -q --detach FETCH_HEAD && \
    cd /tensorfold && for p in /mia/patches/*.patch; do echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast --prefix /opt/tensorfold \
      -j"$(nproc)" fatbins install native && \
    test -x /opt/tensorfold/native/bin/tensorfold-native
# TensorFold's and Mia's licence files travel with the build (Apache-2.0, MIT).
RUN mkdir -p /opt/tensorfold/share/doc/tensorfold && cd /tensorfold && \
    for f in LICENSE LICENSE.md LICENSES NOTICE THIRD_PARTY_NOTICES.md; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done && \
    cp /mia/LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSE.mia && \
    cp /mia/NOTICE /opt/tensorfold/share/doc/tensorfold/NOTICE.mia
RUN cd /tensorfold && python -B -c 'import json, os, sys; t = int(sys.argv[1]); \
      [os.utime(os.path.join("src", f), (t, t)) for f in sorted({k["source"]["file"] for k in \
       json.load(open("zig/tests/cuda/flashnext/kernels.json"))["kernels"]})]' 1791318675 && \
    PYTHONPATH=/tensorfold/src python -B tools/zig/flashnext_aot.py build \
      --spec zig/tests/cuda/flashnext/kernels.json --jit zig/tests/cuda/flashnext/jit.json --tp 1 \
      --out /opt/tensorfold/share/tensorfold/cuda/sm121 > /tmp/aot.log || true && \
    cat /tmp/aot.log && \
    grep -q 'kernels ->' /tmp/aot.log && \
    bad=$(grep PROBLEM /tmp/aot.log | grep -v 'cubin sha256' || true) && \
    test -z "$bad" && \
    test -f /opt/tensorfold/share/tensorfold/cuda/sm121/aot.json

FROM ${BASE_IMAGE}
COPY --from=build /opt/tensorfold /opt/tensorfold
COPY --from=build /tensorfold/src/tensorfold /opt/tensorfold/python/tensorfold
RUN ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native && \
    pip install --no-cache-dir "huggingface_hub>=1.0" && \
    pip install --no-cache-dir --no-deps "transformers==5.17.0" "av==19.0.1" && \
    test -x "$(readlink -f /usr/local/bin/tensorfold-native)"
ENV TENSORFOLD_CUDA_KERNELS=/opt/tensorfold/share/tensorfold/cuda/sm121 \
    CUDA_CACHE_PATH=/cache/nv \
    TENSORFOLD_VISION_PYTHONPATH=/opt/tensorfold/python
WORKDIR /workspace
