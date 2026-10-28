# TensorFold's Zig engine for Qwen3.8-Flash-Next, for the Spark AI Hub
# (DGX Spark: aarch64, GB10 sm_121)
#
# Mia's AI Lab's single-Spark recipe builds this engine on every machine that
# runs it (scripts/prepare.sh) and publishes no image of this build. This repo
# builds exactly their prepare.sh Dockerfile, at their commit fd3f43e
# (2026-10-27: TensorFold v1.1.4 with patches 0001-0017, NVIDIA FP4), once, in
# CI, so the Spark AI Hub recipes pull it instead of compiling it on the
# user's Spark:
#   https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
#
# NVIDIA's PyTorch 26.07 container (arm64, CUDA 13), pinned by digest; Zig
# 0.17.0; TensorFold v1.1.4 (eab8582) with Mia's patches, kernels embedded in
# the binary; libjpeg-turbo and libwebp's decoder from pinned sources; and
# libtfgrammar.so (structured output, required tool calls) from TensorFold's
# v1.1.4 release archive. Every download is checked against its SHA-256. The
# only change from their Dockerfile: the patches are fetched from their repo
# at the pinned commit instead of copied from a checkout. The tag
# v1.1.4-eab8582-da5e32ac724a is the one their scripts/config.sh computes for
# this build (TensorFold commit + hash of the patches, Dockerfile and pins).
#
# No weights here: each Spark AI Hub recipe bakes its checkpoint in on top.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
FROM ${BASE_IMAGE} AS codecs
ARG LIBJPEG_TURBO_VERSION=3.1.4.1
ARG LIBJPEG_TURBO_SHA256=ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022
ARG LIBWEBP_VERSION=1.6.0
ARG LIBWEBP_SHA256=e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564
RUN mkdir -p /src /opt/codecs && cd /src && \
    curl -fsSL -o jt.tar.gz "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${LIBJPEG_TURBO_VERSION}/libjpeg-turbo-${LIBJPEG_TURBO_VERSION}.tar.gz" && \
    echo "${LIBJPEG_TURBO_SHA256}  jt.tar.gz" | sha256sum -c - && \
    tar -xzf jt.tar.gz && cmake -S libjpeg-turbo-${LIBJPEG_TURBO_VERSION} -B jt-build -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_SHARED=ON -DENABLE_STATIC=OFF -DWITH_TURBOJPEG=ON -DCMAKE_INSTALL_PREFIX=/opt/codecs/jpeg -DCMAKE_INSTALL_LIBDIR=lib && \
    cmake --build jt-build -j"$(nproc)" && cmake --install jt-build && \
    test -e /opt/codecs/jpeg/lib/libturbojpeg.so.0
RUN cd /src && \
    curl -fsSL -o webp.tar.gz "https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-${LIBWEBP_VERSION}.tar.gz" && \
    echo "${LIBWEBP_SHA256}  webp.tar.gz" | sha256sum -c - && \
    tar -xzf webp.tar.gz && cd libwebp-${LIBWEBP_VERSION} && \
    ./configure --prefix=/opt/codecs/webp --enable-libwebpdecoder --disable-static --disable-gl --disable-sdl --disable-png --disable-jpeg --disable-tiff --disable-gif --disable-wic && \
    make -j"$(nproc)" && make install && \
    test -e /opt/codecs/webp/lib/libwebpdecoder.so.3

FROM ${BASE_IMAGE} AS grammar
ARG TF_RELEASE_VERSION=1.1.4
ARG TF_RELEASE_AARCH64_SHA256=2c77ff0179dbfb9ae1c6ac647f85210194c151243f49eb93f0e34a8d53d7a4ec
RUN mkdir -p /opt/tfrel && cd /tmp && \
    curl -fsSL -o tf-rel.tar.gz "https://github.com/ashhart/TensorFold/releases/download/v${TF_RELEASE_VERSION}/tensorfold-${TF_RELEASE_VERSION}-linux-aarch64.tar.gz" && \
    echo "${TF_RELEASE_AARCH64_SHA256}  tf-rel.tar.gz" | sha256sum -c - && \
    tar -xzf tf-rel.tar.gz -C /opt/tfrel --strip-components=1 \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/lib/libtfgrammar.so \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/LICENSES/xgrammar-Apache-2.0.txt \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/LICENSES/xgrammar-NOTICE.txt \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/LICENSES/dlpack-Apache-2.0.txt && \
    test -s /opt/tfrel/lib/libtfgrammar.so

FROM ${BASE_IMAGE} AS build
ARG ZIG_VERSION=0.17.0
ARG ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8
RUN curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" && \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz && \
    /opt/zig/zig version
ARG TF_REPO=https://github.com/ashhart/TensorFold.git
ARG TF_REF=eab8582819688325e371907e4706e1c602b8a273
RUN git init -q /tensorfold && cd /tensorfold && git remote add origin "${TF_REPO}" && \
    git fetch -q --depth 1 origin "${TF_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${TF_REF}"
ARG MIA_REF=fd3f43e11238553685ebb68b6d8ab2b112818e3c
RUN git init -q /mia && cd /mia && git remote add origin https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold.git && \
    git fetch -q --depth 1 origin "${MIA_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${MIA_REF}" && mkdir -p /opt/tf-patches && cp /mia/patches/*.patch /opt/tf-patches/
RUN cd /tensorfold && \
    for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
# fatbins + native only: the default install also builds tf-cuda-test, which does not compile on 1.1.4 (contract_controls.cu, w4a8_tests.zig)
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast --prefix /opt/tensorfold \
      -j"$(nproc)" fatbins native && \
    test -x /opt/tensorfold/native/bin/tensorfold-native
RUN mkdir -p /opt/tensorfold/share/doc/tensorfold && cd /tensorfold && \
    for f in LICENSE LICENSE.md LICENSES NOTICE THIRD_PARTY_NOTICES.md; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done && \
    cp /mia/LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSE.mia && \
    cp /mia/NOTICE /opt/tensorfold/share/doc/tensorfold/NOTICE.mia

FROM ${BASE_IMAGE}
COPY --from=build /opt/tensorfold /opt/tensorfold
COPY --from=codecs /opt/codecs/jpeg/lib/libturbojpeg.so* /opt/tensorfold/lib/
COPY --from=codecs /opt/codecs/webp/lib/libwebpdecoder.so* /opt/tensorfold/lib/
# the engine looks for libtfgrammar.so in <binary dir>/../lib, i.e. next to native/bin
COPY --from=grammar /opt/tfrel/lib/libtfgrammar.so /opt/tensorfold/native/lib/libtfgrammar.so
COPY --from=grammar /opt/tfrel/LICENSES /opt/tensorfold/share/doc/tensorfold/LICENSES-grammar
RUN ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native
ARG PATCHES_HASH=da5e32ac724a
LABEL tf.patches=${PATCHES_HASH} tf.kernels=embedded
ENV HF_HOME=/root/.cache/huggingface CUDA_CACHE_PATH=/cache/nv \
    TENSORFOLD_LIBTURBOJPEG=/opt/tensorfold/lib/libturbojpeg.so.0 TENSORFOLD_LIBWEBP=/opt/tensorfold/lib/libwebpdecoder.so.3
WORKDIR /workspace