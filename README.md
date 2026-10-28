# TensorFold Flash-Next engine for Spark AI Hub

TensorFold's Zig engine (`tensorfold-native`) for **Qwen3.8-Flash-Next** on the NVIDIA DGX Spark
(aarch64, GB10 sm_121), as a pre-built image for [Spark AI Hub](https://github.com/WaxacaBytes/spark-ai-hub).

It is the engine of [Mia's AI Lab's single-Spark recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold),
built exactly as their `scripts/prepare.sh` builds it, at their commit `fd3f43e` (2026-10-27, patches 0001–0017,
NVIDIA FP4): NVIDIA's PyTorch 26.07 container, Zig 0.17.0, [TensorFold](https://github.com/ashhart/TensorFold)
[v1.1.4](https://github.com/ashhart/TensorFold/releases/tag/v1.1.4) (`eab8582`) with Mia's patches and its kernels
embedded in the binary, libjpeg-turbo and libwebp's decoder for image input, and `libtfgrammar.so` from TensorFold's
v1.1.4 release (structured output, `tool_choice: "required"`). Mia's recipe compiles this on every machine that runs it
and publishes no image of it; this repo builds it once, in CI, so the Hub's recipes pull it.

| Tag | |
|---|---|
| `v1.1.4-eab8582-da5e32ac724a` | the tag Mia's `scripts/config.sh` computes for this build (TensorFold commit + hash of the patches, Dockerfile and pins) |
| `zig-db28187-c0a2df4c680d` | the previous build (Mia `656879f`: TensorFold `zig-flashnext` db281878, patches 0001–0015), kept for recipes pinned to it |
| `latest` | the newest build from `main` |

**No weights in this image.** Each Hub recipe bakes its checkpoint in on top:

| Hub recipe | Checkpoint |
|---|---|
| `tensorfold-qwen38-flash-next-nvfp4` | [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), NVIDIA's quantization-aware-distilled FP4 (Mia's default) |

Serve it the way Mia's `start.sh` does: `tensorfold-native serve <checkpoint dir> --name … --context 262144 --parallel 5
--kv-cache fp8 --vision …` with the `TF_FLASHNEXT_*` / `TENSORFOLD_*` settings from their `scripts/config.sh`.

## Licenses

TensorFold is Apache-2.0 (code written before 0.6.0 also keeps its MIT notice); Mia's patches and scripts are MIT.
Their LICENSE and NOTICE files are in the image at `/opt/tensorfold/share/doc/tensorfold/`. The base image is NVIDIA's
PyTorch container, under NVIDIA's container license.
