# TensorFold Flash-Next engine for Spark AI Hub

TensorFold's Zig engine (`tensorfold-native`) for **Qwen3.8-Flash-Next** on the NVIDIA DGX Spark
(aarch64, GB10 sm_121), as a pre-built image for [Spark AI Hub](https://github.com/WaxacaBytes/spark-ai-hub).

It is the engine of [Mia's AI Lab's single-Spark recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold),
built exactly as their `scripts/prepare.sh` builds it, at their commit `656879f` (patches 0001–0015, with NVIDIA FP4
support): NVIDIA's PyTorch 26.07 container, Zig 0.17.0, [TensorFold](https://github.com/ashhart/TensorFold)'s
`zig-flashnext` branch at `db281878` with Mia's patches, and the TP=1 CUDA kernel set compiled ahead of time for sm_121.
Mia's recipe compiles this on every machine that runs it, since their published image stops at an older patch set;
this repo builds it once, in CI, so the Hub's recipes pull it.

| Tag | |
|---|---|
| `zig-db28187-c0a2df4c680d` | the tag Mia's `scripts/config.sh` computes for this build (TensorFold commit + hash of the patches and Dockerfile) |
| `latest` | the same, from `main` |

**No weights in this image.** Each Hub recipe bakes its checkpoint in on top:

| Hub recipe | Checkpoint |
|---|---|
| `tensorfold-qwen38-flash-next-nvfp4` | [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), NVIDIA's quantization-aware-distilled FP4 (Mia's default) |

Serve it the way Mia's `start.sh` does: `tensorfold-native serve <checkpoint dir> --name … --context 262144 --parallel 5
--kv-dtype fp8 --vision …` with the `TF_FLASHNEXT_*` / `TENSORFOLD_*` settings from their `scripts/config.sh`.

## Licenses

TensorFold is Apache-2.0 (code written before 0.6.0 also keeps its MIT notice); Mia's patches and scripts are MIT.
Their LICENSE and NOTICE files are in the image at `/opt/tensorfold/share/doc/tensorfold/`. The base image is NVIDIA's
PyTorch container, under NVIDIA's container license.
