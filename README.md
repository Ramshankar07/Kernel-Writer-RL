# Kernel-Writer-RL-v2

An LLM agent (Qwen2.5-Coder-7B-Instruct) writes Triton kernels. A GPU benchmark checks correctness and speed, and the result is the reward for GRPO. April 2026.

**Write-up:** [What I learned training an LLM to write GPU kernels with GRPO](https://ramshankar07.github.io/Kernel-Writer-RL-v2/)

## Results

| | Before | After |
|---|---|---|
| Validation pass@1, 62 problems (SFT) | 0.383 | 0.508 |
| KernelBench L1, correct Triton kernels (SFT) | 0/100 | 20/100 |
| Triton compile errors per turn (SFT) | 8.1% | 1.9% |
| Speedup vs starter kernel (SFT) | 0.950× | 1.003× |

GRPO did not improve the model: most groups had no reward variance, and 78.9% of the advantage came from timing noise.
Sources: [`results/v3/phase3.md`](results/v3/phase3.md), [`results/extra_numbers.json`](results/extra_numbers.json).

## AMD MI300X

The same loop also targets AMD Instinct MI300X (CDNA3, gfx942) and was tested on RunPod MI300X pods.
Those run logs are not in this repo, so the results above are NVIDIA only.

| File | Role |
|---|---|
| [`bench_worker/worker_amd.py`](src/autokernel-rlvr/bench_worker/worker_amd.py) | ROCm bench worker |
| [`agent_loop/reward_amd.py`](src/autokernel-rlvr/agent_loop/reward_amd.py) | log2-speedup reward; clip raised to 4.0 for FP8 kernels |
| [`data/build_amd_dataset.py`](src/autokernel-rlvr/data/build_amd_dataset.py) | Prompts from GPU MODE AMD challenge submissions (FP8 GEMM, MoE, MLA decode) |
| [`hardware/mi300x_specs.py`](src/autokernel-rlvr/hardware/mi300x_specs.py) | MI300X numbers given to the model |

What changes for a kernel writer on MI300X: 64-thread wavefronts instead of 32-thread warps, 304 CUs,
64 KB LDS per CU, a 256 MB Infinity Cache, 5.3 TB/s HBM3, and FP8 at 2,615 dense TFLOPs (2× BF16).

## Layout

| Path | Contents |
|---|---|
| [`modal_app/`](modal_app/) | Benchmarks, vLLM policy, GRPO trainer, SFT pipeline (runs on [Modal](https://modal.com)) |
| [`src/`](src/) | SkyPilot + verl design |
| [`results/`](results/) | All measured numbers and reports |
| [`post/`](post/) | Source of the write-up |

## Run

```bash
pip install modal && modal setup
cd modal_app && modal deploy bench_modal.py && modal deploy policy.py
modal run agent_eval.py --suite autokernel --n 8 --max-turns 8
```

## License

Apache-2.0. Vendored AutoKernel code in `results/autokernel_src/` is MIT.
