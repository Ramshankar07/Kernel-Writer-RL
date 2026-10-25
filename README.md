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
