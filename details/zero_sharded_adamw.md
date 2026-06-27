# ZeRO-Sharded AdamW (DeepSpeed / FSDP)

ZeRO (Zero Redundancy Optimizer) eliminates memory redundancies in distributed training by sharding model states across GPU nodes instead of replicating them.

## Three Stages of ZeRO
1. **ZeRO-Stage 1:** Shards the optimizer states (e.g. AdamW Master Weights and Moments) across GPUs.
2. **ZeRO-Stage 2:** Shards both optimizer states and gradients.
3. **ZeRO-Stage 3:** Shards optimizer states, gradients, and model parameters.

## Sharding Layout
```mermaid
graph TD
    subgraph Replicated (Baseline)
    A[GPU 0: Model, Grad, Opt]
    B[GPU 1: Model, Grad, Opt]
    end
    subgraph ZeRO-Stage 1 (Sharded Optimizer)
    C[GPU 0: Model, Grad, Opt Part 1]
    D[GPU 1: Model, Grad, Opt Part 2]
    end
```

[← Back to README](../README.md)
