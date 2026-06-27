# Distributed Parameter-Efficient Alignment Sprints (LoRA / QLoRA)

For post-training, parameter-efficient alignment (like Low-Rank Adaptation - LoRA) reduces the parameter update footprint. When combined with quantized 8-bit AdamW, training memory overhead is reduced dramatically.

## Mechanics
- **Base Model:** Kept frozen (optionally 4-bit quantized in QLoRA).
- **Adapters:** Small trainable matrices (A and B).
- **Optimizer:** AdamW updates only adapter parameters, dramatically shrinking VRAM.

## Parameter Flow
```mermaid
graph TD
    A[Input X] --> B[Frozen Base Model Weight W]
    A --> C[Adapter Down-projection A]
    C --> D[Adapter Up-projection B]
    B --> E[Output sum: W*X + B*A*X]
    E --> F[Optimizer: Updates only A & B via AdamW]
```

[← Back to README](../README.md)
