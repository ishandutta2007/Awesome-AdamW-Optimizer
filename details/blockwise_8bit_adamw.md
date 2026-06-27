# 8-Bit Block-Wise AdamW (BitsAndBytes)

8-Bit Block-Wise quantization dynamically compresses optimizer tensors down to 8-bit representations, minimizing memory storage overhead while performing mathematical operations in FP32.

## Key Concept
Instead of a single global scale factor, the tensor is partitioned into small blocks (e.g. size 2048). Each block has its own dynamic scale factor, which maintains the precision needed for adaptive moments.

## Compression Flow
```mermaid
graph TD
    A[FP32 Optimizer State] --> B[Partition into Blocks of 2048]
    B --> C[Compute scale factor for each Block]
    C --> D[Quantize Block values to INT8]
    D --> E[Save INT8 + Scale Factors to VRAM]
    E --> F[De-quantize on-the-fly to registers during step]
```

[← Back to README](../README.md)
