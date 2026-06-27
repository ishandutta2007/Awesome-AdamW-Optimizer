# Adafactor: Low-Memory Optimization

Adafactor is an adaptive optimizer designed to drastically reduce memory usage by factoring the second moment matrix $v_t$, which stores the moving average of squared gradients.

## How it Works
Rather than storing $O(N)$ elements for parameter updates, Adafactor factors the $N \times M$ matrix of second moments into two smaller matrices:
- $R$ of size $N \times 1$ (row-wise sum)
- $C$ of size $1 \times M$ (column-wise sum)

This reduces memory requirements from $O(N \cdot M)$ to $O(N + M)$.

## Memory Footprint Reduction
```mermaid
graph TD
    subgraph Standard Adam (O(N * M))
    A[N x M Parameter Matrix] --> B[N x M Second Moment Tensor]
    end
    subgraph Adafactor (O(N + M))
    C[N x M Parameter Matrix] --> D[N x 1 Row Sum]
    C --> E[1 x M Column Sum]
    end
```

[← Back to README](../README.md)
