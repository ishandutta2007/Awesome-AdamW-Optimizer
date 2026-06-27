# Learning Rate Warmup and Schedule Dependency

At the beginning of training, adaptive moments are uninitialized (zero). Standard schedules with high initial learning rates can cause unstable and divergent updates.

## Mitigation
**Linear Warmup:** Linearly scales the learning rate from 0 to its maximum value over a set number of startup steps (e.g. 2000 steps). After warmup, a standard scheduler (like Cosine Annealing) takes over.

## Warmup Schedule Plot
```mermaid
xychart-beta
    title "Learning Rate Warmup & Cosine Decay"
    x-axis [0, 1000, 2000, 4000, 6000, 8000, 10000]
    y-axis "Learning Rate Scale" 0.0 --> 1.0
    line [0.0, 0.5, 1.0, 0.85, 0.5, 0.15, 0.0]
```

[← Back to README](../README.md)
