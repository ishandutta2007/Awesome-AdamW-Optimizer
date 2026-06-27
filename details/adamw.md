# The Decoupled Weight Decay Revolution (AdamW)

AdamW (2017) fixes the fundamental weight decay coupling bug in Standard Adam by decoupling the weight decay step from the gradient-dependent adaptive step update.

## The Bug Fix
Instead of regularizing the gradient vector $g_t$ inside the moment calculations, AdamW directly decays the weight before or after applying the gradient update:

$$\theta_t = \theta_{t-1} - \eta_t \lambda \theta_{t-1} - \frac{\eta_t}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

This restores the behavior of $L_2$ regularization exactly as intended, improving generalization and convergence.

## Process Flow Comparison
```mermaid
graph TD
    subgraph Adam (Coupled)
    A1[Gradient + L2 Penalty] --> B1[Calculate Moments] --> C1[Update Weights]
    end
    subgraph AdamW (Decoupled)
    A2[Gradient Only] --> B2[Calculate Moments] --> C2[Update Weights]
    A2 --> D2[Apply Weight Decay Directly]
    C2 & D2 --> E2[Final Weight Update]
    end
```

[← Back to README](../README.md)
