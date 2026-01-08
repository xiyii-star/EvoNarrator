## Table 1: Overall Performance Comparison

| Method | Accuracy | Macro F1 | Weighted F1 |
|--------|----------|----------|-------------|
| Baseline (Abstract Only) | 0.2826 | 0.1506 | 0.3561 |
| SocketMatch (Deep Info + Logic) | 0.7174 | 0.4615 | 0.7252 |

## Table 2: Per-Class F1-Score Comparison

| Citation Type | Baseline (Abstract Only) | SocketMatch (Deep Info + Logic) | Improvement |
|---------------|----------|----------|-------------|
| **Overcomes** | 0.0690 | 0.3333 | +0.2644 (+383.3%) |
| **Realizes** | 0.0000 | 0.2069 | +0.2069 (+0.0%) |
| **Extends** | 0.1053 | 0.5714 | +0.4662 (+442.9%) |
| **Alternative** | 0.1270 | 0.3529 | +0.2260 (+177.9%) |
| **Adapts_to** | 0.1519 | 0.4516 | +0.2997 (+197.3%) |
| **Baselines** | 0.4502 | 0.8529 | +0.4026 (+89.4%) |
| **Average** | **0.1506** | **0.4615** | **+0.3110 (+206.5%)** |
