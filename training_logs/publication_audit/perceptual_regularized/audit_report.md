# Publication Audit Report

Overall status: `reviewer_risk_needs_documentation`

## Leakage Checks

- Manifest rows: 51023
- Duplicate manifest paths: 0
- Exact cross-split hash overlaps: 0
- Perceptual cross-split hash overlaps: 0
- Missing manifest files: 0

## Metric Risk Flags

- binary: accuracy 1.0000. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.
- dementia: accuracy 0.9975. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.

## Overfitting History

| Task | Epochs | Best Val Acc | Final Train Acc | Final Val Acc | Max Gap | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| binary | 5 | 1.0000 | 1.0 | 1.0 | 0.0000 |  |
| dementia | 17 | 0.9986 | 0.9961666831895839 | 0.9977356937011116 | -0.0004 |  |
| eight_class | 17 | 0.9987 | 0.9931528524986335 | 0.9979492915734527 | -0.0048 |  |
| tumor | 12 | 0.9842 | 0.9919964428634949 | 0.9802371541501976 | 0.0139 |  |

## Warnings

- Near-perfect metrics require leakage/source-bias and external-validity discussion.
- Binary router may learn dataset/domain artifacts because tumor and dementia images come from different source datasets.
