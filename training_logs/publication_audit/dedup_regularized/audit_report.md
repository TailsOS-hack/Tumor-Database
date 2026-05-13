# Publication Audit Report

Overall status: `reviewer_risk_needs_documentation`

## Leakage Checks

- Manifest rows: 51023
- Duplicate manifest paths: 0
- Exact cross-split hash overlaps: 0
- Perceptual cross-split hash overlaps: 22304
- Missing manifest files: 0

## Metric Risk Flags

- binary: accuracy 1.0000. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.
- tumor: accuracy 0.9962. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.
- dementia: accuracy 0.9985. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.
- hierarchical: accuracy 0.9982. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.
- eight_class: accuracy 0.9993. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.
- binary: accuracy 1.0000. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.
- dementia: accuracy 0.9991. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.
- eight_class: accuracy 0.9972. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.
- hierarchical: accuracy 0.9963. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.

## Overfitting History

| Task | Epochs | Best Val Acc | Final Train Acc | Final Val Acc | Max Gap | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| binary | 5 | 1.0000 | 1.0 | 1.0 | 0.0000 |  |
| dementia | 16 | 0.9989 | 0.9964604643610976 | 0.9984087292566493 | -0.0005 |  |
| eight_class | 17 | 0.9982 | 0.9940256839754327 | 0.9979806138933764 | -0.0040 |  |
| tumor | 10 | 0.9855 | 0.9924378109452736 | 0.9819168173598554 | 0.0105 |  |

## Warnings

- Perceptual hash overlap across splits needs manual review.
- Near-perfect metrics require leakage/source-bias and external-validity discussion.
- Binary router may learn dataset/domain artifacts because tumor and dementia images come from different source datasets.
