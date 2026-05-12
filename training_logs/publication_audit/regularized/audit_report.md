# Publication Audit Report

Overall status: `blocking_leakage_risk`

## Leakage Checks

- Manifest rows: 51023
- Duplicate manifest paths: 0
- Exact cross-split hash overlaps: 782
- Perceptual cross-split hash overlaps: 22537
- Missing manifest files: 0

## Metric Risk Flags

- binary: accuracy 1.0000. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.
- tumor: accuracy 0.9962. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.
- dementia: accuracy 0.9985. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.
- hierarchical: accuracy 0.9982. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.
- eight_class: accuracy 0.9993. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.
- binary: accuracy 1.0000. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.
- dementia: accuracy 0.9992. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.
- eight_class: accuracy 0.9984. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.
- experiments_regularized: accuracy 0.9985. Accuracy >= 0.995 needs explicit leakage/source-bias and external-validity discussion.

## Overfitting History

| Task | Epochs | Best Val Acc | Final Train Acc | Final Val Acc | Max Gap | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| binary | 5 | 1.0000 | 1.0 | 0.999597747385358 | 0.0005 |  |
| dementia | 20 | 0.9993 | 0.997012987012987 | 0.9986363636363637 | 0.0097 |  |
| eight_class | 19 | 0.9990 | 0.9944629938786868 | 0.998793242156074 | -0.0043 |  |
| tumor | 15 | 0.9930 | 0.9949416342412452 | 0.9877622377622378 | 0.0072 |  |

## Warnings

- Perceptual hash overlap across splits needs manual review.
- Near-perfect metrics require leakage/source-bias and external-validity discussion.
- Binary router may learn dataset/domain artifacts because tumor and dementia images come from different source datasets.
