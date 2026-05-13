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

## Overfitting History

- Training histories were not available in the inspected experiments directory.

## Warnings

- Perceptual hash overlap across splits needs manual review.
- Near-perfect metrics require leakage/source-bias and external-validity discussion.
- Training histories were not available, so overfitting gaps could not be audited locally.
- Binary router may learn dataset/domain artifacts because tumor and dementia images come from different source datasets.
