# Sentinel's Journal

## 2025-02-18 - Insecure Model Serialization
**Vulnerability:** The application saves and loads PyTorch models as full objects using `torch.save(model)` and `torch.load(model, weights_only=False)`. This uses Python's `pickle` module, which is vulnerable to arbitrary code execution if the model file is malicious.
**Learning:** PyTorch's default `torch.save` stores the entire object hierarchy. Loading it requires `pickle`, which is unsafe.
**Prevention:** Always save model weights using `model.state_dict()` and load them using `model.load_state_dict()`. Use `weights_only=True` when loading to restrict the unpickler to safe types.
