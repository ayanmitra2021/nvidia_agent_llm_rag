"""Model access follows the credential reference owned by the deployment."""

MANAGED_NEBIUS_MODELS = frozenset({
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "nvidia/Nemotron-3_5-Lightning",
    "nvidia/nemotron-3-super-120b-a12b",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
})


def allowed_models(entry):
    if entry["api_key_env"] in {"NEBIUS_TOKEN_FACTORY_KEY", "NEBUIS_TOKEN_FACTORY_KEY"}:
        return MANAGED_NEBIUS_MODELS
    return None


def model_allowed(provider, model):
    if not isinstance(model, str) or not model:
        return False
    permitted = provider[3]
    return permitted is None or provider[1].get(model, model) in permitted
