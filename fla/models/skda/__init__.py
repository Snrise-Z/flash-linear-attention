from fla.models.skda.configuration_skda import SKDAConfig
from fla.models.skda.modeling_skda import SKDAForCausalLM, SKDAModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register(SKDAConfig.model_type, SKDAConfig, exist_ok=True)
AutoModel.register(SKDAConfig, SKDAModel, exist_ok=True)
AutoModelForCausalLM.register(SKDAConfig, SKDAForCausalLM, exist_ok=True)

__all__ = ["SKDAConfig", "SKDAForCausalLM", "SKDAModel"]

