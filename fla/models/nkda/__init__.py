from fla.models.nkda.configuration_nkda import NKDAConfig
from fla.models.nkda.modeling_nkda import NKDAForCausalLM, NKDAModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register(NKDAConfig.model_type, NKDAConfig, exist_ok=True)
AutoModel.register(NKDAConfig, NKDAModel, exist_ok=True)
AutoModelForCausalLM.register(NKDAConfig, NKDAForCausalLM, exist_ok=True)

__all__ = ["NKDAConfig", "NKDAForCausalLM", "NKDAModel"]

