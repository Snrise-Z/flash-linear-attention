from fla.models.fnkda.configuration_fnkda import FNKDAConfig
from fla.models.fnkda.modeling_fnkda import FNKDAForCausalLM, FNKDAModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register(FNKDAConfig.model_type, FNKDAConfig, exist_ok=True)
AutoModel.register(FNKDAConfig, FNKDAModel, exist_ok=True)
AutoModelForCausalLM.register(FNKDAConfig, FNKDAForCausalLM, exist_ok=True)

__all__ = ["FNKDAConfig", "FNKDAForCausalLM", "FNKDAModel"]

