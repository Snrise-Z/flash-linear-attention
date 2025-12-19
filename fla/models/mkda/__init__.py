
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.mkda.configuration_mkda import MKDAConfig
from fla.models.mkda.modeling_mkda import MKDAForCausalLM, MKDAModel

AutoConfig.register(MKDAConfig.model_type, MKDAConfig, exist_ok=True)
AutoModel.register(MKDAConfig, MKDAModel, exist_ok=True)
AutoModelForCausalLM.register(MKDAConfig, MKDAForCausalLM, exist_ok=True)

__all__ = ["MKDAConfig", "MKDAForCausalLM", "MKDAModel"]

