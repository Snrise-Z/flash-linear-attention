from fla.models.snkda.configuration_snkda import SNKDAConfig
from fla.models.snkda.modeling_snkda import SNKDAForCausalLM, SNKDAModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register(SNKDAConfig.model_type, SNKDAConfig, exist_ok=True)
AutoModel.register(SNKDAConfig, SNKDAModel, exist_ok=True)
AutoModelForCausalLM.register(SNKDAConfig, SNKDAForCausalLM, exist_ok=True)

__all__ = ["SNKDAConfig", "SNKDAForCausalLM", "SNKDAModel"]

