from fla.models.fkda.configuration_fkda import FKDAConfig
from fla.models.fkda.modeling_fkda import FKDAForCausalLM, FKDAModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register(FKDAConfig.model_type, FKDAConfig, exist_ok=True)
AutoModel.register(FKDAConfig, FKDAModel, exist_ok=True)
AutoModelForCausalLM.register(FKDAConfig, FKDAForCausalLM, exist_ok=True)

__all__ = ["FKDAConfig", "FKDAForCausalLM", "FKDAModel"]

