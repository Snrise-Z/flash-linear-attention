from fla.models.fsnkda.configuration_fsnkda import FSNKDAConfig
from fla.models.fsnkda.modeling_fsnkda import FSNKDAForCausalLM, FSNKDAModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register(FSNKDAConfig.model_type, FSNKDAConfig, exist_ok=True)
AutoModel.register(FSNKDAConfig, FSNKDAModel, exist_ok=True)
AutoModelForCausalLM.register(FSNKDAConfig, FSNKDAForCausalLM, exist_ok=True)

__all__ = ["FSNKDAConfig", "FSNKDAForCausalLM", "FSNKDAModel"]

