from fla.models.fskda.configuration_fskda import FSKDAConfig
from fla.models.fskda.modeling_fskda import FSKDAForCausalLM, FSKDAModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register(FSKDAConfig.model_type, FSKDAConfig, exist_ok=True)
AutoModel.register(FSKDAConfig, FSKDAModel, exist_ok=True)
AutoModelForCausalLM.register(FSKDAConfig, FSKDAForCausalLM, exist_ok=True)

__all__ = ["FSKDAConfig", "FSKDAForCausalLM", "FSKDAModel"]

