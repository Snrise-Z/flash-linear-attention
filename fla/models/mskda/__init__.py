# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.mskda.configuration_mskda import MSKDAConfig
from fla.models.mskda.modeling_mskda import MSKDAForCausalLM, MSKDAModel

AutoConfig.register(MSKDAConfig.model_type, MSKDAConfig, exist_ok=True)
AutoModel.register(MSKDAConfig, MSKDAModel, exist_ok=True)
AutoModelForCausalLM.register(MSKDAConfig, MSKDAForCausalLM, exist_ok=True)

__all__ = ["MSKDAConfig", "MSKDAForCausalLM", "MSKDAModel"]
