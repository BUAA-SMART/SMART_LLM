import deepspeed as ds
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import determined as det
from determined.experimental import client
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass, field
import typing
from typing import Any, Dict, Union, Iterator, Optional


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    bias: str = "none"
    

model_entry = client.get_model("vicuna_7B_MOSS_SFT_lora")
version = model_entry.get_version(4)
ckpt = version.checkpoint
path = ckpt.download()



state_dict = get_fp32_state_dict_from_zero_checkpoint(path)

pretrained_model_name_or_path = "/root/model"
model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name_or_path,        
)
lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            # target_modules=LoraArguments.lora_target_modules,
            target_modules = ["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias=LoraArguments.bias,
            task_type='CAUSAL_LM',
        )

model = get_peft_model(model, lora_config)
model.load_state_dict(state_dict)
model.save_pretrained("/home/nfs_data/models/vicuna-ft-MOSS-honesty/7B")
