import logging
import typing
from typing import Any, Dict
from dataclasses import dataclass, field

import model_hub.huggingface as hf
import determined.pytorch as det_pytorch
import transformers
from peft import LoraConfig, get_peft_model
import numpy as np

import data


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


class VicunaTrial(hf.BaseTransformerTrial):
    def __init__(self, context: det_pytorch.PyTorchTrialContext) -> None:
        self.logger = logging.getLogger(__name__)
        super(VicunaTrial, self).__init__(context)
        self.logger.info(self.config)
        self.args = self.context.get_hparams()

        # read
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.args.get('pretrained_model_name_or_path'),
            cache_dir=self.args.get('cache_dir'),
            padding_side='right',
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token

        self.model = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.args.get('pretrained_model_name_or_path'),
            cache_dir=self.args.get('cache_dir'),
        )
        lora_config = LoraConfig(
            r=self.args.get('lora_r'),
            lora_alpha=self.args.get('lora_alpha'),
            target_modules=LoraArguments.lora_target_modules,
            lora_dropout=self.args.get('lora_dropout'),
            bias=LoraArguments.bias,
            task_type='CAUSAL_LM',
        )
        self.model = get_peft_model(self.model, lora_config)

        self.data_processors = data
        self.tokenized_datasets = self.data_processors.make_supervised_data_module(
            self.tokenizer, self.data_config)

        self.reducer = self.context.wrap_reducer(
            lambda losses: np.exp(np.mean(losses)), name="perplexity", for_training=False
        )
        self.model = self.context.wrap_model(self.model)


    def build_training_data_loader(self) -> det_pytorch.DataLoader:
        return det_pytorch.DataLoader(
            self.tokenized_datasets['train_dataset'],
            batch_size=self.context.get_per_slot_batch_size()
        )


    def build_validation_data_loader(self) -> det_pytorch.DataLoader:
        return det_pytorch.DataLoader(
            self.tokenized_datasets['eval_dataset'],
            batch_size=self.context.get_per_slot_batch_size()
        )


    def evaluate_batch(self, batch: det_pytorch.TorchData, batch_idx: int) -> Dict[str, Any]:
        outputs = self.model(**batch)
        self.reducer.update(outputs[0].detach().cpu().numpy())
        return {}





