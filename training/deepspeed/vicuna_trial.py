import logging
import typing
import torch
from typing import Any, Dict, Union, Iterator, Optional
from dataclasses import dataclass, field

import deepspeed

import determined.pytorch as det_pytorch
import transformers
from attrdict import AttrDict
from peft import LoraConfig, get_peft_model
from determined.pytorch.deepspeed import (
    DeepSpeedTrial,
    DeepSpeedTrialContext,
    overwrite_deepspeed_config
)
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


class VicunaTrial(DeepSpeedTrial):
    def __init__(self, context: DeepSpeedTrialContext) -> None:
        self.logger = logging.getLogger(__name__)
        super(VicunaTrial, self).__init__(context)
        self.context = context
        self.args = AttrDict(self.context.get_hparams())

        # read
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.args.get('pretrained_model_name_or_path'),
            cache_dir=self.args.get('cache_dir'),
            padding_side='right',
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.args.get('pretrained_model_name_or_path'),
            cache_dir=self.args.get('cache_dir'),
        )

        lora_config = LoraConfig(
            r=self.args.get('lora_r'),
            lora_alpha=self.args.get('lora_alpha'),
            # target_modules=LoraArguments.lora_target_modules,
            target_modules = ["q_proj", "v_proj"],
            lora_dropout=self.args.get('lora_dropout'),
            bias=LoraArguments.bias,
            task_type='CAUSAL_LM',
        )
        self.model = get_peft_model(self.model, lora_config)
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        ds_config = overwrite_deepspeed_config(
            self.args.deepspeed_config, self.args.get('overwrite_deepspeed_args', {})
        )

        model_engine, optimizer, __, __ = deepspeed.initialize(
            model=self.model, model_parameters=parameters, config=ds_config
        )

        self.fp16 = model_engine.fp16_enabled()
        self.model_engine = self.context.wrap_model_engine(model_engine)

        self.data_processors = data
        self.data_config = self.context.get_data_config()
        self.tokenized_datasets = self.data_processors.make_supervised_data_module(
            self.tokenizer, self.data_config
        )

        self.reducer = self.context.wrap_reducer(
            lambda losses: np.exp(np.mean(losses)), name="perplexity", for_training=False
        )


    def build_training_data_loader(self) -> det_pytorch.DataLoader:
        return det_pytorch.DataLoader(
            self.tokenized_datasets['train_dataset'],
            batch_size=self.context.train_micro_batch_size_per_gpu,
            shuffle=True
        )

    def build_validation_data_loader(self) -> det_pytorch.DataLoader:
        return det_pytorch.DataLoader(
            self.tokenized_datasets['eval_dataset'],
            batch_size=self.context.train_micro_batch_size_per_gpu
        )

    def train_batch(
        self,
        dataloader_iter: Optional[Iterator[det_pytorch.TorchData]],
        epoch_idx: int,
        batch_idx: int,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        fetch_data_dict = next(dataloader_iter)
        inputs = {}
        for x in fetch_data_dict:
            inputs[x] = self.context.to_device(fetch_data_dict[x])
            # if self.fp16:
            #     inputs[x]  = inputs[x].half()

        # inputs = self.context.to_device(next(dataloader_iter))
        # if self.fp16:
        #     inputs = inputs.half()
        outputs = self.model_engine(**inputs)
        loss = outputs.loss
        print("input_ids")
        print(inputs["input_ids"])
        print("labels")
        print(inputs["labels"])
        print(loss)
        self.model_engine.backward(loss)
        self.model_engine.step()
        return {'loss': loss.item()}

    def evaluate_batch(
        self, dataloader_iter: Optional[Iterator[det_pytorch.TorchData]], batch_idx: int
    ) -> Dict[str, Any]:
        fetch_data_dict = next(dataloader_iter)
        inputs = {}
        for x in fetch_data_dict:
            inputs[x] = self.context.to_device(fetch_data_dict[x])
        # inputs = self.context.to_device(next(dataloader_iter))
        
        # if self.fp16:
        #     inputs = inputs.half()
        outputs = self.model_engine(**inputs)
        loss = outputs.loss
        self.reducer.update(loss.detach().cpu().numpy())
        return {}

