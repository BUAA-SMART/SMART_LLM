# SMART_LLM
the repo for building SMART_LLM

## Vicuna Training

### done

- 数据的加载和处理应该已经调通. 

- 目前完成了 LoRA 版的 Vicuna 7B Fine-Tuning, 可以在Deepspeed Stage 2和Stage 3上进行ft，具体配置可以参考[vicuna_ds_config_2n8u.yaml](training/deepspeed/vicuna_ds_config_2n8u.yaml)
