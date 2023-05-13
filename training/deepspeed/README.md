# Vicuna Fine-Tuning

### done

- 数据的加载和处理应该已经调通. 

- 目前完成了 LoRA 版的 Vicuna Fine-Tuning 的初步代码. 

### todo

- 试试能不能把 Only apply LoRA 版的代码在4090上跑通 (很有可能跑不通). 
- 进一步 port 到 DeepSpeed 版, 并且 apply Zero-Stage 2, 并且在4090上跑通. 
- 在 DeepSpeed 上尝试 3D-Parallelism, 并且在4090上跑通. 
  - 这里 Pipeline Parallelism 和 Tensor Parallelism 应该是难点. 

> 在所有 todo 的任务里, 需要额外注意一下 Dockerhub 上是否有符合要求的镜像, 若没有就得自己写 Dockerfile 配. 或者, 可以通过 `startup-hook.sh` 额外配置需要的包. 
