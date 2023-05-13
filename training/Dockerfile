FROM  determinedai/environments:cuda-11.3-pytorch-1.10-deepspeed-0.8.3-gpu-0.21.2
RUN apt-get update && apt-get install -y wget
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install peft
RUN mkdir /workspace
COPY model_hub /workspace/model_hub
RUN cd /workspace 
RUN pip install -e model_hub/.
RUN pip install torch==1.9.0

