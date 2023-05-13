apt-get update && apt-get install -y wget
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install peft
python3 -m pip install --upgrade transformers
pip install datasets
pip install sentencepiece
ds_report