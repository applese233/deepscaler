pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0

# pip install verl==0.4.1
pip install -e ./verl
pip install -e .
# pip install -U vllm
pip uninstall vllm -y
pip install pynvml==12.0.0

pip3 install vllm==0.7.3
pip install tensordict==0.6.2
pip install antlr4-python3-runtime==4.9.3