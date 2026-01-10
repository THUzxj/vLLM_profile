apt update
apt install -y python3 python3-pip

pip install uv

uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install matplotlib pandas
uv pip install vllm==0.11.0
