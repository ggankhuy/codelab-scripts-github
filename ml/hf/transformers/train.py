from transformers import AutoConfig
from transformers import pipeline
config = AutoConfig.from_pretrained("/root/.cache/huggingface/hub/models--bigscience--T0_3B/snapshots/8794c7177e3a67b8a0ec739d94eecfa6a591c974/config.json")
generator = pipeline(task="automatic-speech-recognition")
generator("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': 'I HAVE A DREAM BUT ONE DAY THIS NATION WILL RISE UP LIVE UP THE TRUE MEANING OF ITS TREES'}
