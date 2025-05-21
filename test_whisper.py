import time
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import librosa
import numpy as np
from functools import lru_cache

@lru_cache(10**6)
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s]


pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo", # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
    torch_dtype=torch.float16,
    device="mps", # or mps for Mac devices
    model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
)

beg = 0
end = 2
for i in range(1, 31):
    audio_file = f"/Users/rohan/osllm/datasets/whisper-earnings21/2-sec-chunks/4320211_chunk_005_chunk_{i:03d}.wav"

    start = time.time()
    outputs = pipe(
        audio_file,
        batch_size=1,
        return_timestamps="word",
    )
    print(time.time() - start)

# audio_file = f"/Users/rohan/osllm/datasets/whisper-earnings21/4320211_chunk_005.wav"
# # audio = load_audio(audio_file)

# start = time.time()

# outputs = pipe(
#     audio_file,
#     # chunk_length_s=30,
#     batch_size=1,
#     return_timestamps="word",
# )
# # print(outputs)
# print(time.time() - start)
# # print(outputs)