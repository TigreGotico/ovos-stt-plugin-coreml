import coremltools as ct
import numpy as np
import soundfile as sf
import json

# Load metadata
with open("parakeet_ctc_coreml/metadata.json") as f:
    meta = json.load(f)

SAMPLE_RATE = meta["sample_rate"]
MAX_SAMPLES = meta["max_audio_samples"]
BLANK_ID = meta["blank_id"]

# Load models
mel_encoder = ct.models.MLModel("parakeet_ctc_coreml/parakeet_ctc_mel_encoder.mlpackage")
ctc_decoder = ct.models.MLModel("parakeet_ctc_coreml/parakeet_ctc_decoder.mlpackage")

# Load and pad/trim audio
audio, sr = sf.read("yc_first_minute_16k_15s.wav", dtype="float32", always_2d=False)
assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE}Hz, got {sr}Hz"
original_len = len(audio)
if len(audio) < MAX_SAMPLES:
    audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))
else:
    audio = audio[:MAX_SAMPLES]

audio_signal = audio[np.newaxis, :].astype(np.float32)  # [1, N]
audio_length = np.array([min(original_len, MAX_SAMPLES)], dtype=np.int32)  # [1]

# Stage 1: Mel + Encoder
enc_out = mel_encoder.predict({
    "audio_signal": audio_signal,
    "audio_length": audio_length,
})
encoder = enc_out["encoder"]
encoder_length = enc_out["encoder_length"]
print(f"Encoder output shape: {encoder.shape}")  # [1, hidden, T]

# Stage 2: CTC Decoder
dec_out = ctc_decoder.predict({"encoder": encoder})
log_probs = dec_out["log_probs"]  # [1, T, vocab+1]
print(f"Log probs shape: {log_probs.shape}")

# Greedy decode
token_ids = np.argmax(log_probs[0], axis=-1)  # [T]

# CTC collapse (remove blanks and repeated tokens)
decoded = []
prev = None
for t in token_ids:
    if t != BLANK_ID and t != prev:
        decoded.append(int(t))
    prev = t

# Load vocab and decode
with open("vocab.json") as f:
    vocab = json.load(f)

text = "".join(vocab[i] for i in decoded).replace("", " ").strip()
print(f"Transcription: {text}")

