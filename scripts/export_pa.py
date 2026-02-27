import torch
import nemo.collections.asr as nemo_asr
import numpy as np
import json

# ─── Config ───────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "./parakeet-tdt-ctc.nemo"
OUTPUT_ENCODER  = "./parakeet_encoder.pt"
OUTPUT_DECODER  = "./parakeet_ctc_decoder.pt"
SAMPLE_RATE     = 16000
AUDIO_LEN_SEC   = 5
# ──────────────────────────────────────────────────────────────────────────────

print("Loading model...")
model = nemo_asr.models.ASRModel.restore_from(CHECKPOINT_PATH)
model.eval()
model.freeze()

# ─── Preprocessor ─────────────────────────────────────────────────────────────
dummy_audio  = torch.zeros(1, SAMPLE_RATE * AUDIO_LEN_SEC)
dummy_length = torch.tensor([SAMPLE_RATE * AUDIO_LEN_SEC])

with torch.no_grad():
    processed, proc_len = model.preprocessor(
        input_signal=dummy_audio,
        length=dummy_length
    )
print(f"Processed shape: {processed.shape}")  # [1, 80, 501]

# ─── Trace Encoder ────────────────────────────────────────────────────────────
class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, mel, mel_len):
        encoded, encoded_len = self.encoder(audio_signal=mel, length=mel_len)
        return encoded, encoded_len

encoder_wrapper = EncoderWrapper(model.encoder)

with torch.no_grad():
    traced_encoder = torch.jit.trace(encoder_wrapper, (processed, proc_len))
traced_encoder.save(OUTPUT_ENCODER)
print(f"Encoder saved to {OUTPUT_ENCODER}")

# ─── Get encoder output for decoder tracing ───────────────────────────────────
with torch.no_grad():
    encoded, enc_len = encoder_wrapper(processed, proc_len)
print(f"Encoded shape: {encoded.shape}")  # [1, hidden, T']

# ─── Inspect the CTC decoder ──────────────────────────────────────────────────
# model.ctc_decoder is a ConvASRDecoder (simple linear layer over encoder output)
print(f"CTC decoder type: {type(model.ctc_decoder)}")
print(f"CTC decoder: {model.ctc_decoder}")

# ConvASRDecoder.forward signature: forward(encoder_output)
# It expects [B, hidden, T] and returns [B, T, vocab]
class CTCDecoderWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, encoded):
        # ConvASRDecoder returns log_probs: [B, T, vocab]
        log_probs = self.decoder(encoder_output=encoded)
        return log_probs

ctc_wrapper = CTCDecoderWrapper(model.ctc_decoder)

with torch.no_grad():
    # Verify it works before tracing
    test_out = ctc_wrapper(encoded)
    print(f"CTC output shape: {test_out.shape}")  # [1, T', vocab_size]

    traced_decoder = torch.jit.trace(ctc_wrapper, (encoded,))

traced_decoder.save(OUTPUT_DECODER)
print(f"CTC Decoder saved to {OUTPUT_DECODER}")


# ─── Save vocabulary ──────────────────────────────────────────────────────────
# For EncDecHybridRNNTCTCBPEModel, use the decoding vocab directly from the CTC decoder
# The decoder output is 1025: 1024 BPE tokens + 1 blank (index 1024)

vocab_size = model.ctc_decoder.decoder_layers[0].out_channels  # 1025
print(f"Decoder vocab size (with blank): {vocab_size}")

# Get the token list from the tokenizer
# SentencePieceTokenizer exposes .tokenizer (the sentencepiece model)
sp = model.tokenizer.tokenizer  # SentencePieceProcessor

vocab_list = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
print(f"SentencePiece vocab size: {len(vocab_list)}")  # should be 1024

# Append blank token at the end (CTC blank = last index for NeMo hybrid models)
vocab_list.append("<blank>")
print(f"Final vocab list size: {len(vocab_list)}")  # 1025

with open("vocab.json", "w") as f:
    json.dump(vocab_list, f, ensure_ascii=False, indent=2)
print("Vocab saved to vocab.json")

print("Done!")
