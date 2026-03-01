import torch
import nemo.collections.asr as nemo_asr
import numpy as np
import json

# ─── Config ───────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "./parakeet-tdt-ctc.nemo"
# ──────────────────────────────────────────────────────────────────────────────

print("Loading model...")
model = nemo_asr.models.ASRModel.restore_from(CHECKPOINT_PATH)
model.eval()
model.freeze()


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
