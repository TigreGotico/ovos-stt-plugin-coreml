#!/usr/bin/env python3
"""Export vocabulary and model metadata for Parakeet TDT v3 (EncDecRNNTBPEModel)."""
import json
from pathlib import Path

import nemo.collections.asr as nemo_asr

CHECKPOINT_PATH = "./parakeet-tdt-0.6b-v3.nemo"
OUTPUT_DIR = Path("parakeet_coreml")

print("Loading model...")
model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(CHECKPOINT_PATH, map_location="cpu")
model.eval()

# ── Vocabulary ────────────────────────────────────────────────────────────────
# SentencePieceTokenizer exposes .tokenizer (the SentencePieceProcessor)
sp = model.tokenizer.tokenizer
vocab_size = sp.get_piece_size()          # e.g. 1024 BPE tokens
blank_idx = model.decoder.blank_idx       # typically vocab_size (1024)

print(f"SentencePiece vocab size : {vocab_size}")
print(f"Blank token index        : {blank_idx}")

vocab_list = [sp.id_to_piece(i) for i in range(vocab_size)]
vocab_list.append("<blank>")              # blank always appended last

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
vocab_path = OUTPUT_DIR / "vocab.json"
with open(vocab_path, "w") as f:
    json.dump(vocab_list, f, ensure_ascii=False, indent=2)
print(f"Vocab saved to {vocab_path}  ({len(vocab_list)} entries incl. blank)")

# ── Model info ────────────────────────────────────────────────────────────────
num_extra = int(model.joint.num_extra_outputs)
decoder_hidden = int(model.decoder.pred_hidden)
decoder_layers = int(model.decoder.pred_rnn_layers)
sample_rate = int(model.cfg.preprocessor.sample_rate)

info = {
    "model_class": type(model).__name__,
    "vocab_size": vocab_size,
    "blank_idx": blank_idx,
    "joint_extra_outputs": num_extra,
    "decoder_hidden": decoder_hidden,
    "decoder_layers": decoder_layers,
    "sample_rate": sample_rate,
}

info_path = OUTPUT_DIR / "model_info.json"
with open(info_path, "w") as f:
    json.dump(info, f, indent=2)
print(f"Model info saved to {info_path}")
print(json.dumps(info, indent=2))
print("Done!")
