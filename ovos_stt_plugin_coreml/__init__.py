import json
from pathlib import Path
from typing import List, Optional, Tuple

import coremltools as ct
import numpy as np
from ovos_plugin_manager.templates.stt import STT
from ovos_plugin_manager.utils.audio import AudioData, AudioFile
from ovos_utils import classproperty

from ovos_stt_plugin_coreml.lm import ARPALanguageModel, ctc_beam_search


class CoremlSTT(STT):
    """Generic OVOS STT plugin for CoreML-exported speech recognition models.

    Supports two model families detected automatically from metadata.json:

      • ctc  – Parakeet CTC / Hybrid RNNT-CTC (EncDecHybridRNNTCTCBPEModel)
               Pipeline: mel_encoder → ctc_decoder → log_probs → greedy / beam search

      • tdt  – Parakeet TDT v3 pure RNNT (EncDecRNNTBPEModel)
               Pipeline: mel_encoder (once) → per-frame decoder + joint decision step

    Detection logic (checked in order):
      1. config["model_type"] = "ctc" or "tdt" — explicit override
      2. "ctc_decoder" in metadata components → ctc
      3. "joint"       in metadata components → tdt
      4. "blank_id"    in metadata top-level  → ctc  (legacy CTC metadata)
      5. Default: ctc

    Explicit config (all paths set manually — metadata still required for
    sample_rate / max_audio_samples):

        # CTC
        config = {
            "metadata": "parakeet_ctc_coreml/metadata.json",
            "vocab":    "parakeet_ctc_coreml/vocab.json",
            "encoder":  "parakeet_ctc_coreml/parakeet_ctc_mel_encoder.mlpackage",
            "decoder":  "parakeet_ctc_coreml/parakeet_ctc_decoder.mlpackage",
            "lm":         "/path/to/language_model.arpa",
            "lm_weight":  0.3,
            "word_bonus": 1.0,
            "beam_width": 100,
        }

        # TDT
        config = {
            "metadata": "parakeet_tdt_coreml/metadata.json",
            "vocab":    "parakeet_tdt_coreml/vocab.json",
            "encoder":  "parakeet_tdt_coreml/mel_encoder.mlpackage",
            "decoder":  "parakeet_tdt_coreml/decoder.mlpackage",
            "joint_decision_single_step": "parakeet_tdt_coreml/joint_decision_single_step.mlpackage",
            "max_symbols_per_step": 10,
        }

    Minimal config — component paths and vocab are resolved from the metadata
    directory automatically when not set explicitly:

        {"metadata": "/path/to/parakeet_coreml/metadata.json"}

    Full config reference:

      Common:
        metadata    (required) – path to metadata.json
        model_type  (optional) – "ctc" or "tdt"; auto-detected when omitted
        vocab       (optional) – path to vocab.json; defaults to <metadata_dir>/vocab.json
        encoder     (optional) – path to mel_encoder .mlpackage
        decoder     (optional) – CTC: ctc_decoder; TDT: RNNT prediction net .mlpackage

      CTC-specific:
        lm          (optional) – path to ARPA language model (enables beam search)
        lm_weight   (default 0.3)
        word_bonus  (default 1.0)
        beam_width  (default 100)

      TDT-specific:
        joint_decision_single_step (optional) – path to single-step joint .mlpackage
        max_symbols_per_step       (default 10)
    """

    # ── Initialisation ────────────────────────────────────────────────────────

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with open(self.config["metadata"]) as f:
            self.meta = json.load(f)

        self.SAMPLE_RATE: int = self.meta["sample_rate"]
        self.MAX_SAMPLES: int = self.meta["max_audio_samples"]

        # Directory that contains metadata.json; used for relative path resolution
        self._model_dir = Path(self.config["metadata"]).parent

        # Auto-detect model type from metadata structure
        self.model_type: str = self._detect_model_type()

        # Vocab: explicit path > <model_dir>/vocab.json
        vocab_path = self.config.get("vocab") or str(self._model_dir / "vocab.json")
        with open(vocab_path) as f:
            self.vocab: List[str] = json.load(f)

        if self.model_type == "tdt":
            self._init_tdt()
        else:
            self._init_ctc()

    def _detect_model_type(self) -> str:
        """Infer model type from config or metadata structure."""
        if explicit := self.config.get("model_type"):
            return explicit.lower().strip()
        components = self.meta.get("components", {})
        if "ctc_decoder" in components:
            return "ctc"
        if "joint" in components or "joint_extra_outputs" in self.meta:
            return "tdt"
        if "blank_id" in self.meta:
            return "ctc"
        return "ctc"

    def _resolve(self, config_key: str, component_name: str) -> str:
        """Resolve a component path.

        Priority:
          1. Explicit value in self.config
          2. <model_dir>/<components[component_name]["path"]> from metadata
          3. ValueError
        """
        if path := self.config.get(config_key):
            return path
        components = self.meta.get("components", {})
        if component_name in components:
            return str(self._model_dir / components[component_name]["path"])
        raise ValueError(
            f"Cannot resolve '{component_name}': set '{config_key}' in config "
            f"or ensure metadata.json lists it under 'components'."
        )

    def _init_ctc(self) -> None:
        # blank_id: CTC metadata stores it explicitly; TDT-flavoured hybrid uses vocab_size
        self.BLANK_ID: int = self.meta.get("blank_id", self.meta.get("vocab_size", 1024))

        self.mel_encoder = ct.models.MLModel(self._resolve("encoder", "mel_encoder"))
        self.ctc_decoder = ct.models.MLModel(self._resolve("decoder", "ctc_decoder"))

        self.lm: Optional[ARPALanguageModel] = None
        if lm_path := self.config.get("lm"):
            self.lm = ARPALanguageModel.load(lm_path)
        self.lm_weight: float = float(self.config.get("lm_weight", 0.3))
        self.word_bonus: float = float(self.config.get("word_bonus", 1.0))
        self.beam_width: int = int(self.config.get("beam_width", 100))

    def _init_tdt(self) -> None:
        # Blank is always vocab_size for pure RNNT
        self.BLANK_ID: int = self.meta["vocab_size"]

        self.mel_encoder = ct.models.MLModel(self._resolve("encoder", "mel_encoder"))
        self.decoder = ct.models.MLModel(self._resolve("decoder", "decoder"))
        self.joint_step = ct.models.MLModel(
            self._resolve("joint_decision_single_step", "joint_decision_single_step")
        )

        # LSTM state shapes from metadata — no hardcoding
        dec_inputs = self.meta["components"]["decoder"]["inputs"]
        self._h_shape: Tuple[int, ...] = tuple(dec_inputs["h_in"])
        self._c_shape: Tuple[int, ...] = tuple(dec_inputs["c_in"])

        self.max_symbols_per_step: int = int(self.config.get("max_symbols_per_step", 10))

    # ── Audio preparation (shared) ────────────────────────────────────────────

    def _prepare_audio(self, audio: AudioData) -> Tuple[np.ndarray, np.ndarray]:
        audio_array = audio.get_np_float32(convert_rate=self.SAMPLE_RATE)
        original_len = len(audio_array)
        if len(audio_array) < self.MAX_SAMPLES:
            audio_array = np.pad(audio_array, (0, self.MAX_SAMPLES - len(audio_array)))
        else:
            audio_array = audio_array[: self.MAX_SAMPLES]
        audio_signal = audio_array[np.newaxis, :].astype(np.float32)        # [1, N]
        audio_length = np.array([min(original_len, self.MAX_SAMPLES)], dtype=np.int32)
        return audio_signal, audio_length

    # ── CTC decoding ──────────────────────────────────────────────────────────

    def _decode_ctc(self, encoder_out: np.ndarray) -> str:
        """CTC: encoder output → text (greedy or beam search)."""
        dec_out = self.ctc_decoder.predict({"encoder": encoder_out})
        log_probs: np.ndarray = dec_out["log_probs"]        # [1, T, V]

        if self.lm is not None:
            return ctc_beam_search(
                log_probs[0],
                self.vocab,
                self.lm,
                blank_id=self.BLANK_ID,
                beam_width=self.beam_width,
                lm_weight=self.lm_weight,
                word_bonus=self.word_bonus,
            )

        # Greedy
        token_ids = np.argmax(log_probs[0], axis=-1)
        decoded, prev = [], None
        for t in token_ids:
            if t != self.BLANK_ID and t != prev:
                decoded.append(int(t))
            prev = t
        return "".join(self.vocab[i] for i in decoded).replace("▁", " ").strip()

    # ── TDT decoding ──────────────────────────────────────────────────────────

    def _decode_tdt(self, encoder: np.ndarray, encoder_length: np.ndarray) -> str:
        """TDT greedy decoding loop.

        encoder:        [1, D_enc, T_enc]
        encoder_length: [1]  — number of valid frames
        """
        T = int(encoder_length.flat[0])

        h = np.zeros(self._h_shape, dtype=np.float32)
        c = np.zeros(self._c_shape, dtype=np.float32)
        prev_label = np.array([[self.BLANK_ID]], dtype=np.int32)
        target_length = np.array([1], dtype=np.int32)

        tokens: List[int] = []
        t = 0

        while t < T:
            # Prediction network step (once per outer frame iteration)
            dec_out = self.decoder.predict({
                "targets": prev_label,
                "target_length": target_length,
                "h_in": h,
                "c_in": c,
            })
            dec_feat: np.ndarray = dec_out["decoder"]   # [1, D_dec, 1]
            h = dec_out["h_out"]
            c = dec_out["c_out"]

            # Inner loop: may emit multiple symbols for the same encoder frame
            symbols_this_frame = 0
            advanced = False

            while not advanced:
                enc_step = encoder[:, :, t: t + 1]      # [1, D_enc, 1]
                jd = self.joint_step.predict({
                    "encoder_step": enc_step,
                    "decoder_step": dec_feat,
                })
                token_id: int = int(jd["token_id"].flat[0])
                duration: int = int(jd["duration"].flat[0])

                if token_id == self.BLANK_ID or symbols_this_frame >= self.max_symbols_per_step:
                    # Blank (or safety cap): advance frame, prev_label unchanged
                    t += max(1, duration)
                    advanced = True

                else:
                    # Non-blank: emit and update decoder seed
                    tokens.append(token_id)
                    prev_label = np.array([[token_id]], dtype=np.int32)
                    symbols_this_frame += 1

                    if duration > 0:
                        t += duration
                        advanced = True
                    else:
                        # duration == 0: re-run prediction net, stay on frame
                        dec_out = self.decoder.predict({
                            "targets": prev_label,
                            "target_length": target_length,
                            "h_in": h,
                            "c_in": c,
                        })
                        dec_feat = dec_out["decoder"]
                        h = dec_out["h_out"]
                        c = dec_out["c_out"]

        return (
            "".join(self.vocab[i] for i in tokens if i < len(self.vocab))
            .replace("▁", " ")
            .strip()
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def transcribe(self, audio: AudioData, lang: Optional[str] = None) -> List[Tuple[str, float]]:
        audio_signal, audio_length = self._prepare_audio(audio)

        enc_out = self.mel_encoder.predict({
            "audio_signal": audio_signal,
            "audio_length": audio_length,
        })

        if self.model_type == "tdt":
            text = self._decode_tdt(enc_out["encoder"], enc_out["encoder_length"])
        else:
            text = self._decode_ctc(enc_out["encoder"])

        return [(text, 1.0)]

    def execute(self, audio, language=None) -> str:
        transcripts = self.transcribe(audio, language)
        return transcripts[0][0] if transcripts else ""

    @classproperty
    def available_languages(cls) -> set:
        # CTC models are English-only.
        # TDT v3 (parakeet-tdt-0.6b-v3) supports 25 European languages with
        # automatic language detection — no language input is needed or accepted.
        # We return the full superset; the loaded model determines actual coverage.
        return {
            "en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "uk",
            "cs", "ro", "hu", "sv", "fi", "da", "sk", "bg", "hr", "sr",
            "sl", "lt", "lv", "et", "el", "mt",
        }


# Alias kept for backward compatibility with the separate entry point
ParakeetTDTSTT = CoremlSTT


if __name__ == "__main__":
    import sys

    wav_file = "/Users/tigregotico/PycharmProjects/ovos-stt-plugin-coreml/parakeet_export/yc_first_minute_16k_15s.wav"

    _TDT_DIR = "/Users/tigregotico/PycharmProjects/ovos-stt-plugin-coreml/parakeet_export/parakeet_coreml_quantized/int8_linear"

    configs = {
        # "CTC greedy": {
        #     "metadata": "parakeet_ctc_coreml/metadata.json",
        #     "vocab":    "parakeet_ctc_coreml/vocab.json",
        #     "encoder":  "parakeet_ctc_coreml/parakeet_ctc_mel_encoder.mlpackage",
        #     "decoder":  "parakeet_ctc_coreml/parakeet_ctc_decoder.mlpackage",
        # },
        # "CTC + LM": {
        #     "metadata": "parakeet_ctc_coreml/metadata.json",
        #     "vocab":    "parakeet_ctc_coreml/vocab.json",
        #     "encoder":  "parakeet_ctc_coreml/parakeet_ctc_mel_encoder.mlpackage",
        #     "decoder":  "parakeet_ctc_coreml/parakeet_ctc_decoder.mlpackage",
        #     "lm":         "/path/to/language_model.arpa",
        #     "lm_weight":  0.3,
        #     "word_bonus": 1.0,
        #     "beam_width": 100,
        # },
        "TDT": {
            "metadata": f"{_TDT_DIR}/metadata.json",
            "vocab":    f"{_TDT_DIR}/vocab.json",
            "encoder":  f"{_TDT_DIR}/parakeet_mel_encoder.mlpackage",
            "decoder":  f"{_TDT_DIR}/parakeet_decoder.mlpackage",
            "joint_decision_single_step": f"{_TDT_DIR}/parakeet_joint_decision_single_step.mlpackage",
        },
    }

    with AudioFile(wav_file) as f:
        audio = f.read()

    for label, cfg in configs.items():
        try:
            stt = CoremlSTT(config=cfg)
            print(f"{label} ({stt.model_type}): {stt.execute(audio)}")
        except Exception as exc:
            print(f"{label}: skipped — {exc}", file=sys.stderr)
