import json
from typing import List, Tuple, Optional

import coremltools as ct
import numpy as np
from ovos_plugin_manager.templates.stt import STT
from ovos_plugin_manager.utils.audio import AudioData, AudioFile
from ovos_utils import classproperty

from ovos_stt_plugin_coreml.lm import ARPALanguageModel, ctc_beam_search


class CoremlSTT(STT):
    def __init__(self, *args, **kwargs):
        """
        Initialize the CoremlSTT instance by loading model metadata, Core ML encoder/decoder, vocabulary, and optional language model; also configure decoding parameters.
        
        Expects self.config to contain:
        - "metadata": path to a JSON file with keys "sample_rate", "max_audio_samples", and "blank_id" (used to set SAMPLE_RATE, MAX_SAMPLES, BLANK_ID).
        - "encoder": path to a Core ML encoder model file.
        - "decoder": path to a Core ML decoder model file.
        - "vocab": path to a JSON vocabulary file (list or mapping used for token->text conversion).
        
        Optional config keys:
        - "lm": path to an ARPA language model file; when present, an ARPALanguageModel is loaded and beam-search decoding is enabled.
        - "lm_weight": numeric weight applied to the language model during beam search (default 0.3).
        - "word_bonus": numeric bonus applied per word during beam search (default 1.0).
        - "beam_width": integer beam width for beam search (default 100).
        
        Sets the following instance attributes:
        - SAMPLE_RATE, MAX_SAMPLES, BLANK_ID from metadata.
        - mel_encoder (Core ML encoder model) and ctc_decoder (Core ML decoder model).
        - vocab (loaded vocabulary).
        - lm (ARPALanguageModel or None), lm_weight (float), word_bonus (float), beam_width (int).
        """
        super().__init__(*args, **kwargs)
        # Load metadata
        with open(self.config["metadata"]) as f:
            self.meta = json.load(f)

        self.SAMPLE_RATE = self.meta["sample_rate"]
        self.MAX_SAMPLES = self.meta["max_audio_samples"]
        self.BLANK_ID = self.meta["blank_id"]

        # Load models
        self.mel_encoder = ct.models.MLModel(self.config["encoder"])
        self.ctc_decoder = ct.models.MLModel(self.config["decoder"])
        with open(self.config["vocab"]) as f:
            self.vocab = json.load(f)

        # Optional LM — enables beam search when present
        self.lm: Optional[ARPALanguageModel] = None
        if lm_path := self.config.get("lm"):
            self.lm = ARPALanguageModel.load(lm_path)

        self.lm_weight: float = float(self.config.get("lm_weight", 0.3))
        self.word_bonus: float = float(self.config.get("word_bonus", 1.0))
        self.beam_width: int = int(self.config.get("beam_width", 100))

    def transcribe(self, audio: AudioData, lang: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Produce a transcription for the given audio using an optional language model; when a language model is configured, beam-search decoding is applied, otherwise greedy CTC decoding is used.
        
        Returns:
            List[Tuple[str, float]]: A list containing a single tuple with the transcript string and its confidence score (0.0–1.0).
        """

        self.lm_weight: float = float(self.config.get("lm_weight", 0.3))
        self.word_bonus: float = float(self.config.get("word_bonus", 1.0))
        self.beam_width: int = int(self.config.get("beam_width", 100))

    def transcribe(self, audio: AudioData, lang: Optional[str] = None) -> List[Tuple[str, float]]:
        """Transcribe audio. Uses beam search + LM when configured, greedy otherwise."""
        audio_array = audio.get_np_float32(convert_rate=self.SAMPLE_RATE)
        # pad/trim audio
        original_len = len(audio_array)
        if len(audio_array) < self.MAX_SAMPLES:
            audio_array = np.pad(audio_array, (0, self.MAX_SAMPLES - len(audio_array)))
        else:
            audio_array = audio_array[: self.MAX_SAMPLES]

        audio_signal = audio_array[np.newaxis, :].astype(np.float32)  # [1, N]
        audio_length = np.array([min(original_len, self.MAX_SAMPLES)], dtype=np.int32)

        enc_out = self.mel_encoder.predict({
            "audio_signal": audio_signal,
            "audio_length": audio_length,
        })
        dec_out = self.ctc_decoder.predict({"encoder": enc_out["encoder"]})
        log_probs = dec_out["log_probs"]  # [1, T, V]

        if self.lm is not None:
            text = ctc_beam_search(
                log_probs[0],
                self.vocab,
                self.lm,
                blank_id=self.BLANK_ID,
                beam_width=self.beam_width,
                lm_weight=self.lm_weight,
                word_bonus=self.word_bonus,
            )
        else:
            token_ids = np.argmax(log_probs[0], axis=-1)
            decoded = []
            prev = None
            for t in token_ids:
                if t != self.BLANK_ID and t != prev:
                    decoded.append(int(t))
                prev = t
            text = "".join(self.vocab[i] for i in decoded if i < len(self.vocab)).replace("▁", " ").strip()

        return [(text, 1.0)]

    def execute(self, audio, language=None) -> str:
        transcripts = self.transcribe(audio, language)
        if not transcripts:
            return ""
        return transcripts[0][0]

    @classproperty
    def available_languages(cls) -> set:
        """Return languages supported by this TTS implementation in this state
        This property should be overridden by the derived class to advertise
        what languages that engine supports.
        Returns:
            set: supported languages
        """
        return set()


if __name__ == "__main__":
    config = {
        "metadata": "parakeet_ctc_coreml/metadata.json",
        "vocab": "parakeet_ctc_coreml/vocab.json",
        "encoder": "parakeet_ctc_coreml/parakeet_ctc_mel_encoder.mlpackage",
        "decoder": "parakeet_ctc_coreml/parakeet_ctc_decoder.mlpackage",
        "lm": "/tmp/en-mix.lm",
        "lm_weight": 0.3,
        "word_bonus": 1.0,
        "beam_width": 100,
    }
    stt = CoremlSTT(config=config)

    wav_file = "yc_first_minute_16k_15s.wav"
    with AudioFile(wav_file) as f:
        audio = f.read()

    greedy_config = {**config}
    del greedy_config["lm"]
    stt_greedy = CoremlSTT(config=greedy_config)

    print("Greedy:", stt_greedy.execute(audio))
    print("Beam:  ", stt.execute(audio))
