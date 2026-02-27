import json
from typing import List, Tuple, Optional

import coremltools as ct
import numpy as np
from ovos_plugin_manager.templates.stt import STT
from ovos_plugin_manager.utils.audio import AudioData, AudioFile
from ovos_utils import classproperty


class CoremlSTT(STT):
    def __init__(self, *args, **kwargs):
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

    def transcribe(self, audio: AudioData, lang: Optional[str] = None) -> List[Tuple[str, float]]:
        """transcribe audio data to a list of
        possible transcriptions and respective confidences"""

        audio_array = audio.get_np_float32(convert_rate=self.SAMPLE_RATE)
        # pad/trim audio
        original_len = len(audio_array)
        if len(audio_array) < self.MAX_SAMPLES:
            audio_array = np.pad(audio_array, (0, self.MAX_SAMPLES - len(audio_array)))
        else:
            audio_array = audio_array[: self.MAX_SAMPLES]

        audio_signal = audio_array[np.newaxis, :].astype(np.float32)  # [1, N]
        audio_length = np.array([min(original_len, self.MAX_SAMPLES)], dtype=np.int32)  # [1]

        # Stage 1: Mel + Encoder
        enc_out = self.mel_encoder.predict({
            "audio_signal": audio_signal,
            "audio_length": audio_length,
        })
        encoder = enc_out["encoder"]
        encoder_length = enc_out["encoder_length"]
        print(f"Encoder output shape: {encoder.shape}")  # [1, hidden, T]

        # Stage 2: CTC Decoder
        dec_out = self.ctc_decoder.predict({"encoder": encoder})
        log_probs = dec_out["log_probs"]  # [1, T, vocab+1]
        print(f"Log probs shape: {log_probs.shape}")

        # Greedy decode
        token_ids = np.argmax(log_probs[0], axis=-1)  # [T]

        # CTC collapse (remove blanks and repeated tokens)
        decoded = []
        prev = None
        for t in token_ids:
            if t != self.BLANK_ID and t != prev:
                decoded.append(int(t))
            prev = t

        # Load vocab and decode
        text = "".join(self.vocab[i] for i in decoded).replace("", " ").strip()

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
    }
    stt = CoremlSTT(config=config)

    wav_file =  "yc_first_minute_16k_15s.wav"
    with AudioFile(wav_file) as f:
        audio = f.read()
    transcript = stt.execute(audio)
    print(transcript)