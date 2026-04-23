import json
import shutil
from pathlib import Path
from typing import Any, List, Optional, Tuple

import coremltools as ct
import numpy as np
from ovos_plugin_manager.templates.stt import STT
from ovos_plugin_manager.utils.audio import AudioData, AudioFile
from ovos_utils import classproperty

from ovos_stt_plugin_coreml.lm import ARPALanguageModel, ctc_beam_search

# ── Language → best published HF repo ────────────────────────────────────────
# Used for auto-selection when no repo_id / metadata is set in config.
# Order reflects quality preference where multiple models exist for a language.
_V3_MULTILINGUAL_INT8 = "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-int8"

# Community language-specific model constants
_EU_V2_INT8  = "OpenVoiceOS/stt-eu-conformer-transducer-large-v2-coreml-int8"   # Basque (best)
_ES_INT8     = "OpenVoiceOS/parakeet-rnnt-1.1b-es-coreml-int8"                  # Spanish (fine-tuned parakeet)
_CA_ES_INT8  = "OpenVoiceOS/stt-ca-es-conformer-transducer-large-coreml-int8"   # Catalan+Spanish
_LOS_INT8    = "OpenVoiceOS/stt-los-conformer-transducer-large-coreml-int8"      # Catalan+Spanish+Galician+Basque

_LANG_BEST_REPO: dict = {
    # Dedicated language-specific models (preferred when available)
    "en": "OpenVoiceOS/parakeet-tdt-ctc-110m-coreml-int8",  # smallest/fastest English
    "ja": "OpenVoiceOS/parakeet-tdt-ctc-0.6b-ja-coreml-int8",
    "vi": "OpenVoiceOS/parakeet-ctc-0.6b-vi-coreml-int8",
    "da": "OpenVoiceOS/parakeet-rnnt-110m-da-coreml-int8",
    "nl": "OpenVoiceOS/parakeet-tdt-0.6b-dutch-coreml-int8",
    "et": "OpenVoiceOS/parakeet-tdt-0.6b-estonian-coreml-int8",
    "pl": "OpenVoiceOS/parakeet-tdt-0.6b-polish-coreml-int8",
    "pt": "OpenVoiceOS/parakeet-tdt-0.6b-portuguese-coreml-int8",
    "sl": "OpenVoiceOS/parakeet-tdt-0.6b-slovenian-coreml-int8",
    "eu": _EU_V2_INT8,   # Basque — HiTZ stt_eu_conformer_transducer_large_v2
    "es": _ES_INT8,      # Spanish — parakeet-rnnt-1.1b fine-tuned on Spanish CV17
    "ca": _CA_ES_INT8,   # Catalan — projecte-aina bilingual ca-es
    "gl": _LOS_INT8,     # Galician — BSC-LT LoS (only dedicated option)
    # 15 European languages without a dedicated model — served by multilingual v3
    "bg": _V3_MULTILINGUAL_INT8,   # Bulgarian
    "cs": _V3_MULTILINGUAL_INT8,   # Czech
    "de": _V3_MULTILINGUAL_INT8,   # German
    "el": _V3_MULTILINGUAL_INT8,   # Greek
    "fi": _V3_MULTILINGUAL_INT8,   # Finnish
    "fr": _V3_MULTILINGUAL_INT8,   # French
    "hr": _V3_MULTILINGUAL_INT8,   # Croatian
    "hu": _V3_MULTILINGUAL_INT8,   # Hungarian
    "it": _V3_MULTILINGUAL_INT8,   # Italian
    "lt": _V3_MULTILINGUAL_INT8,   # Lithuanian
    "lv": _V3_MULTILINGUAL_INT8,   # Latvian
    "mt": _V3_MULTILINGUAL_INT8,   # Maltese
    "ro": _V3_MULTILINGUAL_INT8,   # Romanian
    "ru": _V3_MULTILINGUAL_INT8,   # Russian
    "sk": _V3_MULTILINGUAL_INT8,   # Slovak
    "sv": _V3_MULTILINGUAL_INT8,   # Swedish
    "uk": _V3_MULTILINGUAL_INT8,   # Ukrainian
}

# All published repos, grouped by language, ordered best → smallest.
# Useful for callers that want to enumerate or let users pick a variant.
_ALL_REPOS: dict = {
    "en": [
        "OpenVoiceOS/parakeet-tdt-1.1b-coreml",
        "OpenVoiceOS/parakeet-tdt-1.1b-coreml-int8",
        "OpenVoiceOS/parakeet-tdt-1.1b-coreml-4bit",
        "OpenVoiceOS/parakeet-tdt-1.1b-coreml-6bit",
        "OpenVoiceOS/parakeet-tdt-1.1b-coreml-fp16",
        "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml",
        "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-int8",
        "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-4bit",
        "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-6bit",
        "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-fp16",
        "OpenVoiceOS/parakeet-tdt-0.6b-v2-coreml",
        "OpenVoiceOS/parakeet-tdt-0.6b-v2-coreml-int8",
        "OpenVoiceOS/parakeet-tdt-0.6b-v2-coreml-4bit",
        "OpenVoiceOS/parakeet-tdt-0.6b-v2-coreml-6bit",
        "OpenVoiceOS/parakeet-tdt-0.6b-v2-coreml-fp16",
        "OpenVoiceOS/parakeet-ctc-1.1b-coreml",
        "OpenVoiceOS/parakeet-ctc-1.1b-coreml-int8",
        "OpenVoiceOS/parakeet-ctc-1.1b-coreml-4bit",
        "OpenVoiceOS/parakeet-ctc-1.1b-coreml-6bit",
        "OpenVoiceOS/parakeet-ctc-1.1b-coreml-fp16",
        "OpenVoiceOS/parakeet-rnnt-1.1b-coreml",
        "OpenVoiceOS/parakeet-rnnt-1.1b-coreml-int8",
        "OpenVoiceOS/parakeet-rnnt-1.1b-coreml-4bit",
        "OpenVoiceOS/parakeet-rnnt-1.1b-coreml-6bit",
        "OpenVoiceOS/parakeet-rnnt-1.1b-coreml-fp16",
        "OpenVoiceOS/parakeet-ctc-0.6b-coreml",
        "OpenVoiceOS/parakeet-ctc-0.6b-coreml-int8",
        "OpenVoiceOS/parakeet-ctc-0.6b-coreml-4bit",
        "OpenVoiceOS/parakeet-ctc-0.6b-coreml-6bit",
        "OpenVoiceOS/parakeet-rnnt-0.6b-coreml",
        "OpenVoiceOS/parakeet-rnnt-0.6b-coreml-int8",
        "OpenVoiceOS/parakeet-rnnt-0.6b-coreml-4bit",
        "OpenVoiceOS/parakeet-rnnt-0.6b-coreml-6bit",
        "OpenVoiceOS/parakeet-tdt-ctc-110m-coreml-fp16",
        "OpenVoiceOS/parakeet-tdt-ctc-110m-coreml-int8",
        "OpenVoiceOS/parakeet-tdt-ctc-110m-coreml-4bit",
        "OpenVoiceOS/parakeet-tdt-ctc-110m-coreml-6bit",
        "OpenVoiceOS/parakeet-rnnt-120m-eou-coreml",
        "OpenVoiceOS/parakeet-rnnt-120m-eou-coreml-fp16",
        "OpenVoiceOS/parakeet-rnnt-120m-eou-coreml-int8",
        "OpenVoiceOS/parakeet-rnnt-120m-eou-coreml-6bit",
        "OpenVoiceOS/parakeet-unified-en-0.6b-coreml",
        "OpenVoiceOS/parakeet-unified-en-0.6b-coreml-int8",
        "OpenVoiceOS/parakeet-unified-en-0.6b-coreml-4bit",
        "OpenVoiceOS/parakeet-unified-en-0.6b-coreml-6bit",
    ],
    "ja": [
        "OpenVoiceOS/parakeet-tdt-ctc-0.6b-ja-coreml",
        "OpenVoiceOS/parakeet-tdt-ctc-0.6b-ja-coreml-int8",
        "OpenVoiceOS/parakeet-tdt-ctc-0.6b-ja-coreml-4bit",
        "OpenVoiceOS/parakeet-tdt-ctc-0.6b-ja-coreml-6bit",
        "OpenVoiceOS/parakeet-tdt-ctc-0.6b-ja-coreml-fp16",
    ],
    "vi": [
        "OpenVoiceOS/parakeet-ctc-0.6b-vi-coreml",
        "OpenVoiceOS/parakeet-ctc-0.6b-vi-coreml-int8",
        "OpenVoiceOS/parakeet-ctc-0.6b-vi-coreml-4bit",
        "OpenVoiceOS/parakeet-ctc-0.6b-vi-coreml-6bit",
    ],
    "nl": [
        "OpenVoiceOS/parakeet-tdt-0.6b-dutch-coreml",
        "OpenVoiceOS/parakeet-tdt-0.6b-dutch-coreml-int8",
        "OpenVoiceOS/parakeet-tdt-0.6b-dutch-coreml-4bit",
        "OpenVoiceOS/parakeet-tdt-0.6b-dutch-coreml-6bit",
        "OpenVoiceOS/parakeet-tdt-0.6b-dutch-coreml-fp16",
    ],
    "et": [
        "OpenVoiceOS/parakeet-tdt-0.6b-estonian-coreml",
        "OpenVoiceOS/parakeet-tdt-0.6b-estonian-coreml-int8",
        "OpenVoiceOS/parakeet-tdt-0.6b-estonian-coreml-4bit",
        "OpenVoiceOS/parakeet-tdt-0.6b-estonian-coreml-6bit",
        "OpenVoiceOS/parakeet-tdt-0.6b-estonian-coreml-fp16",
    ],
    "pl": [
        "OpenVoiceOS/parakeet-tdt-0.6b-polish-coreml",
        "OpenVoiceOS/parakeet-tdt-0.6b-polish-coreml-int8",
        "OpenVoiceOS/parakeet-tdt-0.6b-polish-coreml-4bit",
        "OpenVoiceOS/parakeet-tdt-0.6b-polish-coreml-6bit",
        "OpenVoiceOS/parakeet-tdt-0.6b-polish-coreml-fp16",
    ],
    "pt": [
        "OpenVoiceOS/parakeet-tdt-0.6b-portuguese-coreml",
        "OpenVoiceOS/parakeet-tdt-0.6b-portuguese-coreml-int8",
        "OpenVoiceOS/parakeet-tdt-0.6b-portuguese-coreml-4bit",
        "OpenVoiceOS/parakeet-tdt-0.6b-portuguese-coreml-6bit",
        "OpenVoiceOS/parakeet-tdt-0.6b-portuguese-coreml-fp16",
    ],
    "sl": [
        "OpenVoiceOS/parakeet-tdt-0.6b-slovenian-coreml",
        "OpenVoiceOS/parakeet-tdt-0.6b-slovenian-coreml-int8",
        "OpenVoiceOS/parakeet-tdt-0.6b-slovenian-coreml-4bit",
        "OpenVoiceOS/parakeet-tdt-0.6b-slovenian-coreml-6bit",
        "OpenVoiceOS/parakeet-tdt-0.6b-slovenian-coreml-fp16",
    ],
    "da": [
        "OpenVoiceOS/parakeet-rnnt-110m-da-coreml",
        "OpenVoiceOS/parakeet-rnnt-110m-da-coreml-int8",
        "OpenVoiceOS/parakeet-rnnt-110m-da-coreml-4bit",
        "OpenVoiceOS/parakeet-rnnt-110m-da-coreml-6bit",
        "OpenVoiceOS/parakeet-rnnt-110m-da-coreml-fp16",
    ],
    # Languages with dedicated community models (preferred over multilingual v3)
    "eu": [
        "OpenVoiceOS/stt-eu-conformer-transducer-large-v2-coreml",
        "OpenVoiceOS/stt-eu-conformer-transducer-large-v2-coreml-int8",
        "OpenVoiceOS/stt-eu-conformer-transducer-large-v2-coreml-4bit",
        "OpenVoiceOS/stt-eu-conformer-transducer-large-v2-coreml-6bit",
        "OpenVoiceOS/stt-eu-conformer-transducer-large-coreml",
        "OpenVoiceOS/stt-eu-conformer-transducer-large-coreml-int8",
        "OpenVoiceOS/stt-eu-conformer-transducer-large-coreml-4bit",
        "OpenVoiceOS/stt-eu-conformer-transducer-large-coreml-6bit",
        "OpenVoiceOS/stt-eu-conformer-ctc-large-coreml",
        "OpenVoiceOS/stt-eu-conformer-ctc-large-coreml-int8",
        "OpenVoiceOS/stt-eu-conformer-ctc-large-coreml-4bit",
        "OpenVoiceOS/stt-eu-conformer-ctc-large-coreml-6bit",
        "OpenVoiceOS/stt-eseu-conformer-transducer-large-coreml",
        "OpenVoiceOS/stt-eseu-conformer-transducer-large-coreml-int8",
        "OpenVoiceOS/stt-eseu-conformer-transducer-large-coreml-4bit",
        "OpenVoiceOS/stt-eseu-conformer-transducer-large-coreml-6bit",
        "OpenVoiceOS/bbs-s2tc-conformer-transducer-large-coreml",
        "OpenVoiceOS/bbs-s2tc-conformer-transducer-large-coreml-int8",
        "OpenVoiceOS/bbs-s2tc-conformer-transducer-large-coreml-4bit",
        "OpenVoiceOS/bbs-s2tc-conformer-transducer-large-coreml-6bit",
        "OpenVoiceOS/stt-los-conformer-transducer-large-coreml",
        "OpenVoiceOS/stt-los-conformer-transducer-large-coreml-int8",
        "OpenVoiceOS/stt-los-conformer-transducer-large-coreml-4bit",
        "OpenVoiceOS/stt-los-conformer-transducer-large-coreml-6bit",
    ],
    "es": [
        "OpenVoiceOS/parakeet-rnnt-1.1b-es-coreml",
        "OpenVoiceOS/parakeet-rnnt-1.1b-es-coreml-int8",
        "OpenVoiceOS/parakeet-rnnt-1.1b-es-coreml-4bit",
        "OpenVoiceOS/parakeet-rnnt-1.1b-es-coreml-6bit",
        "OpenVoiceOS/stt-ca-es-conformer-transducer-large-coreml",
        "OpenVoiceOS/stt-ca-es-conformer-transducer-large-coreml-int8",
        "OpenVoiceOS/stt-ca-es-conformer-transducer-large-coreml-4bit",
        "OpenVoiceOS/stt-ca-es-conformer-transducer-large-coreml-6bit",
        "OpenVoiceOS/stt-los-conformer-transducer-large-punctuated-coreml",
        "OpenVoiceOS/stt-los-conformer-transducer-large-punctuated-coreml-int8",
        "OpenVoiceOS/stt-los-conformer-transducer-large-punctuated-coreml-4bit",
        "OpenVoiceOS/stt-los-conformer-transducer-large-punctuated-coreml-6bit",
        "OpenVoiceOS/stt-los-conformer-transducer-large-coreml",
        "OpenVoiceOS/stt-los-conformer-transducer-large-coreml-int8",
        "OpenVoiceOS/stt-los-conformer-transducer-large-coreml-4bit",
        "OpenVoiceOS/stt-los-conformer-transducer-large-coreml-6bit",
        "OpenVoiceOS/stt-eseu-conformer-transducer-large-coreml",
        "OpenVoiceOS/stt-eseu-conformer-transducer-large-coreml-int8",
        "OpenVoiceOS/stt-eseu-conformer-transducer-large-coreml-4bit",
        "OpenVoiceOS/stt-eseu-conformer-transducer-large-coreml-6bit",
        _V3_MULTILINGUAL_INT8,
        "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml",
        "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-4bit",
        "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-6bit",
        "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-fp16",
    ],
    "ca": [
        "OpenVoiceOS/stt-ca-es-conformer-transducer-large-coreml",
        "OpenVoiceOS/stt-ca-es-conformer-transducer-large-coreml-int8",
        "OpenVoiceOS/stt-ca-es-conformer-transducer-large-coreml-4bit",
        "OpenVoiceOS/stt-ca-es-conformer-transducer-large-coreml-6bit",
        "OpenVoiceOS/stt-los-conformer-transducer-large-coreml",
        "OpenVoiceOS/stt-los-conformer-transducer-large-coreml-int8",
        "OpenVoiceOS/stt-los-conformer-transducer-large-coreml-4bit",
        "OpenVoiceOS/stt-los-conformer-transducer-large-coreml-6bit",
    ],
    "gl": [
        "OpenVoiceOS/stt-los-conformer-transducer-large-coreml",
        "OpenVoiceOS/stt-los-conformer-transducer-large-coreml-int8",
        "OpenVoiceOS/stt-los-conformer-transducer-large-coreml-4bit",
        "OpenVoiceOS/stt-los-conformer-transducer-large-coreml-6bit",
    ],
    # 15 European languages without a dedicated model — served by multilingual v3
    "bg": [_V3_MULTILINGUAL_INT8, "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-4bit", "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-6bit",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-fp16"],
    "cs": [_V3_MULTILINGUAL_INT8, "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-4bit", "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-6bit",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-fp16"],
    "de": [_V3_MULTILINGUAL_INT8, "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-4bit", "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-6bit",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-fp16"],
    "el": [_V3_MULTILINGUAL_INT8, "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-4bit", "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-6bit",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-fp16"],
    "fi": [_V3_MULTILINGUAL_INT8, "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-4bit", "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-6bit",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-fp16"],
    "fr": [_V3_MULTILINGUAL_INT8, "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-4bit", "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-6bit",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-fp16"],
    "hr": [_V3_MULTILINGUAL_INT8, "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-4bit", "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-6bit",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-fp16"],
    "hu": [_V3_MULTILINGUAL_INT8, "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-4bit", "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-6bit",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-fp16"],
    "it": [_V3_MULTILINGUAL_INT8, "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-4bit", "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-6bit",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-fp16"],
    "lt": [_V3_MULTILINGUAL_INT8, "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-4bit", "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-6bit",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-fp16"],
    "lv": [_V3_MULTILINGUAL_INT8, "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-4bit", "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-6bit",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-fp16"],
    "mt": [_V3_MULTILINGUAL_INT8, "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-4bit", "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-6bit",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-fp16"],
    "ro": [_V3_MULTILINGUAL_INT8, "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-4bit", "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-6bit",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-fp16"],
    "ru": [_V3_MULTILINGUAL_INT8, "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-4bit", "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-6bit",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-fp16"],
    "sk": [_V3_MULTILINGUAL_INT8, "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-4bit", "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-6bit",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-fp16"],
    "sv": [_V3_MULTILINGUAL_INT8, "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-4bit", "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-6bit",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-fp16"],
    "uk": [_V3_MULTILINGUAL_INT8, "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-4bit", "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-6bit",
           "OpenVoiceOS/parakeet-tdt-0.6b-v3-coreml-fp16"],
}

# ── PyObjC CoreML backend (ANE dispatch) ─────────────────────────────────────
# When pyobjc-framework-CoreML is installed, models are loaded and run through
# the native CoreML Objective-C framework, which properly dispatches to ANE/GPU.
# Falls back to coremltools if PyObjC is not available.
try:
    import CoreML as _CoreML
    from Foundation import NSURL as _NSURL
    _PYOBJC = True
except ImportError:
    _PYOBJC = False


def _deref_mlpackage(pkg_path: str) -> str:
    """Return a real-file (no-symlink) copy of the mlpackage.

    The HF hub cache stores weights as symlinked blobs; the CoreML compiler
    subprocess cannot follow cross-device symlinks. Creates a persistent
    sibling directory (<name>.deref.mlpackage) and reuses it across runs.
    """
    pkg = Path(pkg_path)
    if not any(p.is_symlink() for p in pkg.rglob("*")):
        return pkg_path
    deref = pkg.with_suffix(".deref.mlpackage")
    pkg_mtime = pkg.stat().st_mtime
    if not deref.exists() or deref.stat().st_mtime < pkg_mtime:
        if deref.exists():
            shutil.rmtree(deref)
        shutil.copytree(str(pkg), str(deref), symlinks=False)
    return str(deref)


def _compile_mlpackage(mlpackage_path: str) -> str:
    """Compile a .mlpackage to .mlmodelc (cached alongside the package).

    Uses coremltools' compile_model which invokes the system CoreML compiler.
    The .mlmodelc is stored next to the .mlpackage so it persists across runs.
    """
    pkg = Path(mlpackage_path)
    out = pkg.parent / (pkg.stem + ".mlmodelc")
    pkg_mtime = pkg.stat().st_mtime
    if not out.exists() or out.stat().st_mtime < pkg_mtime:
        if out.exists():
            shutil.rmtree(out)
        real_pkg = _deref_mlpackage(mlpackage_path)
        tmp = ct.utils.compile_model(real_pkg)
        shutil.move(tmp, str(out))
    return str(out)


def _load_model(mlpackage_path: str, compute_units: ct.ComputeUnit) -> Any:
    """Load a CoreML model.

    With PyObjC available: compiles to .mlmodelc and loads through the native
    CoreML ObjC framework for ANE/GPU dispatch.
    Without PyObjC: loads .mlpackage directly via coremltools.
    """
    if _PYOBJC:
        mlmodelc = _compile_mlpackage(mlpackage_path)
        _cu_map = {
            ct.ComputeUnit.ALL: _CoreML.MLComputeUnitsAll,
            ct.ComputeUnit.CPU_ONLY: _CoreML.MLComputeUnitsCPUOnly,
            ct.ComputeUnit.CPU_AND_GPU: _CoreML.MLComputeUnitsCPUAndGPU,
            ct.ComputeUnit.CPU_AND_NE: _CoreML.MLComputeUnitsCPUAndNeuralEngine,
        }
        config = _CoreML.MLModelConfiguration.alloc().init()
        config.setComputeUnits_(_cu_map[compute_units])
        model, err = _CoreML.MLModel.modelWithContentsOfURL_configuration_error_(
            _NSURL.fileURLWithPath_(mlmodelc), config, None
        )
        if err or model is None:
            raise RuntimeError(f"CoreML load failed for {mlmodelc}: {err}")
        return _ObjCModel(model)
    # No PyObjC — resolve any HF-cache symlinks before handing off to
    # coremltools (whose compiler subprocess cannot follow cross-device links).
    real_pkg = _deref_mlpackage(mlpackage_path)
    return ct.models.MLModel(real_pkg, compute_units=compute_units)


class _ObjCModel:
    """Thin wrapper around an ObjC MLModel that exposes a coremltools-compatible
    predict(dict) → dict interface, bridging numpy ↔ MLMultiArray."""

    def __init__(self, objc_model: Any) -> None:
        """Store the native ObjC MLModel instance."""
        self._m = objc_model

    def predict(self, inputs: dict) -> dict:
        """Run inference via the native CoreML ObjC framework.

        Converts numpy arrays to MLMultiArray inputs, calls the model, then
        converts MLMultiArray outputs back to numpy arrays.
        """
        import ctypes

        def _to_mlarray(arr: np.ndarray) -> Any:
            arr = np.ascontiguousarray(arr)
            _dtype_map = {
                np.dtype("float32"): _CoreML.MLMultiArrayDataTypeFloat32,
                np.dtype("float16"): _CoreML.MLMultiArrayDataTypeFloat16,
                np.dtype("int32"):   _CoreML.MLMultiArrayDataTypeInt32,
            }
            ml_dtype = _dtype_map[arr.dtype]
            strides = [s // arr.itemsize for s in arr.strides]
            c_ptr = arr.ctypes.data_as(ctypes.c_void_p)
            ml_arr, err = _CoreML.MLMultiArray.alloc(
            ).initWithDataPointer_shape_dataType_strides_deallocator_error_(
                c_ptr, list(arr.shape), ml_dtype, strides, None, None
            )
            if err:
                raise RuntimeError(f"MLMultiArray init failed: {err}")
            return ml_arr, arr  # return arr to keep it alive

        def _from_mlarray(ml_arr: Any) -> np.ndarray:
            shape = tuple(int(d) for d in ml_arr.shape())
            total = 1
            for d in shape:
                total *= d
            _dtype_info = {
                _CoreML.MLMultiArrayDataTypeFloat32: (np.float32, 4),
                _CoreML.MLMultiArrayDataTypeFloat16: (np.float16, 2),
                _CoreML.MLMultiArrayDataTypeInt32:   (np.int32,   4),
            }
            np_dtype, itemsize = _dtype_info.get(ml_arr.dataType(), (np.float32, 4))
            # Validate C-contiguous strides before raw pointer access
            strides = tuple(int(s) for s in ml_arr.strides())
            expected_stride = 1
            expected: List[int] = []
            for dim in reversed(shape):
                expected.insert(0, expected_stride)
                expected_stride *= dim
            is_contiguous = strides == tuple(expected)
            # Fast path: read raw bytes via data pointer (zero-copy where supported)
            if is_contiguous:
                try:
                    ptr = ml_arr.dataPointer()
                    addr = ptr if isinstance(ptr, int) else int(ptr)
                    buf = ctypes.string_at(addr, total * itemsize)
                    return np.frombuffer(buf, dtype=np_dtype).reshape(shape).copy()
                except Exception:
                    pass
            # Fallback: element-wise (safe on all PyObjC versions, O(n) Python)
            flat = np.empty(total, dtype=np_dtype)
            for i in range(total):
                flat[i] = ml_arr[i]
            return flat.reshape(shape)

        # Build input feature provider
        feat_dict = {}
        refs = []  # keep numpy arrays alive while ObjC holds pointers
        for name, arr in inputs.items():
            ml_arr, ref = _to_mlarray(arr)
            feat_dict[name] = _CoreML.MLFeatureValue.featureValueWithMultiArray_(ml_arr)
            refs.append(ref)

        provider, err = _CoreML.MLDictionaryFeatureProvider.alloc(
        ).initWithDictionary_error_(feat_dict, None)
        if err:
            raise RuntimeError(f"MLDictionaryFeatureProvider failed: {err}")

        result, err = self._m.predictionFromFeatures_error_(provider, None)
        if err:
            raise RuntimeError(f"CoreML prediction failed: {err}")

        # Extract outputs
        outputs = {}
        for name in result.featureNames():
            fv = result.featureValueForName_(name)
            if fv is not None:
                ml_arr = fv.multiArrayValue()
                if ml_arr is not None:
                    outputs[str(name)] = _from_mlarray(ml_arr)
        return outputs


class CoremlSTT(STT):
    """Generic OVOS STT plugin for CoreML-exported speech recognition models.

    Supports three model families detected automatically from metadata.json:

      • ctc  – Pure CTC (EncDecCTCModelBPE)
               Pipeline: mel_encoder → ctc_decoder → log_probs → greedy / beam search
               model_type in metadata: "ctc"

      • tdt  – TDT (Token-and-Duration Transducer, EncDecRNNTBPEModel with num_extra > 0)
               Pipeline: mel_encoder (once) → per-frame decoder + joint decision step
               Duration output drives frame advancement.
               Also covers hybrid TDT-CTC models (parakeet-tdt-ctc-*): TDT path is
               preferred because it gives significantly lower WER than the CTC head.
               model_type in metadata: "parakeet_tdt_rnnt"

      • rnnt – Pure RNNT (EncDecRNNTBPEModel with num_extra == 0)
               Same pipeline as TDT but duration is always 0; decoder re-runs until blank.
               model_type in metadata: "parakeet_rnnt"

    Detection logic (checked in order):
      1. config["model_type"] = "ctc" or "tdt" — explicit override
      2. Both "joint_decision_single_step" AND "ctc_decoder" in components → tdt
         (hybrid TDT-CTC models; TDT path preferred for transcription accuracy)
      3. "joint_decision_single_step" in components only → tdt
      4. "ctc_decoder" in components only → ctc
      5. "blank_id" in metadata top-level → ctc (legacy fallback)
      6. Default: ctc

    Note: use config["model_type"] = "ctc" to force the CTC head on a hybrid model
    (useful for keyword-spotting / constrained decoding applications).

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

    Auto-download from HuggingFace Hub — set repo_id instead of metadata:

        {"repo_id": "OpenVoiceOS/parakeet-tdt-0.6b-v2-coreml"}

    Zero-config — omit both repo_id and metadata to auto-select the best
    published int8 model for the configured language:

        {}   # → parakeet-tdt-ctc-110m-coreml-int8  for lang=en-*
             # → parakeet-tdt-0.6b-dutch-coreml-int8 for lang=nl-*
             # → parakeet-tdt-0.6b-v3-coreml-int8   for lang=de-*, fr-*, … (16 EU langs via multilingual v3)
             # → … (see _LANG_BEST_REPO for the full mapping)

    The model is cached in ~/.cache/huggingface/hub (shared with transformers).
    Requires: pip install huggingface-hub

    Full config reference:

      Common:
        repo_id     (optional) – HF repo id; auto-downloads when metadata is absent
        metadata    (required unless repo_id set) – path to metadata.json
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
        """Initialise the plugin: download model if needed, load metadata, vocab, and CoreML components."""
        super().__init__(*args, **kwargs)

        self._maybe_download_from_hub()

        with open(self.config["metadata"]) as f:
            self.meta = json.load(f)

        self.SAMPLE_RATE: int = self.meta["sample_rate"]
        self.MAX_SAMPLES: int = self.meta["max_audio_samples"]

        # Directory that contains metadata.json; used for relative path resolution
        self._model_dir = Path(self.config["metadata"]).parent

        # Auto-detect model type from metadata structure
        self.model_type: str = self._detect_model_type()

        # Compute units: "all" (default, ANE/GPU), "cpu_only", "cpu_and_gpu", "cpu_and_ne"
        _cu_map = {
            "all":         ct.ComputeUnit.ALL,
            "cpu_only":    ct.ComputeUnit.CPU_ONLY,
            "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
            "cpu_and_ne":  ct.ComputeUnit.CPU_AND_NE,
        }
        cu_cfg = str(self.config.get("compute_units", "all")).lower().replace("-", "_")
        if cu_cfg not in _cu_map:
            raise ValueError(
                f"Unknown compute_units={cu_cfg!r}. "
                f"Valid values: {list(_cu_map)}."
            )
        self._default_cu: ct.ComputeUnit = _cu_map[cu_cfg]

        # Vocab: explicit path > <model_dir>/vocab.json
        vocab_path = self.config.get("vocab") or str(self._model_dir / "vocab.json")
        with open(vocab_path) as f:
            self.vocab: List[str] = json.load(f)

        if self.model_type == "tdt":
            self._init_tdt()
        else:
            self._init_ctc()

    def _maybe_download_from_hub(self) -> None:
        """Download model snapshot from HuggingFace Hub.

        Triggered when ``repo_id`` is set, or when neither ``repo_id`` nor
        ``metadata`` is configured (auto-selects the best published model for
        ``self.lang``).

        Uses the standard HF Hub cache (``~/.cache/huggingface/hub`` by default,
        respects ``HF_HOME`` / ``HUGGINGFACE_HUB_CACHE`` env vars), so the
        download is shared with transformers and other HF tooling.

        Raises ``ImportError`` if ``huggingface_hub`` is not installed.
        """
        if self.config.get("metadata"):
            return  # explicit local path — nothing to do

        repo_id = self.config.get("repo_id")
        if not repo_id:
            # Auto-select: pick the best repo for the configured language
            lang_tag = (self.lang or "en-us").lower()
            base_lang = lang_tag.split("-")[0].split("_")[0]
            if base_lang not in _LANG_BEST_REPO:
                import warnings
                warnings.warn(
                    f"Language {base_lang!r} is not supported; falling back to English model. "
                    f"Supported languages: {sorted(_LANG_BEST_REPO)}",
                    stacklevel=3,
                )
            repo_id = _LANG_BEST_REPO.get(base_lang, _LANG_BEST_REPO["en"])
            self.config["repo_id"] = repo_id

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for HF auto-download. "
                "Install it with: pip install huggingface-hub"
            ) from exc
        local_dir = snapshot_download(repo_id=repo_id, repo_type="model")
        self.config["metadata"] = str(Path(local_dir) / "metadata.json")

    def _detect_model_type(self) -> str:
        """Infer model type from config or metadata structure."""
        if explicit := self.config.get("model_type"):
            return explicit.lower().strip()
        components = self.meta.get("components", {})
        has_tdt = "joint_decision_single_step" in components
        has_ctc = "ctc_decoder" in components
        # TDT (or hybrid TDT-CTC): prefer TDT path — it gives lower WER than CTC head
        if has_tdt:
            return "tdt"
        # Pure CTC
        if has_ctc:
            return "ctc"
        # Legacy CTC metadata (older format, no components section)
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
        """Load CTC encoder and decoder models plus optional ARPA language model."""
        # blank_id: CTC metadata stores it explicitly; TDT-flavoured hybrid uses vocab_size
        self.BLANK_ID: int = self.meta.get("blank_id", self.meta.get("vocab_size", 1024))

        self.mel_encoder = _load_model(self._resolve("encoder", "mel_encoder"),
                                       self._default_cu)
        self.ctc_decoder = _load_model(self._resolve("decoder", "ctc_decoder"),
                                       self._default_cu)

        self.lm: Optional[ARPALanguageModel] = None
        if lm_path := self.config.get("lm"):
            self.lm = ARPALanguageModel.load(lm_path)
        self.lm_weight: float = float(self.config.get("lm_weight", 0.3))
        self.word_bonus: float = float(self.config.get("word_bonus", 1.0))
        self.beam_width: int = int(self.config.get("beam_width", 100))

    def _init_tdt(self) -> None:
        """Load TDT/RNNT encoder, prediction-net decoder, and joint decision models."""
        # blank_id is explicit in metadata for all converted models; fall back to vocab_size
        self.BLANK_ID: int = self.meta.get("blank_id", self.meta.get("vocab_size", 1024))

        self.mel_encoder = _load_model(self._resolve("encoder", "mel_encoder"),
                                       self._default_cu)
        # LSTM decoder: always CPU (autoregressive, not parallelisable on ANE/GPU)
        self.decoder = _load_model(self._resolve("decoder", "decoder"),
                                   ct.ComputeUnit.CPU_ONLY)
        self.joint_step = _load_model(
            self._resolve("joint_decision_single_step", "joint_decision_single_step"),
            self._default_cu,
        )

        # LSTM state shapes from metadata — no hardcoding
        dec_inputs = self.meta["components"]["decoder"]["inputs"]
        self._h_shape: Tuple[int, ...] = tuple(dec_inputs["h_in"])
        self._c_shape: Tuple[int, ...] = tuple(dec_inputs["c_in"])

        # duration_bins maps argmax index → frame count (standard TDT: [0,1,2,3,4])
        # None for pure RNNT (duration is always 0, index == value trivially)
        self._duration_bins: Optional[List[int]] = self.meta.get("duration_bins")

        self.max_symbols_per_step: int = int(self.config.get("max_symbols_per_step", 10))

    # ── Audio preparation (shared) ────────────────────────────────────────────

    def _prepare_audio(self, audio: AudioData) -> Tuple[np.ndarray, np.ndarray]:
        """Convert AudioData to a fixed-length float32 signal and an exact-length array."""
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

    def _decode_ctc(self, encoder_out: np.ndarray, encoder_length: np.ndarray) -> str:
        """CTC: encoder output → text (greedy or beam search).

        Slices encoder_out to the actual (unpadded) length before decoding to
        avoid spurious non-blank tokens from the padded frames.
        """
        T = int(encoder_length.flat[0])
        dec_out = self.ctc_decoder.predict({"encoder": encoder_out[:, :, :T]})
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

    # ── TDT / RNNT decoding ───────────────────────────────────────────────────

    def _duration_frames(self, dur_idx: int) -> int:
        """Convert joint duration argmax index → frame count using duration_bins.

        For standard TDT bins [0,1,2,3,4] the index equals the frame count.
        For pure RNNT, duration is always 0 and bins are absent.
        """
        if self._duration_bins:
            return self._duration_bins[min(dur_idx, len(self._duration_bins) - 1)]
        return dur_idx

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
                dur_idx: int = int(jd["duration"].flat[0])
                duration: int = self._duration_frames(dur_idx)

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
        """Transcribe audio and return a list of (transcript, confidence) tuples."""
        audio_signal, audio_length = self._prepare_audio(audio)

        enc_out = self.mel_encoder.predict({
            "audio_signal": audio_signal,
            "audio_length": audio_length,
        })

        if self.model_type == "tdt":
            text = self._decode_tdt(enc_out["encoder"], enc_out["encoder_length"])
        else:
            text = self._decode_ctc(enc_out["encoder"], enc_out["encoder_length"])

        return [(text, 1.0)]

    def execute(self, audio, language=None) -> str:
        """Return the top transcript string for the given audio."""
        transcripts = self.transcribe(audio, language)
        return transcripts[0][0] if transcripts else ""

    @classproperty
    def available_languages(cls) -> set:
        """Return the set of BCP-47 base language codes supported by published models."""
        return set(_LANG_BEST_REPO.keys())


# Alias kept for backward compatibility with the separate entry point
ParakeetTDTSTT = CoremlSTT


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <metadata.json> <audio.wav> [lm.arpa]")
        sys.exit(1)

    _metadata = sys.argv[1]
    _audio_path = sys.argv[2]
    _lm_path = sys.argv[3] if len(sys.argv) > 3 else None

    _config: dict = {"metadata": _metadata}
    if _lm_path:
        _config.update({"lm": _lm_path, "lm_weight": 0.3, "word_bonus": 1.0, "beam_width": 100})

    with AudioFile(_audio_path) as _f:
        _audio = _f.read()

    if _lm_path:
        _stt_greedy = CoremlSTT(config={k: v for k, v in _config.items() if k != "lm"})
        _stt_lm = CoremlSTT(config=_config)
        print("Greedy:", _stt_greedy.execute(_audio))
        print("Beam:  ", _stt_lm.execute(_audio))
    else:
        print(CoremlSTT(config=_config).execute(_audio))
