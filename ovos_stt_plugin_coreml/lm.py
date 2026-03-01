import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── ARPA language model ────────────────────────────────────────────────────────

_NEG_INF = float("-inf")
_LOG10_TO_NAT = 2.302585
_UNK_LOG_PROB = -23.026  # fallback for OOV words (~log(10^-10))


def _logaddexp(a: float, b: float) -> float:
    """
    Compute the numerically stable logarithm of the sum of two values given in natural log space.
    
    Parameters:
        a (float): First value in natural log space; may be `_NEG_INF` to represent negative infinity.
        b (float): Second value in natural log space; may be `_NEG_INF` to represent negative infinity.
    
    Returns:
        float: The value of `log(exp(a) + exp(b))`. If one operand equals `_NEG_INF`, returns the other operand.
    """
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


class ARPALanguageModel:
    """Bigram ARPA language model with Katz back-off."""

    def __init__(self):
        # {word: (log_prob_nat, backoff_nat)}
        """
        Initialize an empty ARPA language model storage.
        
        Creates containers for unigram and bigram entries:
        - `unigrams`: mapping from word to a tuple (log_probability_in_nats, backoff_in_nats).
        - `bigrams`: mapping from context word to a dictionary mapping next-word to (log_probability_in_nats, backoff_in_nats).
        """
        self.unigrams: Dict[str, Tuple[float, float]] = {}
        # {context: {word: (log_prob_nat, backoff_nat)}}
        self.bigrams: Dict[str, Dict[str, Tuple[float, float]]] = {}

    @classmethod
    def load(cls, path: str) -> "ARPALanguageModel":
        """
        Load an ARPA-format language model file and return a populated ARPALanguageModel instance.
        
        Parses unigram and bigram sections from the ARPA file at `path`, converting log10 probabilities to natural-log space and storing per-token probabilities and backoff weights. Lines that cannot be parsed as valid probabilities are skipped; sections are detected by lines beginning with "\" and parsing stops at "\end\".
        
        Parameters:
            path (str): Filesystem path to the ARPA-format file to load.
        
        Returns:
            ARPALanguageModel: An instance with `unigrams` and `bigrams` populated with tuples of (log_prob_nat, backoff_nat).
        """
        lm = cls()
        section = ""
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n").strip()
                if not line or line.startswith("\\data\\"):
                    continue
                if line.startswith("\\"):
                    section = line
                    continue
                if line == "\\end\\":
                    break
                parts = line.split("\t")
                try:
                    prob = float(parts[0]) * _LOG10_TO_NAT
                except ValueError:
                    continue
                if section == "\\1-grams:" and len(parts) >= 2:
                    word = parts[1]
                    backoff = float(parts[2]) * _LOG10_TO_NAT if len(parts) >= 3 else 0.0
                    lm.unigrams[word] = (prob, backoff)
                elif section == "\\2-grams:" and len(parts) >= 3:
                    ctx, word = parts[1], parts[2]
                    backoff = float(parts[3]) * _LOG10_TO_NAT if len(parts) >= 4 else 0.0
                    lm.bigrams.setdefault(ctx, {})[word] = (prob, backoff)
        return lm

    def score(self, word: str, prev: Optional[str]) -> float:
        """
        Compute the log-probability (in nats) of a target word given an optional previous word.
        
        Parameters:
            word (str): Target word whose probability is requested.
            prev (Optional[str]): Previous completed word used as bigram context, or `None` to use unigram probability.
        
        Returns:
            float: Log probability in nats for `word` given `prev`. If a bigram entry for (prev, word) exists that value is returned; otherwise the previous word's backoff (if any) is added to the unigram probability for `word`, with a fallback unknown-word log-probability for unseen words.
        """
        if prev is not None and prev in self.bigrams and word in self.bigrams[prev]:
            return self.bigrams[prev][word][0]
        backoff = self.unigrams[prev][1] if prev is not None and prev in self.unigrams else 0.0
        return backoff + (self.unigrams[word][0] if word in self.unigrams else _UNK_LOG_PROB)


# ── CTC beam search ────────────────────────────────────────────────────────────

@dataclass
class _Beam:
    p_blank: float = _NEG_INF
    p_nonblank: float = _NEG_INF
    lm_score: float = 0.0
    word_pieces: List[str] = field(default_factory=list)
    prev_word: Optional[str] = None

    @property
    def acoustic(self) -> float:
        """
        Combined acoustic log-probability of the beam.
        
        Returns:
            float: log-sum-exp of `p_blank` and `p_nonblank`, i.e. the total acoustic log-probability for this beam.
        """
        return _logaddexp(self.p_blank, self.p_nonblank)

    @property
    def total(self) -> float:
        """
        Total score combining the acoustic probability and accumulated language-model score.
        
        Returns:
            total (float): Sum of the beam's acoustic score and its LM score.
        """
        return self.acoustic + self.lm_score


def ctc_beam_search(
        log_probs: np.ndarray,
        vocab: List[str],
        lm: ARPALanguageModel,
        blank_id: int,
        beam_width: int,
        lm_weight: float,
        word_bonus: float,
        token_candidates: int = 40,
) -> str:
    """
        Perform CTC beam search decoding with ARPA bigram language-model rescoring.
        
        Parameters:
            log_probs (np.ndarray): [T, V] array of per-timestep log-probabilities.
            vocab (List[str]): Mapping from token index to token string (word-piece tokens, use "▁" as word boundary).
            lm (ARPALanguageModel): Bigram ARPA language model used to rescore completed words.
            blank_id (int): Index of the CTC blank token in `vocab` / probability columns.
            beam_width (int): Number of candidate beams to keep after each timestep.
            lm_weight (float): Multiplier applied to LM log-probabilities when rescoring completed words.
            word_bonus (float): Additive score applied each time a word finishes (at word-boundary tokens).
            token_candidates (int): Maximum number of non-blank token indices considered per frame (default 40).
        
        Returns:
            str: Decoded hypothesis as a human-readable string (word-piece tokens joined, "▁" replaced by spaces, trimmed).
        """
    T, V = log_probs.shape
    beams: Dict[tuple, _Beam] = {(): _Beam(p_blank=0.0, p_nonblank=_NEG_INF)}

    for t in range(T):
        frame = log_probs[t]
        blank_lp = float(frame[blank_id])

        # top-k non-blank candidates
        non_blank = np.delete(np.arange(V), blank_id)
        k = min(token_candidates, len(non_blank))
        top_tokens = non_blank[np.argpartition(frame[non_blank], -k)[-k:]]

        new_beams: Dict[tuple, _Beam] = {}

        def merge(prefix: tuple, b: _Beam) -> None:
            """
            Merge a beam into the next-step beam dictionary, accumulating probabilities when the prefix already exists.
            
            Parameters:
                prefix (tuple): Token-index tuple representing the beam prefix key.
                b (_Beam): Beam to insert or merge into the surrounding `new_beams` mapping.
            
            Notes:
                Mutates the outer-scope `new_beams` dictionary by either inserting `b` under `prefix`
                or updating the existing entry's `p_blank` and `p_nonblank` by adding probabilities in log-space.
            """
            if prefix in new_beams:
                e = new_beams[prefix]
                e.p_blank = _logaddexp(e.p_blank, b.p_blank)
                e.p_nonblank = _logaddexp(e.p_nonblank, b.p_nonblank)
            else:
                new_beams[prefix] = b

        for prefix, beam in beams.items():
            prev_total = beam.acoustic
            last_token = prefix[-1] if prefix else None

            # blank extension — keeps same prefix
            merge(prefix, _Beam(
                p_blank=prev_total + blank_lp,
                p_nonblank=_NEG_INF,
                lm_score=beam.lm_score,
                word_pieces=list(beam.word_pieces),
                prev_word=beam.prev_word,
            ))

            for v in top_tokens:
                token_lp = float(frame[v])
                piece = vocab[v] if v < len(vocab) else ""
                is_repeat = (last_token == v)

                new_pieces = list(beam.word_pieces)
                new_prev = beam.prev_word
                lm_delta = 0.0

                if piece.startswith("▁"):
                    completed = "".join(new_pieces)
                    if completed:
                        lm_delta = lm_weight * lm.score(completed, new_prev) + word_bonus
                        new_prev = completed
                    stripped = piece[len("▁"):]
                    new_pieces = [stripped] if stripped else []
                else:
                    new_pieces = new_pieces + [piece]

                if is_repeat:
                    # same-token repeat: can only extend from the blank path
                    merge(prefix, _Beam(
                        p_blank=_NEG_INF,
                        p_nonblank=beam.p_nonblank + token_lp,
                        lm_score=beam.lm_score,
                        word_pieces=list(beam.word_pieces),
                        prev_word=beam.prev_word,
                    ))
                    merge(prefix + (v,), _Beam(
                        p_blank=_NEG_INF,
                        p_nonblank=beam.p_blank + token_lp,
                        lm_score=beam.lm_score + lm_delta,
                        word_pieces=new_pieces,
                        prev_word=new_prev,
                    ))
                else:
                    merge(prefix + (v,), _Beam(
                        p_blank=_NEG_INF,
                        p_nonblank=prev_total + token_lp,
                        lm_score=beam.lm_score + lm_delta,
                        word_pieces=new_pieces,
                        prev_word=new_prev,
                    ))

        beams = dict(sorted(new_beams.items(), key=lambda x: x[1].total, reverse=True)[:beam_width])

    # score the final partial word and pick best hypothesis
    best_score = _NEG_INF
    best_prefix: tuple = ()
    for prefix, beam in beams.items():
        last_word = "".join(beam.word_pieces)
        final_lm = beam.lm_score
        if last_word:
            final_lm += lm_weight * lm.score(last_word, beam.prev_word) + word_bonus
        score = beam.acoustic + final_lm
        if score > best_score:
            best_score = score
            best_prefix = prefix

    return "".join(vocab[i] for i in best_prefix if i < len(vocab)).replace("▁", " ").strip()
