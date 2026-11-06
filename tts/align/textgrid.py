"""TextGrid parsing rules (docs/PLAN.md Section 3, "TextGrid parsing rules").

1. Read with praatio. Phone labels "", "sil", "sp" are silence; "spn" is a failed word.
2. Leading / trailing silence -> `sil` (always present, possibly 0 frames). Internal silence ->
   `sp` if >= min_sp_frames frames, otherwise merged into the preceding phone.
3. Boundary frame = round(t * sr / hop) on cumulative times; durations are differences; the last
   token is adjusted so durations sum to the training-spectrogram frame count.
4. Each phone token stores its word index; `sil` / `sp` store -1.

The same function is used for LJSpeech, L2 references and model outputs (Section 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig
from praatio import textgrid as ptg

from tts.text.symbols import SIL, SP


class AlignmentError(ValueError):
    """Alignment unusable; `reason` is a short machine-readable tag for QC reports."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass
class Interval:
    start: float
    end: float
    label: str


@dataclass
class Word:
    word: str
    start: float
    end: float


@dataclass
class Parsed:
    phones: list[str]
    durations: list[int]
    word_ids: list[int]
    words: list[Word]                     # aligned (non-silence) words in order, with times
    raw_frame_sum: int                    # sum of durations before the last-token adjustment
    max_phone_sec: float                  # longest non-silence phone (seconds)
    n_merged_short_sil: int = 0
    extra: dict = field(default_factory=dict)


def read_tiers(path: str | Path) -> tuple[list[Interval], list[Interval]]:
    """(words, phones) intervals from an MFA TextGrid, including empty-label gaps."""
    tg = ptg.openTextgrid(str(path), includeEmptyIntervals=True)
    names = {n.lower(): n for n in tg.tierNames}
    if "words" not in names or "phones" not in names:
        raise AlignmentError("bad_textgrid", f"tiers {tg.tierNames} in {path}")

    def tier(name):
        return [Interval(float(e.start), float(e.end), e.label.strip())
                for e in tg.getTier(names[name]).entries]
    return tier("words"), tier("phones")


def frame(t: float, sr: int, hop: int) -> int:
    return int(round(t * sr / hop))


def parse(words: list[Interval], phones: list[Interval], n_frames: int, cfg: DictConfig,
          expected_words: list[str] | None = None) -> Parsed:
    """Apply rules 1-4. `n_frames` comes from the training spectrogram of the original audio."""
    sil_labels = set(cfg.align.silence_labels)
    spn = cfg.align.unknown_label
    sr, hop = int(cfg.audio.sample_rate), int(cfg.audio.hop_length)
    min_sp = int(cfg.align.min_sp_frames)

    phones = sorted(phones, key=lambda x: x.start)
    if not phones:
        raise AlignmentError("empty", "no phone intervals")
    if any(p.label == spn for p in phones) or any(w.label == spn or w.label == "<unk>"
                                                 for w in words):
        raise AlignmentError("spn", "unknown word in alignment")

    # Aligned words (non-silence) and phone -> word index by time containment.
    aligned = [Word(w.label, w.start, w.end) for w in sorted(words, key=lambda x: x.start)
               if w.label not in sil_labels]
    if not aligned:
        raise AlignmentError("no_words")
    if expected_words is not None and [w.word for w in aligned] != list(expected_words):
        raise AlignmentError("word_mismatch",
                             f"aligned {[w.word for w in aligned][:8]}... vs "
                             f"expected {list(expected_words)[:8]}...")

    def word_of(p: Interval) -> int:
        mid = 0.5 * (p.start + p.end)
        for i, w in enumerate(aligned):
            if w.start - 1e-6 <= mid <= w.end + 1e-6:
                return i
        raise AlignmentError("orphan_phone", f"{p.label} at {p.start:.3f}s outside every word")

    # Collapse to a token stream of speech phones and silence spans (adjacent silences merged).
    tokens: list[list] = []  # [label, start, end, word_id]
    for p in phones:
        if p.label in sil_labels:
            if tokens and tokens[-1][0] is None:
                tokens[-1][2] = p.end
            else:
                tokens.append([None, p.start, p.end, -1])
        else:
            tokens.append([p.label, p.start, p.end, word_of(p)])
    max_phone = max((t[2] - t[1] for t in tokens if t[0] is not None), default=0.0)

    # Boundaries in frames on cumulative times; the timeline starts at 0 (Section 3 rule 3).
    starts = [frame(t[1], sr, hop) for t in tokens]
    ends = [frame(t[2], sr, hop) for t in tokens]
    if starts[0] != 0:
        raise AlignmentError("bad_textgrid", f"first interval starts at {tokens[0][1]}s")

    out_p, out_d, out_w = [], [], []
    merged = 0
    last_speech = max(i for i, t in enumerate(tokens) if t[0] is not None)
    first_speech = min(i for i, t in enumerate(tokens) if t[0] is not None)
    for i, (lab, _, _, wid) in enumerate(tokens):
        d = ends[i] - starts[i]
        if lab is not None:
            out_p.append(lab), out_d.append(d), out_w.append(wid)
        elif i < first_speech or i > last_speech:
            out_p.append(SIL), out_d.append(d), out_w.append(-1)
        elif d >= min_sp:
            out_p.append(SP), out_d.append(d), out_w.append(-1)
        else:
            out_d[-1] += d  # short internal silence: merged into the preceding phone (rule 2)
            merged += 1
    # Leading / trailing sil always present (0 frames if speech touches the edge).
    if out_p[0] != SIL:
        out_p.insert(0, SIL), out_d.insert(0, 0), out_w.insert(0, -1)
    if out_p[-1] != SIL:
        out_p.append(SIL), out_d.append(0), out_w.append(-1)

    raw = sum(out_d)
    out_d[-1] += n_frames - raw
    if out_d[-1] < 0:
        # trailing sil too short to absorb the excess: take it from the last speech token(s)
        deficit = -out_d[-1]
        out_d[-1] = 0
        j = len(out_d) - 2
        while deficit and j >= 0:
            take = min(deficit, out_d[j] - (1 if out_p[j] not in (SIL, SP) else 0))
            out_d[j] -= max(take, 0)
            deficit -= max(take, 0)
            j -= 1
        if deficit:
            raise AlignmentError("frame_mismatch", "cannot fit durations to n_frames")
    zero = _fill_zero_phones(out_p, out_d)
    assert sum(out_d) == n_frames
    return Parsed(out_p, out_d, out_w, aligned, raw, max_phone, merged,
                  extra={"zero_frame_fixed": zero})


def _fill_zero_phones(phones: list[str], durs: list[int]) -> int:
    """Every speech phone gets >= 1 frame (D-33): a 0-frame phone takes one frame from its longer
    neighbour (that keeps >= 1 frame, or >= 0 for sil/sp). Sum is unchanged; boundaries move by at
    most one frame. Returns the number of phones fixed."""
    fixed = 0
    for i, (ph, d) in enumerate(zip(phones, durs)):
        if d > 0 or ph in (SIL, SP):
            continue
        best = None
        for j in (i - 1, i + 1):
            if 0 <= j < len(durs):
                floor = 0 if phones[j] in (SIL, SP) else 1
                if durs[j] > floor and (best is None or durs[j] > durs[best]):
                    best = j
        if best is None:
            raise AlignmentError("frame_mismatch", f"cannot give phone {i} ({ph}) a frame")
        durs[best] -= 1
        durs[i] = 1
        fixed += 1
    return fixed


def parse_file(path: str | Path, n_frames: int, cfg: DictConfig,
               expected_words: list[str] | None = None) -> Parsed:
    words, phones = read_tiers(path)
    return parse(words, phones, n_frames, cfg, expected_words)


def qc_reason(p: Parsed, n_frames: int, cfg: DictConfig) -> str | None:
    """LJSpeech quality checks (Section 3). None = keep."""
    if p.max_phone_sec > cfg.align.qc.max_phone_sec:
        return "long_phone"
    if abs(p.raw_frame_sum - n_frames) > cfg.align.qc.max_frame_diff:
        return "frame_diff"
    return None
