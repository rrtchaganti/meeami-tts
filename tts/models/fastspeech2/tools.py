"""Helpers from ming024/FastSpeech2 utils/tools.py (MIT); `device` taken from the input."""
import torch
import torch.nn.functional as F


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    return ids >= lengths.unsqueeze(1).expand(-1, max_len)


def pad(input_ele, mel_max_length=None):
    max_len = mel_max_length or max(x.size(0) for x in input_ele)
    out = []
    for x in input_ele:
        if x.dim() == 1:
            out.append(F.pad(x, (0, max_len - x.size(0)), "constant", 0.0))
        else:
            out.append(F.pad(x, (0, 0, 0, max_len - x.size(0)), "constant", 0.0))
    return torch.stack(out)
