import ast
import random
import re

_PROTECT_RE = re.compile(r"""^(?:[0-9]+(?:[.,][0-9]+)?%?|\.[A-Z]{2,5}|[A-Z]{2,5}$)$""", re.X)

def _is_protected(tok: str) -> bool:
    return bool(_PROTECT_RE.match(tok))

def noise_sentence_mask(sentence: str, tokenizer, p_mask: float = 0.2, min_keep: int = 5) -> str:
    toks = sentence.split()
    if len(toks) <= min_keep:
        return sentence
    return " ".join(t if _is_protected(t) or random.random() >= p_mask else tokenizer.mask_token for t in toks)

def _mild_random_mask(sentence: str, tokenizer, p_mask: float = 0.05, min_keep: int = 5) -> str:
    toks = sentence.split()
    if len(toks) <= min_keep:
        return sentence
    mask_tok = tokenizer.mask_token or "<mask>"
    out = [t if _is_protected(t) or random.random() >= p_mask else mask_tok for t in toks]
    return " ".join(out) if len(out) >= min_keep else sentence

def _make_focus_mask(seg_locs_list, type_ids_list, n_tokens):
    try:
        types = set(int(x) for x in ast.literal_eval(type_ids_list))
    except Exception:
        types = set()
    try:
        segs = [int(x) for x in ast.literal_eval(seg_locs_list)]
    except Exception:
        segs = [0] * n_tokens
    segs = (segs + [0] * n_tokens)[:n_tokens]
    return [(s in types and s != 0) for s in segs]

def _delete_clause_near_focus(toks, focus, p_clause=0.20):
    if random.random() >= p_clause:
        return toks
    commas = [i for i,t in enumerate(toks) if t.endswith(",") or t == ","]
    if not commas:
        return toks
    try:
        f_idx = focus.index(True)
    except ValueError:
        return toks
    left = max([c for c in commas if c < f_idx], default=None)
    right = min([c for c in commas if c > f_idx], default=None)
    if left is None or right is None:
        return toks
    span = list(range(left+1, right))
    if 3 <= len(span) <= 12 and any(focus[i] for i in span):
        return [t for i,t in enumerate(toks) if i not in span]
    return toks

def _type_aware_augment(sentence: str, seg_locs_list, type_ids_list, tokenizer,
                        p_edit: float = 0.30, min_keep: int = 5) -> str:
    toks = sentence.split()
    if len(toks) <= min_keep:
        return sentence
    focus = _make_focus_mask(seg_locs_list, type_ids_list, len(toks))
    if not any(focus):
        return sentence
    toks = _delete_clause_near_focus(toks, focus)
    mask_tok = tokenizer.mask_token or "<mask>"
    out = []
    for i,t in enumerate(toks):
        if i < len(focus) and focus[i] and not _is_protected(t) and random.random() < p_edit:
            r = random.random()
            if r < 0.45:
                out.append(mask_tok)
            elif r < 0.80:
                continue
            else:
                out.append(mask_tok)
        else:
            out.append(t)
    return " ".join(out) if len(out) >= min_keep else sentence
