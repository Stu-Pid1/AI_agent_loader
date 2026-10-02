import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Multi-part GGUF files look like "model-00001-of-00003.gguf" — all parts
# belong to the same quantization and must be downloaded/deleted together.
_MULTIPART_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)

# Matches the quantization tag at the end of a GGUF filename, e.g.
# "...Q4_K_M.gguf" or "...-q5_k_s.gguf" or "...F16.gguf".
_QUANT_RE = re.compile(
    r"(?:^|[.\-_])((?:IQ[1-4]_[A-Z0-9_]+|Q[2-8](?:_[A-Z0-9_]+)?|F16|FP16|BF16|F32|FP32))$",
    re.IGNORECASE,
)

# Preferred display order (roughly smallest/lowest-quality to largest/highest).
_QUANT_ORDER = [
    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q3_K",
    "IQ4_XS", "IQ4_NL", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q4_K",
    "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q5_K",
    "Q6_K", "Q8_0", "F16", "FP16", "BF16", "F32", "FP32",
]

ALL_VARIANTS_LABEL = "__ALL__"


@dataclass
class GGUFVariant:
    label: str
    files: List[str] = field(default_factory=list)
    size_bytes: int = 0


def _quant_sort_key(label: str):
    try:
        return (_QUANT_ORDER.index(label), label)
    except ValueError:
        return (len(_QUANT_ORDER), label)


def group_gguf_variants(file_entries: List[Dict]) -> List[GGUFVariant]:
    """Groups a repo's/cache's files into one entry per GGUF quantization.

    `file_entries` is a list of {"filename": str, "size": Optional[int]}.
    Multi-part files (model-00001-of-00003.gguf) are collapsed into a single
    variant. Non-.gguf files are ignored. Returns variants sorted from
    smallest/lowest-quality quant to largest.
    """
    groups: Dict[str, GGUFVariant] = {}
    for entry in file_entries:
        filename = entry.get("filename", "")
        if not filename.lower().endswith(".gguf"):
            continue
        size = entry.get("size") or 0

        multipart = _MULTIPART_RE.search(filename)
        base = filename[: multipart.start()] + ".gguf" if multipart else filename

        stem = base[: -len(".gguf")]
        match = _QUANT_RE.search(stem)
        label = match.group(1).upper() if match else base

        variant = groups.setdefault(label, GGUFVariant(label=label))
        variant.files.append(filename)
        variant.size_bytes += size

    return sorted(groups.values(), key=lambda v: _quant_sort_key(v.label))


def pick_default_variant(variants: List[GGUFVariant]) -> Optional[str]:
    """A sensible default quantization to pre-select — Q4_K_M if present,
    the nearest similarly-sized alternative otherwise, else the first."""
    if not variants:
        return None
    for preferred in ("Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M"):
        for v in variants:
            if v.label == preferred:
                return v.label
    return variants[0].label
