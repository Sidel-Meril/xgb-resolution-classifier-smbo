"""Unicode / punctuation normalization for city strings.

Vendored verbatim from ``data_cleaning.geo.normalize`` so the pair location
features match the ones that produced the shipped classifier byte for byte.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_city_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.casefold()
    name = re.sub(r"[^a-z0-9\s-]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name
