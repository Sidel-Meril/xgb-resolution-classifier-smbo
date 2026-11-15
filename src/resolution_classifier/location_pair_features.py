"""Pair-level features from aggregated geo location JSON.

Vendored verbatim from ``data_cleaning.resolve.location_pair_features`` (only the
``normalize_city_name`` import is rewired to the vendored copy) so the pair
feature matrix matches the one that produced the shipped classifier.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from resolution_classifier.geo.normalize import normalize_city_name

EARTH_RADIUS_KM = 6371.0

LOCATION_PAIR_FEATURES = (
    "location_same_city",
    "location_same_country",
    "location_country_conflict",
    "location_top1_distance_km",
    "location_min_distance_km",
    "location_candidate_count_left",
    "location_candidate_count_right",
    "location_granularity_match",
)


def stored(value: object) -> str | None:
    """Text a parent step already stored. A non-text cell is missing."""
    return value if isinstance(value, str) and value else None


def candidate_list(value: object) -> list[dict[str, object]]:
    """Geo rows for one article. A missing value is empty."""
    if isinstance(value, list):
        items = value
    else:
        text = stored(value)
        if text is None:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, dict) and "candidates" in parsed:
            return candidate_list(parsed.get("candidates"))
        if not isinstance(parsed, list):
            return []
        items = parsed
    return [item for item in items if isinstance(item, dict)]


def place_list(value: object) -> list[dict[str, object]]:
    """Administrative hits inside one candidate ``location`` cell."""
    if isinstance(value, list):
        items = value
    else:
        text = stored(value)
        if text is None:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        items = parsed
    return [item for item in items if isinstance(item, dict)]


def iso3_code(value: object) -> str | None:
    """Uppercase ISO3 on an administrative hit."""
    text = stored(value)
    if text is None or len(text) != 3 or not text.isalpha():
        return None
    return text.upper()


def coordinates(place: dict[str, object]) -> tuple[float, float] | None:
    """Latitude and longitude when both are finite."""
    try:
        lat = float(place.get("lat"))
        lon = float(place.get("lon"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(lat) or not math.isfinite(lon):
        return None
    return lat, lon


def article_side(value: object) -> tuple[list[dict[str, object]], list[dict[str, object]], set[str], set[str], str]:
    """Candidate rows, flattened places, iso3 codes, city keys, granularity."""
    candidates = candidate_list(value)
    places: list[dict[str, object]] = []
    iso3s: set[str] = set()
    city_keys: set[str] = set()
    for candidate in candidates:
        city = stored(candidate.get("city"))
        country = stored(candidate.get("country"))
        if city:
            normalized = normalize_city_name(city)
            if normalized:
                city_keys.add(normalized)
        for place in place_list(candidate.get("location")):
            enriched = dict(place)
            enriched["city"] = city
            enriched["country"] = country
            places.append(enriched)
            code = iso3_code(place.get("iso3"))
            if code is not None:
                iso3s.add(code)
    if city_keys:
        granularity = "city"
    elif iso3s or any(stored(c.get("country")) for c in candidates):
        granularity = "country"
    elif candidates:
        granularity = "region"
    else:
        granularity = "unknown"
    return candidates, places, iso3s, city_keys, granularity


def primary_city_key(candidates: list[dict[str, object]]) -> str | None:
    """Normalized city from the first provided row, else the first row with a city."""
    ordered = sorted(
        candidates,
        key=lambda row: (not bool(row.get("is_provided")), stored(row.get("city")) or ""),
    )
    for candidate in ordered:
        city = stored(candidate.get("city"))
        if city:
            normalized = normalize_city_name(city)
            if normalized:
                return normalized
    return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def comparable_places(left: dict[str, object], right: dict[str, object]) -> bool:
    """True when distance between these hits is meaningful."""
    left_code = iso3_code(left.get("iso3"))
    right_code = iso3_code(right.get("iso3"))
    if left_code and right_code:
        return left_code == right_code
    left_city = stored(left.get("city"))
    right_city = stored(right.get("city"))
    if left_city and right_city:
        return normalize_city_name(left_city) == normalize_city_name(right_city)
    return False


def location_pair_metrics(left_value: object, right_value: object) -> dict[str, float]:
    """Eight location comparison features for one pair of articles."""
    left_candidates, left_places, left_iso, _left_cities, left_gran = article_side(left_value)
    right_candidates, right_places, right_iso, _right_cities, right_gran = article_side(
        right_value
    )
    left_city = primary_city_key(left_candidates)
    right_city = primary_city_key(right_candidates)

    if left_city and right_city:
        same_city = 1.0 if left_city == right_city else 0.0
    else:
        same_city = float("nan")

    if left_iso and right_iso:
        overlap = left_iso & right_iso
        same_country = 1.0 if overlap else 0.0
        country_conflict = 0.0 if overlap else 1.0
    elif left_iso or right_iso:
        same_country = float("nan")
        country_conflict = 0.0
    else:
        same_country = float("nan")
        country_conflict = 0.0

    comparable: list[float] = []
    for left_place in left_places:
        left_coords = coordinates(left_place)
        if left_coords is None:
            continue
        for right_place in right_places:
            if not comparable_places(left_place, right_place):
                continue
            right_coords = coordinates(right_place)
            if right_coords is None:
                continue
            comparable.append(haversine_km(*left_coords, *right_coords))

    min_distance = min(comparable) if comparable else float("nan")
    top1_distance = float("nan")
    if left_gran == "city" and right_gran == "city" and left_places and right_places:
        left_coords = coordinates(left_places[0])
        right_coords = coordinates(right_places[0])
        if left_coords is not None and right_coords is not None:
            top1_distance = haversine_km(*left_coords, *right_coords)

    if left_gran == "country" or right_gran == "country":
        min_distance = float("nan")
        top1_distance = float("nan")

    if left_gran == "unknown" or right_gran == "unknown":
        granularity_match = float("nan")
    elif left_gran == right_gran:
        granularity_match = 1.0
    else:
        granularity_match = 0.0

    return {
        "location_same_city": same_city,
        "location_same_country": same_country,
        "location_country_conflict": country_conflict,
        "location_top1_distance_km": top1_distance,
        "location_min_distance_km": min_distance,
        "location_candidate_count_left": float(len(left_candidates)),
        "location_candidate_count_right": float(len(right_candidates)),
        "location_granularity_match": granularity_match,
    }


def location_pair_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Location columns from ``left_location`` and ``right_location``."""
    if "left_location" not in frame.columns or "right_location" not in frame.columns:
        return pd.DataFrame(
            {column: np.nan for column in LOCATION_PAIR_FEATURES},
            index=frame.index,
        )
    rows = [
        location_pair_metrics(left, right)
        for left, right in zip(
            frame["left_location"].to_numpy(),
            frame["right_location"].to_numpy(),
            strict=True,
        )
    ]
    return pd.DataFrame(rows, index=frame.index)
