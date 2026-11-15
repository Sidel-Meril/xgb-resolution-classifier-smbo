"""Same-event pair resolution classifier (XGBoost + Platt calibration)."""

from resolution_classifier.model import load_model, load_scorer

__all__ = ["load_model", "load_scorer"]
