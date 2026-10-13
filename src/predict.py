"""Load the trained model and score one machine reading."""
import json
from functools import lru_cache
from pathlib import Path

import joblib

from src.preprocessing import row_from_input

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@lru_cache(maxsize=1)
def _load():
    model_file = MODELS_DIR / "best_model.joblib"
    if not model_file.exists():
        raise FileNotFoundError("Model not trained yet. Run:  python -m src.train")
    return joblib.load(model_file), json.loads((MODELS_DIR / "metrics.json").read_text())


def get_metrics() -> dict:
    return _load()[1]


def predict_one(reading: dict) -> dict:
    model, meta = _load()
    prob = float(model.predict_proba(row_from_input(reading))[0, 1])
    thr = meta["threshold"]
    status = "failure" if prob >= thr else "elevated" if prob >= thr / 2 else "normal"
    return {"probability": round(prob, 4), "threshold": round(thr, 3), "status": status}


if __name__ == "__main__":
    demo = dict(air_temp=300, process_temp=309, rpm=1300, torque=65, tool_wear=220, type="L")
    print(predict_one(demo))
