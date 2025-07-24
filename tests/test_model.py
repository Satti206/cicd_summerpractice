from src.model import train_model
import pandas as pd


def test_train_model_basic():
    # Простейший синтетический датасет
    X = pd.DataFrame({
        "hour": [8, 9, 10],
        "day_of_week": [0, 1, 2],
        "month": [5, 5, 5]
    })
    y = pd.Series([100, 150, 130])

    model, mse = train_model(X, y)

    assert mse >= 0
    assert hasattr(model, "predict")
