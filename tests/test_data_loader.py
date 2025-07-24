import pandas as pd
from src.data_loader import preprocess_data


def test_preprocess_data():
    # Примерные данные
    data = {
        "Date/Time": pd.to_datetime([
            "2024-07-01 10:05:00",
            "2024-07-01 10:45:00",
            "2024-07-01 11:00:00"
        ]),
        "Lat": [40.7, 40.7, 40.8],
        "Lon": [-73.9, -73.9, -73.8],
        "Base": ["B02512", "B02512", "B02512"]
    }

    df = pd.DataFrame(data)
    X, y = preprocess_data(df)

    # Проверка размерности (две поездки в 10:00 и одна в 11:00)
    assert len(X) == 2
    assert len(y) == 2

    # Проверка наличия нужных колонок
    assert all(col in X.columns for col in ["hour", "day_of_week", "month"])

    # Проверка типа целевой переменной
    assert pd.api.types.is_numeric_dtype(y)
