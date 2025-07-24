import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str):
    """Загружает данные из CSV и возвращает DataFrame."""
    return pd.read_csv(path, parse_dates=["Date/Time"])


def preprocess_data(df: pd.DataFrame):
    """Предобрабатывает DataFrame:
    - извлекает признаки из времени
    - группирует поездки по часу
    Возвращает (X, y), где y — количество поездок в час.
    """

    df = df.copy()

    # Извлекаем временные признаки
    df["hour"] = df["Date/Time"].dt.hour
    df["day_of_week"] = df["Date/Time"].dt.dayofweek
    df["month"] = df["Date/Time"].dt.month
    df["date_hour"] = df["Date/Time"].dt.floor("h")

    # Группировка по часу
    agg_df = df.groupby(["date_hour", "hour", "day_of_week", "month"]).size().reset_index(name="rides")

    # Формируем X и y
    X = agg_df[["hour", "day_of_week", "month"]]
    y = agg_df["rides"]

    return X, y


def load_and_preprocess(path: str):
    """Загружает данные и возвращает (X, y)."""
    df = load_data(path)
    return preprocess_data(df)


def load_sample_data(path: str, test_size=0.2, random_state=42):
    """Загружает и делит данные на обучающую и тестовую выборки."""
    X, y = load_and_preprocess(path)
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
