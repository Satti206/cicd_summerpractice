from src.data_loader import load_sample_data


def test_local_data_loading():
    PATH = "data/raw/uber-raw-data-may14.csv"

    X_train, X_test, y_train, y_test = load_sample_data(PATH)

    # Проверка, что выборки не пустые
    assert X_train.shape[0] > 0, "X_train пустой"
    assert X_test.shape[0] > 0, "X_test пустой"

    # Проверка размерностей
    assert X_train.shape[1] == X_test.shape[1], (
        "Количество признаков не совпадает"
    )
    assert len(y_train) == X_train.shape[0], "X_train и y_train не совпадают"

    print(f"✅ Загружено: {X_train.shape[0]} train, {X_test.shape[0]} test")
