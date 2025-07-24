from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def train_model(X, y):
    """
    Обучает модель линейной регрессии на данных X и y.
    Возвращает обученную модель и MSE на обучающей выборке.
    """
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    return model, mse
