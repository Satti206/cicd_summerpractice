import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from src.data_loader import load_sample_data

# Пути
DATA_PATH = '/Users/satti/PycharmProjects/cicd_summerpractice/data/raw/uber-raw-data-may14.csv'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'uber_model.joblib')

# Создание директории models, если её нет
os.makedirs(MODEL_DIR, exist_ok=True)

# Загрузка данных
X_train, X_test, y_train, y_test = load_sample_data(DATA_PATH)

# Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Оценка качества
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'✅ MSE on test set: {mse:.2f}')

# Сохранение модели
joblib.dump(model, MODEL_PATH)
print(f'✅ Model saved to {MODEL_PATH}')

# Вывод признаков
print("📊 Features used:", model.feature_names_in_)
