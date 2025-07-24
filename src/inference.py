import os
import joblib
import pandas as pd
from datetime import datetime
from src.data_loader import preprocess_data

# Пути
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'uber_model.joblib')
INPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'uber-raw-data-may14.csv')
PRED_PATH = 'predictions.csv'
REPORT_PATH = 'report.html'

# Загрузка модели
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Модель не найдена: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Загрузка данных
raw_data = pd.read_csv(INPUT_PATH, parse_dates=["Date/Time"]).head(10)

# Предобработка
X, _ = preprocess_data(raw_data)
X = X.reindex(columns=model.feature_names_in_, fill_value=0)

# Предсказания
preds = model.predict(X)
X["predicted_rides"] = preds
X.to_csv(PRED_PATH, index=False)
print(f"✅ Предсказания сохранены в {PRED_PATH}")

# HTML отчёт
html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Uber Inference Report</title>
</head>
<body>
    <h1>🚕 Uber Inference Report</h1>
    <p><strong>Дата:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Количество примеров:</strong> {len(preds)}</p>
    <p><strong>Предсказания (первые 10):</strong></p>
    {X[['hour', 'day_of_week', 'month', 'predicted_rides']].to_html(index=False)}
</body>
</html>
"""

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"📄 Отчёт сохранён в {REPORT_PATH}")
