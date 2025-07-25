# 🚖 Uber NYC Pickups — CI/CD Анализ и Предсказания

Этот проект — практическая реализация полного CI/CD пайплайна по анализу и предсказанию активности поездок Uber в Нью-Йорке. Мы обучаем модель на данных о времени поездок и разворачиваем FastAPI-приложение в Яндекс Облаке.

---

## 📊 Что мы предсказываем?

Модель предсказывает **количество поездок Uber** в зависимости от:

- 🕒 Часа суток
- 📅 Дня недели
- 📆 Месяца

---

## 🏗️ Структура проекта

```

├── data/                  # Данные (CSV)
├── models/                # Сохранённая модель
├── notebooks/             # Jupyter-исследования
├── predictions/           # CSV и HTML-отчёты
├── src/                   # Исходный код (обработка, обучение, API)
├── tests/                 # Pytest-тесты
├── .github/workflows/     # CI/CD пайплайн
├── requirements.txt       # Зависимости
└── README.md              # Документация

````

---

## 🚀 Используемые технологии

- 🐍 **Python 3.10**
- 📦 **Pandas, scikit-learn**
- 🧪 **Pytest + Flake8**
- 🐳 **Docker**
- 🔁 **GitHub Actions**
- ☁️ **Yandex Cloud Run + YCR + SA**
- ⚡ **FastAPI**

---

## 🔄 CI/CD пайплайн

CI/CD автоматизирован через GitHub Actions:

1. Проверка кода (flake8) и запуск тестов (pytest)
2. Обучение модели + сохранение
3. Генерация отчёта (`predictions.csv`, `report.html`)
4. Сборка и пуш Docker-образа:
   - в **GHCR**
   - затем в **Yandex Container Registry**
5. Деплой FastAPI-приложения в **Yandex Cloud Run**

---

## 📡 Пример API-запроса

POST-запрос к `/predict`:

```json
{
  "hour": 14,
  "day_of_week": 2,
  "month": 5
}
````

Ответ:

```json
{
  "predicted_rides": 213.45
}
```

---


## 📁 Источник данных

> [Uber Pickups in New York City](https://www.kaggle.com/datasets/fivethirtyeight/uber-pickups-in-new-york-city)

---

## 👩‍💻 Автор проекта

Сати Сакаева | 2025
Проект выполнен в рамках учебной практики по CI/CD в анализе данных.
