# Используем официальный Python-образ
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальной код
COPY . .

# Открываем порт
EXPOSE 8000

# Команда запуска FastAPI
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
