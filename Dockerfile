FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render cap cong qua bien moi truong PORT (mac dinh 10000 neu khong duoc set)
ENV PORT=10000
EXPOSE 10000

CMD uvicorn app:app --host 0.0.0.0 --port $PORT
