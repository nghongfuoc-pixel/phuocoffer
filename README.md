# Telco Next Best Offer API

## Chay local
```
pip install -r requirements.txt
uvicorn app:app --reload
```
Sau do goi API: POST http://localhost:8000/predict

## Deploy len Render (co san Dockerfile)
1. Push thu muc nay len 1 GitHub repo
2. Tren Render Dashboard: New -> Web Service -> chon repo vua tao
3. Render tu dong nhan Dockerfile, chon san Runtime: Docker (khong can nhap Build/Start Command)
4. Instance Type: Free -> Create Web Service
5. Doi build xong, se co URL dang https://ten-service.onrender.com
