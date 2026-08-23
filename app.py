
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix

app = FastAPI(title="Telco Next Best Offer API")

model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
structured_cols = joblib.load("structured_columns.pkl")


class Customer(BaseModel):
    State: str
    Account_length: int
    Area_code: int
    International_plan: str
    Voice_mail_plan: str
    Number_vmail_messages: int
    Total_day_minutes: float
    Total_day_calls: int
    Total_day_charge: float
    Total_eve_minutes: float
    Total_eve_calls: int
    Total_eve_charge: float
    Total_night_minutes: float
    Total_night_calls: int
    Total_night_charge: float
    Total_intl_minutes: float
    Total_intl_calls: int
    Total_intl_charge: float
    Customer_service_calls: int
    tariff_plan_conds: str = ""


@app.get("/")
def root():
    return {"status": "ok", "message": "Telco Next Best Offer API dang chay"}


@app.post("/predict")
def predict(customer: Customer):
    data = customer.dict()
    text = data.pop("tariff_plan_conds", "")

    row = pd.DataFrame([data])
    row["International_plan"] = row["International_plan"].map({"Yes": 1, "No": 0})
    row["Voice_mail_plan"] = row["Voice_mail_plan"].map({"Yes": 1, "No": 0})
    row = pd.get_dummies(row, columns=["State", "Area_code"])
    row = row.reindex(columns=structured_cols, fill_value=0)

    text_vec = tfidf.transform([text])
    combined = hstack([csr_matrix(row.values.astype(float)), text_vec]).tocsr()

    pred_idx = model.predict(combined)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    result = {"offer_du_doan": pred_label}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(combined)[0]
        result["xac_suat"] = {
            label_encoder.classes_[i]: float(p) for i, p in enumerate(proba)
        }
    return result
