
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import io
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


def preprocess_and_predict(df: pd.DataFrame):
    """Xu ly 1 hoac nhieu dong khach hang (DataFrame) va tra ve du doan."""
    df = df.copy()
    df.columns = df.columns.str.replace(".", "_", regex=False)

    text = df["tariff_plan_conds"].fillna("") if "tariff_plan_conds" in df.columns else pd.Series([""] * len(df))

    drop_cols = [c for c in ["Cust_ID", "tariff_plan_conds", "Offers"] if c in df.columns]
    structured = df.drop(columns=drop_cols)

    structured["International_plan"] = structured["International_plan"].map({"Yes": 1, "No": 0})
    structured["Voice_mail_plan"] = structured["Voice_mail_plan"].map({"Yes": 1, "No": 0})
    structured = pd.get_dummies(structured, columns=["State", "Area_code"])
    structured = structured.reindex(columns=structured_cols, fill_value=0)

    text_vec = tfidf.transform(text)
    combined = hstack([csr_matrix(structured.values.astype(float)), text_vec]).tocsr()

    pred_idx = model.predict(combined)
    pred_labels = label_encoder.inverse_transform(pred_idx)

    results = []
    proba_matrix = model.predict_proba(combined) if hasattr(model, "predict_proba") else None
    for i, label in enumerate(pred_labels):
        item = {"offer_du_doan": label}
        if proba_matrix is not None:
            for j, p in enumerate(proba_matrix[i]):
                item[f"xac_suat_{label_encoder.classes_[j]}"] = float(p)
        results.append(item)
    return results


@app.get("/")
def root():
    return {"status": "ok", "message": "Telco Next Best Offer API dang chay"}


@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.dict()])
    result = preprocess_and_predict(df)[0]
    flat_keys = [k for k in result if k.startswith("xac_suat_")]
    if flat_keys:
        result["xac_suat"] = {k.replace("xac_suat_", ""): result.pop(k) for k in flat_keys}
    return result


@app.post("/predict_batch")
async def predict_batch(
    file: UploadFile = File(...),
    output_format: str = Query("json", enum=["json", "csv"])
):
    """Nhan 1 file CSV chua nhieu khach hang (chua co cot Offers), tra ve du doan cho tung dong.
    output_format=json (mac dinh) tra ve JSON, output_format=csv tra ve file CSV tai ve truc tiep."""
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    predictions = preprocess_and_predict(df)
    pred_df = pd.DataFrame(predictions)

    result_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)

    if output_format == "csv":
        buffer = io.StringIO()
        result_df.to_csv(buffer, index=False, encoding="utf-8-sig")
        buffer.seek(0)
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=ket_qua_du_doan.csv"}
        )

    return {"so_luong": len(result_df), "ket_qua": result_df.to_dict(orient="records")}
