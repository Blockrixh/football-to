# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# 모델 불러오기
@st.cache_resource
def load_models():
    model_cat = joblib.load("model_cat.pkl")
    model_xgb = xgb.Booster()
    model_xgb.load_model("model_xgb.json")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model_cat, model_xgb, label_encoder, feature_columns

model_cat, model_xgb, label_encoder, feature_columns = load_models()

# UI
st.title("📊 일반 경기 결과 예측 (before.csv 전용)")

uploaded_file = st.file_uploader("before.csv 파일 업로드", type="csv")

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    df_general = df_input[df_input["type"] == "일반"].copy()

    if df_general.empty:
        st.warning("⚠ 업로드한 파일에 일반 경기가 없습니다.")
    else:
        X = pd.get_dummies(df_general)
        X = X.reindex(columns=feature_columns, fill_value=0)

        # 예측
        proba_cat = model_cat.predict_proba(X)
        dtest = xgb.DMatrix(X)
        proba_xgb = model_xgb.predict(dtest)
        proba_ensemble = (proba_cat + proba_xgb) / 2
        pred = np.argmax(proba_ensemble, axis=1)

        df_general["result"] = label_encoder.inverse_transform(pred)

        st.success("✅ 예측 완료!")
        st.dataframe(df_general)

        csv = df_general.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 예측 결과 다운로드", csv, "prediction_result.csv", "text/csv")
