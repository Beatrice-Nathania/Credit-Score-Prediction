"""
Streamlit UI for Credit Score Model hosted on SageMaker.
Reads endpoint name and region from environment variables.
"""

import json
import os
import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

# Ambil dari environment EC2 (atau sesuaikan default endpoint name kamu)
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-scoring-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")


# ================= SAGEMAKER CLIENT =================
@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(features: list) -> dict:
    runtime = get_runtime_client()

    payload = {"instances": [features]}

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )

    return json.loads(response["Body"].read().decode("utf-8"))


# ================= UI CONFIGURATION =================
st.set_page_config(page_title="Credit Score Predictor", page_icon="💳", layout="wide")
st.title("💳 Credit Score Predictor")
st.markdown("Predict customer credit score classification (**Poor**, **Standard**, **Good**).")

# ===== FORM INPUT =====
st.subheader("Customer Profile & Financial Metrics")

# Membagi form menjadi 3 kolom agar tampilan UI lebih rapi dan seimbang
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 Identity & Basic Info")
    unnamed_0 = 0
    id_val = 0
    customer_id = 0
    name = st.text_input("Customer Name", "Beatrice")
    ssn = 0
    age = st.number_input("Age", 32)
    month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June", "July", "August"])
    occupation = st.text_input("Occupation", "Doctor")

with col2:
    st.markdown("### 💰 Income & Debt Banking")
    annual_income = st.text_input("Annual Income", "56125.5")
    monthly_inhand_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=4875.125)
    num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, value=8)
    num_credit_card = st.number_input("Number of Credit Cards", min_value=0, value=3)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0, value=18)
    num_of_loan = st.number_input("Number of Loans", 2)
    type_of_loan = st.text_input("Type of Loans", "Credit-Builder Loan, and Mortgage Loan")
    outstanding_debt = st.number_input("Outstanding Debt", "370.22")

with col3:
    st.markdown("### 📊 Credit Behavior History")
    delay_from_due_date = st.number_input("Delay from Due Date (Days)", min_value=0, value=30)
    num_of_delayed_payment = st.number_input("Number of Delayed Payments", 14)
    changed_credit_limit = st.number_input("Changed Credit Limit", 17.89)
    num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0.0, value=4.0)
    credit_mix = st.selectbox("Credit Mix", ["Good", "Standard", "Bad"])
    credit_utilization_ratio = st.number_input("Credit Utilization Ratio", min_value=0.0, value=32.01418)
    credit_history_age = st.text_input("Credit History Age", "28 Years and 10 Months")
    payment_of_min_amount = st.selectbox("Payment of Minimum Amount", ["Yes", "No"])
    total_emi_per_month = st.number_input("Total EMI per Month", min_value=0.0, value=81.8228)
    amount_invested_monthly = st.number_input("Amount Invested Monthly", 182.0655)
    payment_behaviour = st.selectbox("Payment Behaviour", [
        "High_spent_Medium_value_payments",
        "High_spent_Small_value_payments",
        "High_spent_Large_value_payments",
        "Low_spent_Medium_value_payments",
        "Low_spent_Small_value_payments",
        "Low_spent_Large_value_payments"
    ])
    monthly_balance = st.text_input("Monthly Balance", "473.6241")


# ===== PREDICT BUTTON =====
st.markdown("---")
if st.button("Analyze Credit Score", type="primary", use_container_width=True):

    # Susunan list harus presisi 100% sama dengan urutan FEATURE_NAMES di inference.py
    features = [
        unnamed_0,
        id_val,
        customer_id,
        month,
        name,
        age,
        ssn,
        occupation,
        annual_income,
        monthly_inhand_salary,
        num_bank_accounts,
        num_credit_card,
        interest_rate,
        num_of_loan,
        type_of_loan,
        delay_from_due_date,
        num_of_delayed_payment,
        changed_credit_limit,
        num_credit_inquiries,
        credit_mix,
        outstanding_debt,
        credit_utilization_ratio,
        credit_history_age,
        payment_of_min_amount,
        total_emi_per_month,
        amount_invested_monthly,
        payment_behaviour,
        monthly_balance
    ]

    try:
        with st.spinner("Invoking SageMaker endpoint... Please wait."):
            result = invoke_endpoint(features)

    except NoCredentialsError:
        st.error(
            "AWS Credentials tidak ditemukan.\n"
            "Pastikan LabInstanceProfile sudah terpasang jika dijalankan di EC2."
        )

    except ClientError as e:
        st.error(f"SageMaker Endpoint Error: {e.response['Error'].get('Message', str(e))}")

    else:
        st.subheader("🎯 Prediction Result")
        
        label = result["labels"][0]          # Teks string: "Poor", "Standard", atau "Good"
        probs = result["probabilities"][0]    # Array float probabilitas nilai kelas

        # Menampilkan metrik utama berdasarkan hasil klasifikasi skor kredit
        if label == "Good":
            st.success(f"### Result: **{label}** Kredit ✅ (Nasabah Sangat Aman)")
        elif label == "Standard":
            st.warning(f"### Result: **{label}** Kredit ⚠️ (Nasabah Cukup Aman)")
        else:
            st.error(f"### Result: **{label}** Kredit ❌ (Nasabah Berisiko Tinggi)")

        # Visualisasi diagram probabilitas kelas prediksi
        st.write("#### Class Probabilities Graph:")
        chart_data = {
            "Credit Score Class": ["Poor", "Standard", "Good"],
            "Probability Score": probs
        }
        st.bar_chart(data=chart_data, x="Credit Score Class", y="Probability Score")
