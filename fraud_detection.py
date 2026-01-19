import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection System")
st.caption("AI & Machine Learning Based Application")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Transaction Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload your CSV file to continue")
    st.stop()

# ---------------- LOAD DATA ----------------
data = pd.read_csv(uploaded_file)
data.columns = data.columns.str.strip()   # remove spaces

st.success("Dataset Loaded Successfully")
st.write("Dataset Preview:")
st.write(data.head())

# ---------------- HANDLE DATETIME COLUMNS ----------------
for col in data.columns:
    if data[col].dtype == object:
        try:
            pd.to_datetime(data[col])
            data.drop(col, axis=1, inplace=True)
        except:
            pass

# ---------------- ENCODE CATEGORICAL DATA ----------------
label_encoder = LabelEncoder()
for col in data.select_dtypes(include=["object"]).columns:
    data[col] = label_encoder.fit_transform(data[col])

# ---------------- TARGET & FEATURES ----------------
if "IsFraud" not in data.columns:
    st.error("Target column 'IsFraud' not found in dataset")
    st.stop()

X = data.drop("IsFraud", axis=1)
y = data["IsFraud"]

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- TRAIN MODEL ----------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# ---------------- MODEL ACCURACY ----------------
accuracy = accuracy_score(y_test, model.predict(X_test))
st.info(f"Model Accuracy: {accuracy*100:.2f}%")

# ---------------- CLASSIFICATION REPORT ----------------
if st.checkbox("Show Classification Report"):
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

# ---------------- PREDICTION SECTION ----------------
st.subheader("🔍 Test a Transaction")

index = st.number_input(
    "Select Transaction Index",
    min_value=0,
    max_value=len(X_test)-1,
    value=0
)

sample = X_test.iloc[[index]]

if st.button("Detect Fraud"):
    prediction = model.predict(sample)[0]
    if prediction == 1:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✅ Genuine Transaction")
