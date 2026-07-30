import streamlit as st
import requests

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
API_URL = "http://127.0.0.1:8000/predict"  # change this if your FastAPI runs elsewhere

st.set_page_config(page_title="Cancer Detection", page_icon="🩺", layout="centered")

st.title("🩺 Breast Cancer Detection")
st.write(
    "Enter the tumor measurements below. This app sends the data to a FastAPI "
    "backend serving an MLflow model (`CancerDetectionModel`) and shows the prediction."
)

# ---------------------------------------------------------
# FEATURE GROUPS (30 features total, same order as backend schema)
# ---------------------------------------------------------
FEATURES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]
SUFFIXES = {"mean": "_mean", "se": "_se", "worst": "_worst"}

# Reasonable default values (dataset means) so the form isn't empty
DEFAULTS = {
    "radius_mean": 14.1, "texture_mean": 19.3, "perimeter_mean": 91.9, "area_mean": 654.9,
    "smoothness_mean": 0.096, "compactness_mean": 0.104, "concavity_mean": 0.089,
    "concave_points_mean": 0.048, "symmetry_mean": 0.181, "fractal_dimension_mean": 0.063,
    "radius_se": 0.405, "texture_se": 1.217, "perimeter_se": 2.866, "area_se": 40.34,
    "smoothness_se": 0.007, "compactness_se": 0.025, "concavity_se": 0.032,
    "concave_points_se": 0.012, "symmetry_se": 0.021, "fractal_dimension_se": 0.004,
    "radius_worst": 16.27, "texture_worst": 25.68, "perimeter_worst": 107.3, "area_worst": 880.6,
    "smoothness_worst": 0.132, "compactness_worst": 0.254, "concavity_worst": 0.272,
    "concave_points_worst": 0.115, "symmetry_worst": 0.290, "fractal_dimension_worst": 0.084,
}

st.subheader("Tumor Measurements")

input_data = {}

tab_mean, tab_se, tab_worst = st.tabs(["📏 Mean", "📊 Standard Error", "⚠️ Worst"])

with tab_mean:
    cols = st.columns(2)
    for i, feat in enumerate(FEATURES):
        key = feat + SUFFIXES["mean"]
        with cols[i % 2]:
            input_data[key] = st.number_input(
                key.replace("_", " ").title(), value=DEFAULTS[key], format="%.4f", key=key
            )

with tab_se:
    cols = st.columns(2)
    for i, feat in enumerate(FEATURES):
        key = feat + SUFFIXES["se"]
        with cols[i % 2]:
            input_data[key] = st.number_input(
                key.replace("_", " ").title(), value=DEFAULTS[key], format="%.4f", key=key
            )

with tab_worst:
    cols = st.columns(2)
    for i, feat in enumerate(FEATURES):
        key = feat + SUFFIXES["worst"]
        with cols[i % 2]:
            input_data[key] = st.number_input(
                key.replace("_", " ").title(), value=DEFAULTS[key], format="%.4f", key=key
            )

st.divider()

# ---------------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------------
if st.button("🔍 Predict", use_container_width=True, type="primary"):
    try:
        with st.spinner("Contacting model server..."):
            response = requests.post(API_URL, json=input_data, timeout=10)

        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction", [None])[0]

            st.success("Prediction received")

            # Adjust this mapping based on how your model was trained
            # (commonly 0 = Malignant, 1 = Benign for the sklearn breast cancer dataset)
            if prediction == 1:
                st.markdown("### 🟢 Result: **Benign**")
            elif prediction == 0:
                st.markdown("### 🔴 Result: **Malignant**")
            else:
                st.markdown(f"### Result: `{prediction}`")

            with st.expander("Raw API response"):
                st.json(result)
        else:
            st.error(f"API returned status code {response.status_code}")
            st.code(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the backend API. Make sure your FastAPI server "
            f"is running and reachable at `{API_URL}`."
        )
    except requests.exceptions.Timeout:
        st.error("The request timed out. The model server may be slow to respond.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

st.divider()
with st.expander("ℹ️ How this works"):
    st.write(
        "- This form collects the 30 features expected by the `CancerData` Pydantic model.\n"
        "- On clicking **Predict**, the values are sent as a JSON POST request to your "
        "FastAPI `/predict` endpoint.\n"
        "- The FastAPI backend loads `CancerDetectionModel` from MLflow and returns a prediction.\n"
        "- Update the `API_URL` at the top of this file if your backend runs on a different host/port."
    )