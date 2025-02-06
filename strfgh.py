import streamlit as st
import tensorflow as tf
import numpy as np
import sqlite3
import cv2
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# ✅ Ensure page config is at the top
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# ✅ Initialize SQLite Database
conn = sqlite3.connect("predictions.db", check_same_thread=False)
cursor = conn.cursor()

# ✅ Create table if not exists (Ensures confidence is stored as FLOAT)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT,
        prediction_result TEXT,
        confidence FLOAT,
        timestamp TEXT
    )
""")
conn.commit()

# ✅ Load the trained model (cache for efficiency)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("optimized_brain_tumor_model.h5")

model = load_model()

# ✅ Image preprocessing function
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize image to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# ✅ Prediction function
def predict_tumor(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]  # Get prediction score
    return prediction

# ✅ Store prediction in database (Ensures confidence is stored as a float)
def save_prediction(image_name, result, confidence):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    confidence = float(confidence)  # ✅ Ensure it's a float before inserting
    cursor.execute("""
        INSERT INTO predictions (image_name, prediction_result, confidence, timestamp) 
        VALUES (?, ?, ?, ?)""", (image_name, result, confidence, timestamp))
    conn.commit()

# ✅ Fetch past predictions (Handles corrupt data safely)
def get_past_predictions():
    cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
    records = cursor.fetchall()

    fixed_data = []
    for row in records:
        try:
            confidence_value = float(row[3])  # ✅ Try to convert confidence to float
        except (ValueError, TypeError):
            confidence_value = 0.0  # ✅ Default confidence if data is corrupted

        fixed_data.append((row[1], row[2], confidence_value, row[4]))  # ✅ Store clean data

    return fixed_data  # ✅ Returns only valid float confidence values

# ✅ Streamlit UI
st.title("🧠 Brain Tumor Detection Using CNN")
st.write("Upload a brain scan image to check for the presence of a tumor.")

# Sidebar for file upload
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a brain scan image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image_name = uploaded_file.name  # Get image filename

        # Resize image for display
        display_image = image.resize((300, 300))
        st.image(display_image, caption="Uploaded Image", use_container_width=False)

        if st.button("🔍 Predict"):
            with st.spinner("Analyzing image... ⏳"):
                prediction = predict_tumor(image)

            # Convert prediction score to result
            if prediction > 0.5:
                result = "Tumor Detected"
                confidence = prediction
                st.error(f"🚨 {result}! Confidence: {confidence:.2f}")
            else:
                result = "No Tumor Detected"
                confidence = 1 - prediction
                st.success(f"✅ {result}. Confidence: {confidence:.2f}")

            # ✅ Save prediction to database
            save_prediction(image_name, result, confidence)

            # Display probability as a bar chart
            st.subheader("Prediction Confidence")
            fig, ax = plt.subplots()
            sns.barplot(x=["Healthy", "Tumor"], y=[1 - prediction, prediction], palette=["green", "red"], ax=ax)
            ax.set_ylim(0, 1)
            st.pyplot(fig)

    except Exception as e:
        st.error("Error processing the image. Please try a different file.")

# ✅ Show past predictions
st.sidebar.subheader("📋 Past Predictions")
past_predictions = get_past_predictions()

if past_predictions:
    st.sidebar.write("Recent Predictions:")
    for row in past_predictions[:5]:  # Show last 5 predictions
        st.sidebar.write(f"🖼️ {row[0]} - **{row[1]}** (Conf: {row[2]:.2f}) 🕒 {row[3]}")
    st.sidebar.write("[View All Predictions Below]")

st.subheader("📊 Past Predictions History")
if past_predictions:
    st.table(past_predictions)
else:
    st.info("No past predictions available.")

# Footer
st.markdown("\n\n© 2025 Brain Tumor Detection App")
