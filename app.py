import streamlit as st
import google.generativeai as genai
import mimetypes
from pathlib import Path
import api_key

genai.configure(api_key=api_key.api_key)

st.set_page_config(
    page_title="MediScan AI",
    page_icon="🧠",
    layout="centered",
)

st.image("assets/logo.png", width=300)  # Adjust the path and size as needed


st.title("🧠 MediScan AI - Diagnosis Assistant")
st.markdown("Provide patient details and/or upload a medical image to receive AI-based insights and recommendations.")

prompt_path = Path("prompts/medical_analysis_prompt.txt")
if not prompt_path.exists():
    st.error("❌ Prompt file not found. Make sure 'prompts/medical_analysis_prompt.txt' exists.")
    st.stop()

with open(prompt_path, "r", encoding="utf-8") as f:
    system_prompt = f.read()

text_model = genai.GenerativeModel("gemini-1.5-pro")
vision_model = genai.GenerativeModel("gemini-1.5-pro")

st.subheader("📝 Patient Information")
patient_name = st.text_input("Patient Name")
age = st.number_input("Age", min_value=0, max_value=120, step=1)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
symptoms = st.text_area("Reported Symptoms")
duration = st.text_input("Duration of Symptoms (e.g., 2 weeks)")
pre_existing = st.text_area("Pre-existing Conditions (if any)")
medications = st.text_area("Current Medications (if any)")

user_context = f"""
Patient Name: {patient_name or "N/A"}
Age: {age}
Gender: {gender}
Reported Symptoms: {symptoms or "N/A"}
Duration: {duration or "N/A"}
Pre-existing Conditions: {pre_existing or "N/A"}
Current Medications: {medications or "N/A"}
"""

st.subheader("📎 Upload Medical Image (Optional)")
uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

if st.button("Submit"):
    try:
        if uploaded_file:
            mime_type, _ = mimetypes.guess_type(uploaded_file.name)
            if mime_type and mime_type.startswith("image/"):
                image_data = uploaded_file.getvalue()
                response = vision_model.generate_content([
                    system_prompt,
                    user_context,
                    {"mime_type": mime_type, "data": image_data}
                ])
            else:
                st.error("❌ Invalid image format. Please upload JPG or PNG.")
                st.stop()
        else:
            response = text_model.generate_content([
                system_prompt,
                user_context
            ])

        if response:
            st.subheader("📋 Diagnosis & Recommendations")
            st.success("✅ Analysis Completed")
            st.markdown(response.text)
        else:
            st.warning("⚠️ No response from the model.")

    except Exception as e:
        if "ResourceExhausted" in str(e) or "429" in str(e):
            st.error("🚫 API quota exceeded. Please wait or upgrade your plan.")
            st.info("See: https://ai.google.dev/gemini-api/docs/rate-limits")
        else:
            st.error("⚠️ An unexpected error occurred.")
            st.exception(e)
