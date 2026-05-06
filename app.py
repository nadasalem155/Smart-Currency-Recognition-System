import streamlit as st
import cv2
import numpy as np
import joblib
import time
from skimage.feature import hog
from PIL import Image
from gtts import gTTS
from io import BytesIO
import base64

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Currency AI", layout="centered")

# ======================
# UI DESIGN
# ======================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #000000, #0f2027, #203a43, #2c5364);
    color: white;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00ffcc;
    margin-bottom: 25px;
    text-shadow: 2px 2px 10px #00ffcc;
}

.stButton>button {
    background-color: #00ffcc;
    color: black;
    font-weight: bold;
    border-radius: 12px;
    padding: 8px 16px;
    width: 100%;
}

.stRadio label {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODELS
# ======================
# تأكد أن الملفات دي موجودة في نفس الفولدر
model = joblib.load("svm_currency_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# ======================
# TEXT TO SPEECH (FIXED)
# ======================
def speak(text):
    tts = gTTS(text=text, lang='en')
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    
    # تحويل الصوت لبيانات يمكن قراءتها
    final_audio = audio_bytes.read()
    audio_base64 = base64.b64encode(final_audio).decode()

    st.markdown("### 🔊 Voice Output")
    
    # 1. كود HTML للتشغيل التلقائي (يعمل في بعض المتصفحات)
    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

    # 2. المشغل الرسمي (لضمان وجود زر تشغيل يدوي إذا فشل التلقائي)
    st.audio(final_audio, format="audio/mp3")

# ======================
# FEATURE EXTRACTION
# ======================
def extract_features(img):
    img = cv2.resize(img, (128, 128))
    # تحويل الصورة لـ RGB لو كانت مخزنة بصيغة مختلفة لضمان دقة الألوان
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_features = hog(
        gray,
        orientations=12,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )

    hist_b = cv2.calcHist([img], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [32], [0, 256])

    color_features = np.concatenate([
        hist_b.flatten(),
        hist_g.flatten(),
        hist_r.flatten()
    ])

    return np.concatenate([hog_features, color_features])

# ======================
# PREDICT PIPELINE
# ======================
def predict(img):
    feat = extract_features(img)
    feat = np.array(feat).reshape(1, -1)
    feat = scaler.transform(feat)
    feat = pca.transform(feat)
    return model.predict(feat)[0]

# ======================
# LABEL CONVERSION
# ======================
num_words = {
    "1": "One", "2": "Two", "5": "Five", "10": "Ten",
    "20": "Twenty", "50": "Fifty", "100": "One Hundred",
    "200": "Two Hundred", "500": "Five Hundred"
}

def convert_to_sentence(label):
    parts = label.split("_")
    if len(parts) >= 2:
        country = parts[0]
        value = parts[1]
        num_word = num_words.get(value, value)
        
        if country == "Egyptian":
            return f"{num_word} Egyptian Pounds"
        elif country == "Dollar":
            return f"{num_word} US Dollars"
        elif country == "Indian":
            return f"{num_word} Indian Rupees"
    return label

# ======================
# UI MAIN
# ======================
st.markdown("<div class='title'>💰 Currency Recognition AI</div>", unsafe_allow_html=True)

option = st.radio("Choose input method:", ["📂 Upload Image", "📷 Camera"])

img = None

if option == "📂 Upload Image":
    file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    if file:
        img = np.array(Image.open(file))

elif option == "📷 Camera":
    camera = st.camera_input("Take a picture")
    if camera:
        img = np.array(Image.open(camera))

# ======================
# OUTPUT LOGIC
# ======================
if img is not None:
    st.image(img, use_container_width=True)

    if st.button("Predict 🔍"):
        with st.spinner("Analyzing currency..."):
            # استخراج التوقع
            pred = predict(img)
            sentence = convert_to_sentence(pred)
            
            # عرض النتيجة
            st.markdown(f"### 🎯 Prediction: **{sentence}**")
            
            # تشغيل الصوت
            speak(sentence)
            
            st.success("Done ✔️ Prediction completed")