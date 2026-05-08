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
from streamlit_mic_recorder import mic_recorder # Required: pip install streamlit-mic-recorder

# ======================
# PAGE CONFIGURATION
# ======================
st.set_page_config(page_title="Currency AI Pro", layout="centered")

# ======================
# CUSTOM UI DESIGN (CSS)
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
/* Style for voice recorder container */
.voice-ctrl {
    background: rgba(255, 255, 255, 0.1);
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD PRE-TRAINED MODELS
# ======================
@st.cache_resource
def load_models():
    model = joblib.load("svm_currency_model.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    return model, scaler, pca

model, scaler, pca = load_models()

# ======================
# CORE IMAGE PROCESSING FUNCTIONS
# ======================

def auto_crop_currency(img_bgr):
    """
    Detects the currency note using edges and contours, then crops it.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        if w > 50 and h > 50:
            return img_bgr[y:y+h, x:x+w]
    return img_bgr

def extract_features(img_bgr):
    """
    Extracts HOG (Texture) and HSV Histograms (Color) features.
    """
    img_resized = cv2.resize(img_bgr, (128, 128))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, orientations=12, pixels_per_cell=(8, 8), 
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    color_feat = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()])
    return np.concatenate([hog_feat, color_feat])

def speak(text):
    """
    Converts text to speech and plays it automatically in the browser.
    """
    tts = gTTS(text=text, lang='en')
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    audio_base64 = base64.b64encode(audio_bytes.read()).decode()
    audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

def convert_to_sentence(label):
    """
    Converts internal labels like 'Egyptian_100' to 'One Hundred Egyptian Pounds'.
    """
    num_words = {"1": "One", "2": "Two", "5": "Five", "10": "Ten", "20": "Twenty", 
                 "50": "Fifty", "100": "One Hundred", "200": "Two Hundred", "500": "Five Hundred"}
    parts = label.split("_")
    if len(parts) >= 2:
        country, value = parts[0], parts[1]
        currency_name = {"Egyptian": "Egyptian Pounds", "Dollar": "US Dollars", "Indian": "Indian Rupees"}.get(country, country)
        return f"{num_words.get(value, value)} {currency_name}"
    return label

# ======================
# VOICE COMMAND LOGIC (JavaScript Integration)
# ======================
# This JS script listens for "Camera" or "Upload" and updates the radio button selection
st.components.v1.html("""
    <script>
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.continuous = true;
    recognition.interimResults = false;

    recognition.onresult = (event) => {
        const command = event.results[event.results.length - 1][0].transcript.toLowerCase();
        if (command.includes('camera')) {
            window.parent.postMessage({type: 'voice_command', value: '📷 Camera'}, '*');
        } else if (command.includes('upload')) {
            window.parent.postMessage({type: 'voice_command', value: '📂 Upload Image'}, '*');
        }
    };
    recognition.start();
    </script>
""", height=0)

# ======================
# MAIN STREAMLIT APP UI
# ======================
st.markdown("<div class='title'>💰 Currency Recognition AI</div>", unsafe_allow_html=True)

# Accessibility Voice Guide
if 'first_load' not in st.session_state:
    speak("Voice control active. Say Camera or Upload to select method.")
    st.session_state.first_load = True

# Radio selection logic (Modified to listen to voice)
if 'method_choice' not in st.session_state:
    st.session_state.method_choice = "📂 Upload Image"

option = st.radio("Choose input method:", ["📂 Upload Image", "📷 Camera"], key="method_radio")

raw_img = None

# Input selection logic
if option == "📂 Upload Image":
    file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    if file:
        raw_img = np.array(Image.open(file))

elif option == "📷 Camera":
    camera = st.camera_input("Take a picture")
    if camera:
        raw_img = np.array(Image.open(camera))

# ======================
# AUTO-EXECUTION LOGIC
# ======================
if raw_img is not None:
    st.image(raw_img, caption="Input Image", use_container_width=True)
    
    with st.spinner("Analyzing..."):
        # 1. Convert to BGR for OpenCV
        img_bgr = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
        
        # 2. Apply Automatic Cropping
        processed_img = auto_crop_currency(img_bgr)
        
        # 3. Show detected area
        with st.expander("Show Detected Currency Area"):
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))

        # 4. Feature Extraction and Prediction
        features = extract_features(processed_img).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        
        prediction = model.predict(features_pca)[0]
        result_text = convert_to_sentence(prediction)

        # 5. Display Result and Play Audio
        st.markdown(f"### 🎯 Prediction: **{result_text}**")
        speak(result_text)
        st.success("Recognition Successful!")