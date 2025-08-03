import streamlit as st
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import DenseNet121, InceptionV3, MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.regularizers import l2
import zipfile
import glob

# --- CONFIG ---
st.set_page_config(page_title="Leukemia Subtype Classifier", layout="wide")
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Benign', 'Early-Precursor', 'Precursor', 'Pro-lymphoblast']
NUM_CLASSES = 4

# --- UNZIP SPLIT FILES ---
if not os.path.exists("savedmodels2"):
    st.warning("📦 Extracting model files... Please wait.")
    split_files = sorted(glob.glob("savedmodels2_2.zip.*"))
    with open("savedmodels2_2.zip", "wb") as f_out:
        for part in split_files:
            with open(part, "rb") as f_in:
                f_out.write(f_in.read())
    with zipfile.ZipFile("savedmodels2_2.zip", 'r') as zip_ref:
        zip_ref.extractall("savedmodels2")
    st.success("✅ Models extracted!")

SAVE_DIR = "savedmodels2"
MODEL_CONFIGS = {
    "DenseNet121": DenseNet121,
    "InceptionV3": InceptionV3,
    "MobileNetV2": MobileNetV2
}

# --- FUNCTIONS ---
def load_model_weights(model_name, base_fn):
    base = base_fn(include_top=False, weights=None, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=output, name=model_name)
    weight_path = os.path.join(SAVE_DIR, model_name, f"{model_name}.weights.h5")
    model.load_weights(weight_path)
    return model

def preprocess_image(uploaded_file):
    img = keras_image.load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array, np.array(img)

def get_gradcam(model, img_array):
    pred_index = np.argmax(model.predict(img_array)[0])
    last_conv = next(layer.name for layer in reversed(model.layers) if 'conv' in layer.name.lower())
    grad_model = Model([model.inputs], [model.get_layer(last_conv).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), IMG_SIZE)
    return np.uint8(255 * heatmap)

def overlay_heatmap(heatmap, image):
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image = cv2.resize(np.array(image), IMG_SIZE)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlay

def ensemble_prediction(models, img_array):
    preds = [model.predict(img_array)[0] for model in models.values()]
    avg_pred = np.mean(preds, axis=0)
    return avg_pred

# --- SIDEBAR & LOAD MODELS ---
st.sidebar.title("🧠 Leukemia Predictor")
with st.sidebar:
    st.markdown("This app classifies microscopic images into leukemia subtypes using 3 pre-trained models.")
    st.markdown("---")
    st.markdown("📦 Loading models...")
    models = {}
    for name in MODEL_CONFIGS:
        model = load_model_weights(name, MODEL_CONFIGS[name])
        models[name] = model
    st.success("Models loaded.")

# --- MAIN APP ---
st.title("🩸 Leukemia Subtype Classifier with Grad-CAM")
st.markdown("Upload a microscopic blood cell image to get predictions from AI models and compare them to how doctors diagnose.")

uploaded_file = st.file_uploader("📤 Upload a blood cell image (jpg/png)", type=["jpg", "jpeg", "png"])
submit_btn = st.button("🔍 Diagnose Image")

if uploaded_file and submit_btn:
    img_array, original_img = preprocess_image(uploaded_file)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(original_img, caption="🖼 Uploaded Image", use_column_width=True)
    with col2:
        st.subheader("📊 Predictions")
        results = {}
        for name, model in models.items():
            pred = model.predict(img_array)[0]
            results[name] = pred
            st.write(f"{name}** → {CLASS_NAMES[np.argmax(pred)]} ({100 * np.max(pred):.2f}%)")

    # Ensemble prediction
    st.subheader("🤝 Ensemble Prediction")
    final_pred = ensemble_prediction(models, img_array)
    st.success(f"*Predicted Subtype:* {CLASS_NAMES[np.argmax(final_pred)]}")
    st.bar_chart(final_pred)

    # Optional Grad-CAM
    if st.checkbox("🧠 Show Grad-CAM Visualizations"):
        st.subheader("Grad-CAM (Model Focus Regions)")
        for name, model in models.items():
            heatmap = get_gradcam(model, img_array)
            overlay = overlay_heatmap(heatmap, original_img)
            st.image(overlay, caption=f"Grad-CAM for {name}", use_column_width=True)
        st.markdown("### 📌 Ensemble Grad-CAM")
        heatmaps = [get_gradcam(model, img_array) for model in models.values()]
        avg_heatmap = np.uint8(np.mean(heatmaps, axis=0))
        overlay = overlay_heatmap(avg_heatmap, original_img)
        st.image(overlay, caption="Ensemble Grad-CAM", use_column_width=True)

    # Doctor Panel
    st.markdown("---")
    st.subheader("🧬 How Doctors Diagnose")
    with st.expander("🔍 Key visual markers doctors look for"):
        st.markdown("""
        - *Nucleus-to-Cytoplasm Ratio:* Higher in leukemia cells.
        - *Irregular Nucleus Shape:* More common in abnormal blasts.
        - *Granules & Auer Rods:* Important in differentiating AML.
        - *Chromatin Texture:* Fine and immature in malignant cells.
        """)
        st.image("https://i.imgur.com/TVf3f5L.png", caption="Example of nucleus differences", use_column_width=True)

else:
    st.info("📂 Please upload an image and click 'Diagnose Image' to begin.")gi