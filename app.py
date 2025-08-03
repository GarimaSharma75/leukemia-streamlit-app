import streamlit as st
import numpy as np
import os
import cv2
import zipfile
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import DenseNet121, InceptionV3, MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.regularizers import l2

# --- CONFIG ---
st.set_page_config(page_title="Leukemia Subtype Detector", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: #e63946;'>🔬 Leukemia Subtype Detection using Ensemble Deep Learning</h1>",
    unsafe_allow_html=True,
)

IMG_SIZE = (224, 224)
CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']
NUM_CLASSES = 4

SAVE_DIR = "savedmodels2"
MODEL_CONFIGS = {
    "DenseNet121": DenseNet121,
    "InceptionV3": InceptionV3,
    "MobileNetV2": MobileNetV2
}

# --- Extract ZIP if necessary ---
if not os.path.exists(SAVE_DIR):
    try:
        with open("savedmodels2_full.zip", "wb") as full_zip:
            for part in ["savedmodels2_2.zip.001", "savedmodels2_2.zip.002"]:
                with open(part, "rb") as pf:
                    full_zip.write(pf.read())
        with zipfile.ZipFile("savedmodels2_full.zip", "r") as zip_ref:
            zip_ref.extractall(SAVE_DIR)
        os.remove("savedmodels2_full.zip")
        st.success("✅ Models extracted successfully.")
    except Exception as e:
        st.error(f"❌ Error extracting models: {e}")

# --- Load Model ---
@st.cache_resource
def load_model_weights(model_name, base_fn):
    base = base_fn(include_top=False, weights=None, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=output, name=model_name)
    weights_path = os.path.join(SAVE_DIR, model_name, f"{model_name}.weights.h5")
    model.load_weights(weights_path)
    return model

def preprocess_image(uploaded_file):
    img = keras_image.load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    return np.expand_dims(img_array, axis=0) / 255.0, np.array(img)

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
    return cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

def ensemble_prediction(models, img_array):
    preds = [model.predict(img_array)[0] for model in models.values()]
    return np.mean(preds, axis=0)

# --- SIDEBAR ---
st.sidebar.header("🩸 Upload Blood Cell Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
show_cam = st.sidebar.checkbox("Show Grad-CAMs", value=True)
analyze = st.sidebar.button("🔍 Run Inference")

# --- Load Models ---
with st.spinner("🔄 Loading all models..."):
    models = {name: load_model_weights(name, fn) for name, fn in MODEL_CONFIGS.items()}

# --- MAIN LOGIC ---
if analyze and uploaded_file:
    img_array, original_img = preprocess_image(uploaded_file)
    st.image(original_img, caption="🖼️ Uploaded Cell Image", use_column_width=False, width=300)

    st.markdown("## 📊 Prediction Results")
    results = {}
    for name, model in models.items():
        pred = model.predict(img_array)[0]
        results[name] = pred
        pred_class = CLASS_NAMES[np.argmax(pred)]
        confidence = 100 * np.max(pred)
        st.markdown(f"""
        <div style='padding:10px; border-radius:8px; background-color:#f1faee; margin-bottom:10px;'>
            <b style='color:#1d3557;'>{name}</b>: <span style='color:#e63946;'>{pred_class}</span>
            <br>Confidence: <b>{confidence:.2f}%</b>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(np.max(pred)))

    st.markdown("---")
    st.subheader("🤝 Ensemble Prediction")
    final_pred = ensemble_prediction(models, img_array)
    final_class = CLASS_NAMES[np.argmax(final_pred)]
    st.success(f"✅ Final Predicted Subtype: **{final_class}**")
    st.bar_chart(final_pred)

    if show_cam:
        st.markdown("---")
        st.subheader("🧠 Grad-CAM Visualizations")
        cols = st.columns(3)
        for i, (name, model) in enumerate(models.items()):
            heatmap = get_gradcam(model, img_array)
            overlay = overlay_heatmap(heatmap, original_img)
            cols[i].image(overlay, caption=f"Grad-CAM: {name}", use_column_width=True)

        st.markdown("### 🤝 Ensemble Grad-CAM")
        heatmaps = [get_gradcam(model, img_array) for model in models.values()]
        avg_heatmap = np.uint8(np.mean(heatmaps, axis=0))
        overlay = overlay_heatmap(avg_heatmap, original_img)
        st.image(overlay, caption="Ensemble Grad-CAM", use_column_width=True)

    st.markdown("---")
    st.subheader("🩺 Compare with Doctor’s Observations")
    st.markdown("""
    <div style='background-color:#f8edeb; padding:20px; border-radius:8px;'>
        <ul>
            <li><b>Nucleus-to-Cytoplasm Ratio:</b> High in leukemic blasts</li>
            <li><b>Granules & Auer Rods:</b> Present in AML cells</li>
            <li><b>Chromatin Texture:</b> Fine & immature in leukemia</li>
            <li><b>Irregular Nuclei:</b> Lobulated or misshaped in T-ALL</li>
            <li><b>Blast Size:</b> Large and inconsistent in malignancy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("📂 Upload an image from sidebar and click **Run Inference** to begin.")
