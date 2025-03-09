import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import gdown  # 用于从 Google Drive 下载模型
import os
from keras.models import load_model
from keras.preprocessing import image
from vit_keras import vit
from keras.applications.densenet import preprocess_input as densenet_preprocess_input
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_preprocess_input
from keras.applications.resnet import preprocess_input as resnet_preprocess_input

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        focal_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_factor = (1 - y_pred) ** gamma
        return focal_weight * focal_factor * bce
    return loss

# 🔹 Google Drive 共享模型链接（替换 YOUR_MODEL_ID）
MODEL_DRIVE_LINKS = {
    "DenseNet121": "https://drive.google.com/file/d/1a4-BPbaUn5PwqC1UpSGEPTFDPw400WAG/view?usp=drive_link",
    "VGG16": "https://drive.google.com/file/d/1IQS_UDIrMB6lUKxPuIiPGJN7DEIJqvGE/view?usp=drive_link",
    "MobileNetV3Large": "https://drive.google.com/file/d/1tcc6VreVeP1Vz9gefTOCgmxiFBTbPwas/view?usp=drive_link",
    "ResNet101": "https://drive.google.com/file/d/1YI-Cw6FPPtfDmB-KSrhh-QoE6y9O_oWD/view?usp=drive_link",
    "ViT-B16": "https://drive.google.com/file/d/1YvrE8oJe1jrwScXT1EN2QzfcXPo1n2i8/view?usp=drive_link",
    "ViT-B32": "https://drive.google.com/file/d/19l5KrNrYdeV0juzzvAjRMAz67N18sOt_/view?usp=drive_link",
    "ViT-L16": "https://drive.google.com/file/d/1c6-j4B_Jf2KSTu8IR3aZWKtzF6StzkoP/view?usp=drive_link",
    "ViT-L32": "https://drive.google.com/file/d/1NHz4Wg2zO8N1H_0EIWXPvObdQdKguwFD/view?usp=drive_link",
}

MODEL_SAVE_PATH = "./models"

# 🔹 确保 `models` 目录存在
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# 🔹 下载模型（仅下载一次）
@st.cache_resource()
def download_and_load_model(model_name):
    """从 Google Drive 下载模型并加载"""
    model_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}.h5")

    if not os.path.exists(model_path):
        st.sidebar.info(f"Downloading {model_name} model from Google Drive...")
        gdown.download(MODEL_DRIVE_LINKS[model_name], model_path, quiet=False)

    st.sidebar.success(f"Model {model_name} loaded successfully!")
    model = load_model(model_path, custom_objects={"focal_loss": focal_loss()})
    return model

# 🔹 预处理函数
def vit_preprocess_input(img_array):
    return img_array / 255.0  # ViT 需要归一化到 [0,1]

def preprocess_input_image(img_array, model_name):
    if model_name == "DenseNet121":
        return densenet_preprocess_input(img_array)
    elif model_name == "VGG16":
        return vgg16_preprocess_input(img_array)
    elif model_name == "MobileNetV3Large":
        return mobilenet_v3_preprocess_input(img_array)
    elif model_name == "ResNet101":
        return resnet_preprocess_input(img_array)
    elif model_name in ["ViT-B16", "ViT-B32", "ViT-L16", "ViT-L32"]:
        return vit_preprocess_input(img_array)
    else:
        raise ValueError("Invalid model name!")

# 🔹 Grad-CAM
def compute_gradcam(model, img_array):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        st.warning("No Conv2D layer found in model!")
        return None

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        top_class = np.argmax(predictions[0])
        loss = predictions[:, top_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
    heatmap[heatmap < 0.2] = 0
    heatmap = cv2.normalize(heatmap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    return heatmap

# 🔹 ViT Attention Rollout
def compute_vit_attention_rollout(model, img_array):
    vit_layer = None
    for layer in model.layers:
        if "vit" in layer.name.lower():
            vit_layer = layer
            break

    if vit_layer is None:
        st.warning("No ViT layer found!")
        return None

    vit_model = model.get_layer(vit_layer.name)

    attention_maps = []
    for layer in vit_model.layers:
        if "encoderblock" in layer.name.lower():
            attention_maps.append(layer.output[1])

    if not attention_maps:
        st.warning("No Transformer Encoder Blocks found!")
        return None

    get_attention_maps = tf.keras.backend.function([vit_model.input], attention_maps)
    attention_map_list = get_attention_maps([img_array])
    rollout_attention = np.eye(attention_map_list[0].shape[-1])

    for attention in attention_map_list:
        attention = np.mean(attention, axis=1)
        attention = attention / attention.sum(axis=-1, keepdims=True)
        rollout_attention = np.matmul(attention, rollout_attention)

    attention_cls = rollout_attention[:, 0, 1:197]
    attention_cls = np.mean(attention_cls, axis=0)

    patch_grid_mapping = {196: 14, 49: 7, 256: 16}
    patch_grid_size = patch_grid_mapping.get(attention_cls.shape[0], None)

    if patch_grid_size is None:
        st.error(f"Unsupported Patch Grid Size: {attention_cls.shape[0]} patches")
        return None

    attention_cls = attention_cls.reshape((patch_grid_size, patch_grid_size))
    attention_cls = cv2.resize(attention_cls, (224, 224))
    attention_cls = (attention_cls - np.min(attention_cls)) / (np.max(attention_cls) - np.min(attention_cls))

    return attention_cls

# 🔹 Streamlit UI
st.title("Chest X-ray Disease Classification & Visualization")
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox("Choose a model:", list(MODEL_DRIVE_LINKS.keys()))

model = download_and_load_model(selected_model)
st.sidebar.success(f"Loaded model: {selected_model}")

uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    uploaded_file.seek(0)
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input_image(img_array, selected_model)
    predictions = model.predict(img_array)[0]
    st.image(img, caption="Uploaded Chest X-ray", use_container_width=True)

    st.subheader("Prediction Results")
    LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
              'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
              'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

    for i, label in enumerate(LABELS):
        st.write(f"{label}: **{predictions[i]:.4f}**")
