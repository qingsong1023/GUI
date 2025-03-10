import gdown
import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_addons as tfa
from keras.models import load_model
from keras.preprocessing import image
from vit_keras import vit
from keras.applications.densenet import preprocess_input as densenet_preprocess_input
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_preprocess_input
from keras.applications.resnet import preprocess_input as resnet_preprocess_input


# 1. 预处理函数
def vit_preprocess_input(img_array):
    return img_array / 255.0  # ViT 需要归一化到 [0,1]

def preprocess_input_image(img_array, model_name):
    """根据模型选择合适的预处理方法"""
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

# 2. Google Drive 共享模型链接
MODEL_DRIVE_LINKS = {
    "DenseNet121": "https://drive.google.com/uc?id=1a4-BPbaUn5PwqC1UpSGEPTFDPw400WAG",
    "VGG16": "https://drive.google.com/uc?id=1IQS_UDIrMB6lUKxPuIiPGJN7DEIJqvGE",
    "MobileNetV3Large": "https://drive.google.com/uc?id=1tcc6VreVeP1Vz9gefTOCgmxiFBTbPwas",
    "ResNet101": "https://drive.google.com/uc?id=1YI-Cw6FPPtfDmB-KSrhh-QoE6y9O_oWD",
    "ViT-B16": "https://drive.google.com/uc?id=1YvrE8oJe1jrwScXT1EN2QzfcXPo1n2i8",
    "ViT-B32": "https://drive.google.com/uc?id=19l5KrNrYdeV0juzzvAjRMAz67N18sOt_",
    "ViT-L16": "https://drive.google.com/uc?id=1c6-j4B_Jf2KSTu8IR3aZWKtzF6StzkoP",
    "ViT-L32": "https://drive.google.com/uc?id=1NHz4Wg2zO8N1H_0EIWXPvObdQdKguwFD",
}

MODEL_SAVE_PATH = "./models_basic"

# 确保 `models_basic` 目录存在
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# 15种疾病标签
LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
          'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 
          'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

# 下载并加载模型
@st.cache_resource()
def load_selected_model(model_name):
    model_path = os.path.join(MODEL_SAVE_PATH, f"final_model_{model_name.lower()}.h5")
    
    # 如果模型文件不存在，则从 Google Drive 下载
    if not os.path.exists(model_path):
        st.sidebar.info(f"Downloading {model_name} model from Google Drive...")
        try:
            gdown.download(MODEL_DRIVE_LINKS[model_name], model_path, quiet=False)
        except Exception as e:
            st.error(f"Error downloading {model_name}: {e}")
            return None
    
    # 确保文件正确下载
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
        st.error(f"Error: {model_name}.h5 is missing or corrupted. Try re-uploading it to Google Drive.")
        return None
    
    # 加载模型
    model = load_model(model_path, custom_objects={
        "SigmoidFocalCrossEntropy": tfa.losses.SigmoidFocalCrossEntropy
    })
    return model

# 5. Grad-CAM 计算
def compute_gradcam(model, img_array):
    """计算 Grad-CAM 并优化背景影响"""
    
    # 自动获取最后一个 Conv2D 层
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        st.warning("No Conv2D layer found in model!")
        return None

    # 构建 Grad-CAM 计算模型
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    # 计算 Gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        top_class = np.argmax(predictions[0])  # 获取模型预测的最高类别
        loss = predictions[:, top_class]  # 计算损失

    # 计算 Gradients
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 计算 Grad-CAM 热力图
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    # 优化 Grad-CAM
    heatmap = np.maximum(heatmap, 0)  # 只保留正值
    heatmap /= np.max(heatmap)  # 归一化

    # 去除背景高亮，平滑热力图
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)  # 5x5 高斯模糊
    heatmap[heatmap < 0.2] = 0  # 去除低置信度区域（减少背景）
    heatmap = cv2.normalize(heatmap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)  # 归一化 0-1

    return heatmap

# 6. ViT-ReciproCAM 计算
def compute_vit_attention_rollout(model, img_array):
    """使用 Attention Rollout 计算 ViT 热力图"""
    # 获取 ViT 主模型
    vit_layer = None
    for layer in model.layers:
        if "vit" in layer.name.lower():
            vit_layer = layer
            break

    if vit_layer is None:
        st.warning("No ViT layer found! Check model.summary().")
        return None

    vit_model = model.get_layer(vit_layer.name)

    # 收集所有 Transformer Encoder Blocks
    attention_maps = []
    for layer in vit_model.layers:
        if "encoderblock" in layer.name.lower():
            attention_maps.append(layer.output[1])

    if not attention_maps:
        st.warning("No Transformer Encoder Blocks found! Check model.summary().")
        return None

    # 创建 Keras Function 提取所有 Attention Weights
    get_attention_maps = tf.keras.backend.function([vit_model.input], attention_maps)

    # 计算 Attention Map
    attention_map_list = get_attention_maps([img_array])
    rollout_attention = np.eye(attention_map_list[0].shape[-1])  # Identity Matrix

    # 执行 Attention Rollout
    for attention in attention_map_list:
        attention = np.mean(attention, axis=1)  # 平均所有 Heads
        attention = attention / attention.sum(axis=-1, keepdims=True)  # 归一化
        rollout_attention = np.matmul(attention, rollout_attention)  # 传播 Attention

    # 正确提取 CLS Token Attention
    attention_cls = rollout_attention[:, 0, 1:197]  # 取 CLS Token 对所有 Patches 的关注
    attention_cls = np.mean(attention_cls, axis=0)  # 取平均值

    num_patches = attention_cls.shape[0]  # 计算 Patch 数量

    # 直接匹配 ViT 结构
    patch_grid_mapping = {
        196: 14,  # ViT-B16, ViT-L16
        49: 7,    # ViT-B32, ViT-L32
        256: 16,  # 可能的 ViT 变种
    }

    if num_patches in patch_grid_mapping:
        patch_grid_size = patch_grid_mapping[num_patches]
    else:
        st.error(f"Unsupported Patch Grid Size: {num_patches} patches")
        return None

    # 重塑 Attention Map
    attention_cls = attention_cls.reshape((patch_grid_size, patch_grid_size))

    # 上采样到 224x224
    attention_cls = cv2.resize(attention_cls, (224, 224))

    # 归一化
    attention_cls = (attention_cls - np.min(attention_cls)) / (np.max(attention_cls) - np.min(attention_cls))

    return attention_cls

# 7. 叠加热力图
def overlay_heatmap(uploaded_file, heatmap, alpha=0.4):
    """将 Grad-CAM 或 ViT-ReciproCAM 叠加到 X-ray 影像上"""
    
    uploaded_file.seek(0)  # 确保文件读取不会丢失
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # 用 OpenCV 解析字节流

    if img is None:
        st.error("Error loading image! Please upload a valid X-ray image.")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    
    return overlayed_img

# 8. Streamlit UI
st.title("Chest X-ray Disease Classification & Visualization")
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox("Choose a model:", list(MODEL_DRIVE_LINKS.keys()))

model = load_selected_model(selected_model)
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
    for i, label in enumerate(LABELS):
        st.write(f"{label}: **{predictions[i]:.4f}**")

    if "ViT" in selected_model:
        st.subheader("Visualization")
        heatmap = compute_vit_attention_rollout(model, img_array)
    else:
        st.subheader("Visualization")
        heatmap = compute_gradcam(model, img_array)

    if heatmap is not None:
        overlayed_img = overlay_heatmap(uploaded_file, heatmap)
        st.image(overlayed_img, caption="Heatmap", use_container_width=True)
