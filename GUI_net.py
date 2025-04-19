import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Layer
from vit_keras import vit
from keras.applications.densenet import preprocess_input as densenet_preprocess_input
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_preprocess_input
from keras.applications.resnet import preprocess_input as resnet_preprocess_input
from keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from keras.applications.convnext import preprocess_input as convnext_preprocess_input

class LayerScale(tf.keras.layers.Layer):
    def __init__(self, init_values=1e-6, projection_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        dim = self.projection_dim or input_shape[-1]
        self.gamma = self.add_weight(
            name="gamma",
            shape=(dim,),
            initializer=tf.keras.initializers.Constant(self.init_values),
            trainable=True,
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update({
            "init_values": self.init_values,
            "projection_dim": self.projection_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# 1. 预处理函数
def vit_preprocess_input(img_array):
    return img_array / 255.0  # ViT 需要归一化到 [0,1]

def preprocess_input_image(img_array, model_name):
    # 标准化模型名：统一小写 + 去空格 + 下划线替换空格
    model_key = model_name.strip().lower().replace(" ", "_")

    if model_key == "densenet121":
        return densenet_preprocess_input(img_array)
    elif model_key == "vgg16":
        return vgg16_preprocess_input(img_array)
    elif model_key == "mobilenetv3large":
        return mobilenet_v3_preprocess_input(img_array)
    elif model_key == "resnet101":
        return resnet_preprocess_input(img_array)
    elif model_key == "inceptionv3":
        return inception_preprocess_input(img_array)
    elif model_key in ['convnext_small', 'convnext_base', 'convnext_large']:
        return convnext_preprocess_input(img_array)
    elif model_key in ['vit_b16', 'vit_b32', 'vit_l16', 'vit_l32']:
        return vit_preprocess_input(img_array)
    else:
        raise ValueError(f"Invalid model name! Got '{model_name}'")

# 2. 模型路径
MODEL_PATHS = {
    "densenet121": "./models/final_model_densenet121.h5",
    "vgg16": "./models/final_model_vgg16.h5",
    "mobilenetv3large": "./models/final_model_mobilenetV3Large.h5",
    "resnet101": "./models/final_model_resnet101.h5",
    "inceptionv3": "./models/final_model_inceptionV3.h5",
    "convnext_small": "./models/final_model_convnext_small.h5",
    "convnext_base": "./models/final_model_convnext_base.h5",
    "convnext_large": "./models/final_model_convnext_large.h5",
    "vit_b16": "./models/final_model_vit_b16.h5",
    "vit_b32": "./models/final_model_vit_b32.h5",
    "vit_l16": "./models/final_model_vit_l16.h5",
    "vit_l32": "./models/final_model_vit_l32.h5",
}

# 15种疾病标签
LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
          'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 
          'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

BEST_THRESHOLDS = {
    "vgg16": [0.291, 0.176, 0.213, 0.199, 0.312, 0.156, 0.144, 0.078, 0.347, 0.196, 0.395, 0.189, 0.168, 0.144, 0.193],
    "resnet101": [0.238, 0.151, 0.177, 0.146, 0.278, 0.140, 0.123, 0.072, 0.298, 0.192, 0.370, 0.186, 0.160, 0.135, 0.177],
    "inceptionv3": [0.244, 0.148, 0.188, 0.158, 0.275, 0.158, 0.114, 0.057, 0.323, 0.185, 0.376, 0.190, 0.165, 0.131, 0.178],
    "densenet121": [0.245, 0.145, 0.184, 0.152, 0.269, 0.141, 0.113, 0.073, 0.308, 0.181, 0.373, 0.176, 0.162, 0.125, 0.177],
    "mobilenetv3large": [0.229, 0.139, 0.173, 0.145, 0.278, 0.140, 0.114, 0.064, 0.310, 0.182, 0.394, 0.178, 0.161, 0.116, 0.176],
    "convnext_small": [0.267, 0.132, 0.211, 0.169, 0.300, 0.140, 0.113, 0.076, 0.352, 0.215, 0.316, 0.174, 0.162, 0.157, 0.190],
    "convnext_base": [0.238, 0.147, 0.196, 0.174, 0.276, 0.116, 0.137, 0.033, 0.320, 0.197, 0.357, 0.158, 0.144, 0.108, 0.148],
    "convnext_large": [0.223, 0.126, 0.209, 0.130, 0.210, 0.065, 0.056, 0.074, 0.311, 0.201, 0.373, 0.173, 0.145, 0.141, 0.113],
    "vit_b16": [0.253, 0.157, 0.215, 0.241, 0.269, 0.144, 0.117, 0.067, 0.375, 0.192, 0.330, 0.207, 0.150, 0.172, 0.181],
    "vit_b32": [0.221, 0.163, 0.197, 0.197, 0.271, 0.119, 0.132, 0.069, 0.330, 0.154, 0.375, 0.194, 0.145, 0.146, 0.177],
    "vit_l16": [0.244, 0.160, 0.207, 0.170, 0.311, 0.099, 0.111, 0.064, 0.340, 0.182, 0.339, 0.179, 0.143, 0.137, 0.128],
    "vit_l32": [0.217, 0.161, 0.184, 0.210, 0.280, 0.116, 0.133, 0.064, 0.336, 0.153, 0.371, 0.204, 0.138, 0.127, 0.149],
}


# 3. 加载模型
@st.cache_resource()
def load_selected_model(model_name):
    """加载 CNN 或 ViT 模型"""
    model_path = MODEL_PATHS[model_name]
    # 直接加载完整模型
    model = load_model(model_path, custom_objects={
        "SigmoidFocalCrossEntropy": tfa.losses.SigmoidFocalCrossEntropy, 
        "LayerScale": LayerScale
    })
    return model

# 4. 计算 Grad-CAM++
def get_last_conv_layer(model_name, model):
    layer_map = {
        "densenet121": "conv5_block16_2_conv",
        "vgg16": "block5_conv3",
        "mobilenetv3large": "expanded_conv_14/project",
        "resnet101": "conv5_block3_out",
        "inceptionv3": "mixed10",
        "convnext_small": "convnext_small_stage_3_block_2_pointwise_conv_2",
        "convnext_base": "convnext_base_stage_3_block_2_pointwise_conv_2",
        "convnext_large": "convnext_large_stage_3_block_2_pointwise_conv_2",
    }
    # 先检查 layer_map 是否有手动设置的层
    if model_name in layer_map:
        return layer_map[model_name]
    # 如果找不到，自动查找最后的 Conv2D 层
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name  # 返回最后一个 Conv2D 层的名字
    return None  # 如果仍然找不到，则返回 None

def grad_cam_plus(model, img, model_name):
    # 修复输入维度问题
    if len(img.shape) == 3:  
        img_tensor = np.expand_dims(img, axis=0)
    elif len(img.shape) == 4 and img.shape[0] == 1:  
        img_tensor = img
    else:
        raise ValueError(f"Unexpected input shape: {img.shape}")
    
    # 计算 Grad-CAM++ 的梯度
    layer_name = get_last_conv_layer(model_name, model)
    conv_layer = model.get_layer(layer_name)
    # 构建模型
    heatmap_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])
    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                conv_output, predictions = heatmap_model(img_tensor)
                category_id = np.argmax(predictions[0])
                output = predictions[:, category_id]
                conv_first_grad = gtape3.gradient(output, conv_output)
            conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
        conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

    # 计算 Grad-CAM++ 权重
    if conv_first_grad is None or conv_second_grad is None or conv_third_grad is None:
        st.warning("Gradient computation failed. Check conv_output tracking!")
        return None
    global_sum = np.sum(conv_output, axis=(0, 1, 2))
    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)  # 避免除 0 错误
    alphas = alpha_num / alpha_denom
    alpha_normalization_constant = np.sum(alphas, axis=(0, 1))
    alphas /= alpha_normalization_constant
    weights = np.maximum(conv_first_grad[0], 0.0)
    deep_linearization_weights = np.sum(weights * alphas, axis=(0, 1))

    # 生成 Grad-CAM++ 热力图
    grad_cam_map = np.sum(deep_linearization_weights * conv_output[0], axis=-1)
    heatmap = np.maximum(grad_cam_map, 0)
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)  # 处理 NaN 和 Inf
    heatmap = cv2.normalize(heatmap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)  # 归一化 0-1
    heatmap = np.uint8(255 * heatmap)  # 转换为 0-255 格式
    return heatmap

# 5. Transformer Attribution 计算
def transformer_attribution(model, img_array, target_class=None):
    # 提取 ViT 层
    vit_layer = None
    for layer in model.layers:
        if "vit" in layer.name.lower():
            vit_layer = layer
            break
    if vit_layer is None:
        st.warning("No ViT layer found! Check model.summary().")
        return None
    vit_model = model.get_layer(vit_layer.name)

    # 提取 Transformer Encoder Block 的注意力权重
    attention_maps = []
    for layer in vit_model.layers:
        if "encoderblock" in layer.name.lower():
            attention_maps.append(layer.output[1])
    if not attention_maps:
        st.warning("No Transformer Encoder Blocks found! Check model.summary().")
        return None

    # 计算 Attention Rollout
    get_attention_maps = tf.keras.backend.function([vit_model.input], attention_maps)
    attention_map_list = get_attention_maps([img_array])
    rollout_attention = np.eye(attention_map_list[0].shape[-1])
    for attention in attention_map_list:
        attention = np.mean(attention, axis=1)
        attention = attention / attention.sum(axis=-1, keepdims=True)
        rollout_attention = np.matmul(attention, rollout_attention)

    #  计算梯度归因 (Gradient Attribution)
    inputs = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        preds = model(inputs)
        if target_class is None:
            target_class = np.argmax(preds[0])
        target_score = preds[:, target_class]
    grads = tape.gradient(target_score, inputs)
    grads_scalar = tf.reduce_mean(grads).numpy()

    # 结合梯度和注意力，生成最终热力图
    attention_cls = rollout_attention[:, 0, 1:]
    attention_cls = np.mean(attention_cls, axis=0)
    attention_cls *= grads_scalar
    num_patches = attention_cls.shape[0]
    patch_grid_mapping = {196: 14, 49: 7, 256: 16}
    if num_patches in patch_grid_mapping:
        patch_grid_size = patch_grid_mapping[num_patches]
    else:
        st.error(f"Unsupported Patch Grid Size: {num_patches} patches")
        return None
    attention_cls = attention_cls.reshape((patch_grid_size, patch_grid_size))
    attention_cls = cv2.resize(attention_cls, (224, 224))
    attention_cls = (attention_cls - np.min(attention_cls)) / (np.max(attention_cls) - np.min(attention_cls))
    return attention_cls

# 6. 叠加热力图
def overlay_heatmap(uploaded_file, heatmap, alpha=0.4, box=True):
    """叠加热力图并绘制疾病高激活区域的 Bounding Box"""
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Error loading image! Please upload a valid X-ray image.")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    normalized_heatmap = np.uint8(255 * heatmap)
    color_heatmap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(img, 1 - alpha, color_heatmap, alpha, 0)

    if box:
        # ➤ 阈值提取高响应区域
        _, binary_map = cv2.threshold(normalized_heatmap, 180, 255, cv2.THRESH_BINARY)
        binary_map = cv2.GaussianBlur(binary_map, (3, 3), 0)
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ➤ 在原图上画 Bounding Box
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(overlayed, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 蓝色矩形框

    return overlayed

# 7. Streamlit UI
st.title("Chest X-ray Disease Classification & Visualization")
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox("Choose a model:", list(MODEL_PATHS.keys()))
model = load_selected_model(selected_model)
st.sidebar.success(f"Loaded model: {selected_model}")

st.sidebar.subheader("Adjust Thresholds")
selected_thresholds = []
default_thresholds = BEST_THRESHOLDS.get(selected_model, [0.2]*15)

for i, label in enumerate(LABELS):
    threshold = st.sidebar.slider(
        f"{label}", 
        min_value=0.0, 
        max_value=1.0, 
        value=float(default_thresholds[i]), 
        step=0.001,
        key=f"{selected_model}_{label}_threshold"
    )
    selected_thresholds.append(threshold)

uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    uploaded_file.seek(0)
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input_image(img_array, selected_model)
    predictions = model.predict(img_array)[0]
    st.image(img, caption="Uploaded Chest X-ray", use_column_width=True)

    st.subheader("Diseases Predicted as Positive")
    positive_labels = []
    for i, label in enumerate(LABELS):
        if predictions[i] >= selected_thresholds[i]:
            positive_labels.append((label, predictions[i]))

    if positive_labels:
        for disease, prob in positive_labels:
            st.write(f"- {disease}: **{prob:.4f}**")
    else:
        st.info("No diseases predicted positive with current thresholds.")

    if "vit" in selected_model:
        st.subheader("Visualization")
        heatmap = transformer_attribution(model, img_array)
    else:
        st.subheader("Visualization")
        heatmap = grad_cam_plus(model, img_array, selected_model)

    if heatmap is not None:
        overlayed_img = overlay_heatmap(uploaded_file, heatmap)
        st.image(overlayed_img, caption="Heatmap with Disease Bounding Boxes", use_column_width=True)
