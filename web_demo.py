"""
猫狗分类网页测试平台
运行: python web_demo.py
本机浏览器访问: http://127.0.0.1:7860
同一局域网手机访问: http://你的电脑IP:7860
支持：上传图片、摄像头拍照、输出类别+置信度
"""
import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
from torchvision import models, transforms

CLASSES = {0: "🐱 猫", 1: "🐶 狗"}

# 加载训练好的权重 best_model.pth
def load_infer_model(weight_path="best_model.pth"):
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(torch.load(weight_path, map_location="cpu"))
    net.eval()
    return net

# 图像预处理（必须和训练保持一致）
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

net = load_infer_model()

def predict_func(pil_image: Image.Image):
    if pil_image is None:
        return "请上传图片或者使用摄像头拍照", {}
    img = pil_image.convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        out = net(tensor)
        probs = torch.softmax(out, dim=1)[0]
    pred_idx = int(probs.argmax())
    conf = float(probs[pred_idx])
    result_text = f"识别结果：{CLASSES[pred_idx]}，置信度 {conf:.2%}"
    prob_dict = {CLASSES[0]: float(probs[0]), CLASSES[1]: float(probs[1])}
    return result_text, prob_dict

# 搭建网页UI
demo = gr.Interface(
    fn=predict_func,
    inputs=gr.Image(type="pil", sources=["upload", "webcam"], label="📷上传图片 / 摄像头拍照"),
    outputs=[
        gr.Textbox(label="🔍识别结果"),
        gr.Label(label="📊类别概率分布")
    ],
    title="🐱🐶猫狗图像分类测试平台",
    description="实验项目：基于ResNet18迁移学习猫狗二分类；支持上传图片、摄像头实时拍照测试。",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    # server_name=0.0.0.0 允许局域网其他设备（手机）访问
    demo.launch(server_name="0.0.0.0", server_port=7860)
