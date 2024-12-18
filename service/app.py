import json
import torch
import mlflow.pytorch
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

model_path = "../models/model"
model = mlflow.pytorch.load_model(model_path)
model.eval()

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    image_data = data.get("image", None)

    if image_data is None or len(image_data) != 784:
        return jsonify({"error": "Invalid input. 'image' should be a list of length 784."}), 400

    arr = np.array(image_data, dtype=np.float32).reshape(28, 28)
    pil_img = Image.fromarray((arr*255).astype(np.uint8), mode='L')  # 'L' – grayscale

    img_tensor = transform(pil_img).unsqueeze(0)  # shape: [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_class = predicted.item()

    return jsonify({"prediction": pred_class})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
