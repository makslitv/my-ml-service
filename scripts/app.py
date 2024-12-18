import json
import torch
import mlflow.pytorch
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

# Загрузка модели из MLflow (предполагается, что модель сохранена локально)
# Укажите путь к модели, которую залогировали в train.py:
# Например, после обучения вы можете выполнить:
# mlflow artifacts download -r <run_id> -d ./models
# где <run_id> - id последнего запуска
# Внутри ./models будет папка model с PyTorch моделью.
model_path = "models/model"  # путь к модели, adjust по вашему
model = mlflow.pytorch.load_model(model_path)
model.eval()

# Преобразования как в train.py
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    # ожидаем json: { "image": [0.0, 0.1, ... 784 значений ...] }
    data = request.get_json(force=True)
    image_data = data.get("image", None)

    if image_data is None or len(image_data) != 784:
        return jsonify({"error": "Invalid input. 'image' should be a list of length 784."}), 400

    # Превращаем список в PIL изображение 28x28
    # image_data – список float пикселей (0..1). Преобразуем в np.array, затем PIL
    arr = np.array(image_data, dtype=np.float32).reshape(28, 28)
    pil_img = Image.fromarray((arr*255).astype(np.uint8), mode='L')  # 'L' – grayscale

    # Применяем transform
    img_tensor = transform(pil_img).unsqueeze(0)  # shape: [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_class = predicted.item()

    return jsonify({"prediction": pred_class})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
