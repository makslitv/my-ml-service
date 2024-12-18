# scripts/train.py
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms, models
import numpy as np

def main():
    # Параметры
    batch_size = 64
    lr = 0.001
    epochs = 2
    val_ratio = 0.2
    random_seed = 42

    # Установка воспроизводимости
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Преобразования для изображений: меняем размер, переводим в 3 канала, нормализация
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Загрузка датасета MNIST (train=True включает тренировочные данные)
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Загрузка предобученной модели ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Заменяем последний слой на классификатор для 10 классов (цифры 0-9)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Устанавливаем эксперимент MLflow (создаст если нет)
    mlflow.set_experiment("mnist_resnet_exp")

    with mlflow.start_run():
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("epochs", epochs)

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = correct / total
            avg_train_loss = train_loss / total

            # Валидация
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            val_acc = correct_val / total_val
            avg_val_loss = val_loss / total_val

            print(f"Epoch [{epoch+1}/{epochs}]: "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Логирование метрик в MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

        # Сохраняем финальную модель в MLflow
        mlflow.pytorch.log_model(model, artifact_path="model")

        print("Training finished. Model logged to MLflow.")

if __name__ == "__main__":
    main()

