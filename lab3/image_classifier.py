import random
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os


class ConvolutionalNetwork(nn.Module):
    def __init__(self, n_classes, image_resize, channels=3, n1=5, n2=10, n3=20, n_inside=512):
        super(ConvolutionalNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=n1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n2)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=n2, out_channels=n3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(n3)
        self.act3 = nn.ReLU()

        self.flat_size = n3 * (image_resize // 4) ** 2
        self.fc1 = nn.Linear(self.flat_size, n_inside)
        self.fc1_act = nn.ReLU()
        self.fc2 = nn.Linear(n_inside, n_classes)


class DatasetManager:
    def __init__(self, image_folder_path, image_resize=(64, 64), absolute_path=False,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        if not absolute_path:
            image_folder_path = os.path.join(os.getcwd(), image_folder_path)
        self.image_folder_path = image_folder_path
        self.image_resize = image_resize
        self.mean = mean
        self.std = std
        self.classes = None

    def create_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.image_resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def load_dataset(self):
        transform = self.create_transforms()
        dataset = ImageFolder(self.image_folder_path, transform=transform)
        self.classes = dataset.classes
        return dataset

    def create_dataloaders(self, batch_size=64, test_size=0.2, validation_size=0.05, shuffle_test=True):
        dataset = self.load_dataset()
        total = len(dataset)
        test_len = int(test_size * total)
        train_len = total - test_len
        val_len = int(validation_size * train_len)
        train_len = train_len - val_len

        train_ds, test_ds, val_ds = random_split(dataset, [train_len, test_len, val_len])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle_test)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, val_loader

class ImageClassifier:
    def __init__(self, model=None, n_classes=None, classes=None, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
        elif n_classes is not None:
            self.n_classes = n_classes
            self.classes = [f"class_{i}" for i in range(n_classes)]
        else:
            raise ValueError("Необходимо указать либо `classes`, либо `n_classes`.")

        if model is None:
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.fc = nn.Linear(self.model.fc.in_features, self.n_classes)
        else:
            self.model = model

        self.model.to(self.device)

        self.train_loader = self.test_loader = self.validation_loader = None

    def set_dataloaders(self, train_loader, test_loader, validation_loader=None):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.validation_loader = validation_loader

    def fit(self, epochs=10, learning_rate=0.001, validation=True):
        if self.train_loader is None:
            raise RuntimeError("Не задан train_loader. Используйте set_dataloaders().")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        self.model.train()
        print(f"Обучение модели на {self.device}")

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f"Эпоха: {epoch+1}, Функция потерь: {avg_loss:.4f}")

            if validation and self.validation_loader is not None:
                train_acc = self._accuracy(self.train_loader, max_batches=50)
                val_acc = self._accuracy(self.validation_loader)
                print(f"Точность на обучении: {train_acc:.4f}, на валидации: {val_acc:.4f}")
                self.model.train()

    def _accuracy(self, loader, max_batches=None):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(loader):
                if max_batches and i >= max_batches:
                    break
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return correct / total

    def evaluate_metrics(self, loader=None):
        if loader is None:
            loader = self.test_loader
            if loader is None:
                raise RuntimeError("Не задан test_loader.")

        self.model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Метрики
        print("\n=== Classification Report ===")
        print(classification_report(all_labels, all_preds, target_names=self.classes))

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, xticks_rotation=45, cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def plot_images(self, images, labels, n=4, size_per_img=3):
        if n > len(images):
            n = len(images)
        figsize = (n * size_per_img, size_per_img)
        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        for i in range(n):
            img = images[i].cpu()
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            img = img.numpy().transpose((1, 2, 0))
            axes[i].imshow(img)
            axes[i].set_title(f"True: {self.classes[labels[i]]}")
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()


    def predict_and_visualize(self, loader=None, n=4, pool_size=100):
        if loader is None:
            loader = self.test_loader
            if loader is None:
                raise RuntimeError("Не задан test_loader.")

        self.model.eval()

        all_images = []
        all_labels = []

        for images, labels in loader:
            all_images.append(images)
            all_labels.append(labels)
            if len(torch.cat(all_images)) >= pool_size:
                break

        if not all_images:
            raise ValueError("Нет данных в loader.")

        all_images = torch.cat(all_images)
        all_labels = torch.cat(all_labels)

        total = len(all_images)
        if n > total:
            n = total
        indices = random.sample(range(total), n)

        images = all_images[indices].to(self.device)
        labels = all_labels[indices].cpu()

        with torch.no_grad():
            outputs = self.model(images)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu()

        # Визуализация
        self.plot_images(images.cpu(), labels, n=n)

        print("Предсказания:")
        for i in range(n):
            true_label = self.classes[labels[i].item()]
            pred_label = self.classes[preds[i].item()]
            print(f"  [{i + 1}] Истинный: {true_label}, Предсказанный: {pred_label}")

    def save(self, path='model.pt'):
        torch.save(self.model.state_dict(), path)

    def load(self, path='model.pt'):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)