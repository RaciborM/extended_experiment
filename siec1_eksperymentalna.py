import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.patches as patches

class PairDataset(Dataset):
    def __init__(self, image_dataset):
        if isinstance(image_dataset, torch.utils.data.Subset):
            self.data = image_dataset.dataset
            self.indices = image_dataset.indices
        else:
            self.data = image_dataset
            self.indices = list(range(len(image_dataset)))

        self.class_to_indices = self._create_class_index()
        self.classes = list(self.class_to_indices.keys())
        self.transform = self.data.transform
        self.front_images = self._get_front_images()

    def _create_class_index(self):
        class_to_indices = {}
        for idx in self.indices:
            _, label = self.data.samples[idx]
            class_to_indices.setdefault(label, []).append(idx)
        return class_to_indices

    def _get_front_images(self):
        front_images = {}
        for idx in self.indices:
            _, label = self.data.samples[idx]
            if label not in front_images:
                front_images[label] = idx
        return front_images

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx2 = self.indices[index]
        img2, label2 = self.data[idx2]

        idx1 = self.front_images[label2]
        img1, label1 = self.data[idx1]

        if random.random() > 0.5:
            label = 1
        else:
            other_labels = [l for l in self.classes if l != label2]
            random_label = random.choice(other_labels)
            idx1 = self.front_images[random_label]
            img1, label1 = self.data[idx1]
            label = 0

        return img1, img2, torch.tensor(label, dtype=torch.float32)

# neural model with trained resnet-18
class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(num_ftrs * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward_once(self, x):
        return self.cnn(x)

    def forward(self, x1, x2):
        f1 = self.forward_once(x1)
        f2 = self.forward_once(x2)
        combined = torch.cat([f1, f2], dim=1)
        out = self.fc(combined)
        return out

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for x1, x2, y in dataloader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x1, x2).squeeze()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred_scores = []
    images = []

    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            outputs = model(x1, x2).squeeze()
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds.float() == y).sum().item()
            total += y.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred_scores.extend(torch.sigmoid(outputs).cpu().numpy())
            images.extend(zip(x1.cpu(), x2.cpu()))

    return correct / total, y_true, y_pred_scores, images

def load_data(train_val_dir, test_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_val_dataset = ImageFolder(root=train_val_dir, transform=transform)
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_set, val_set = random_split(train_val_dataset, [train_size, val_size])

    test_dataset = ImageFolder(root=test_dir, transform=transform)

    train_dataset = PairDataset(train_set)
    val_dataset = PairDataset(val_set)
    test_dataset = PairDataset(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# loss plot
def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid()
    plt.show()

# metrics plot
def plot_line_chart(thresholds, precision, recall, f1):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision, 'o-', label='Precision')
    plt.plot(thresholds, recall, 's-', label='Recall')
    plt.plot(thresholds, f1, '^-', label='F1-Score')
    plt.xlabel("Significance level (alpha)")
    plt.ylabel("Score")
    plt.title("Significance level vs. Evaluation Metrics")
    plt.legend()
    plt.grid(True)
    plt.show()

# table of results
def print_metric_table(precision, recall, f1, thresholds):
    print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    for i, t in enumerate(thresholds):
        print(f"{t:<10.2f} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f}")
    print(f"\nAverage   {np.mean(precision):<10.4f} {np.mean(recall):<10.4f} {np.mean(f1):<10.4f}")

# bar plot
def plot_bar_chart(precision, recall, f1, label="Ours DB"):
    metrics = ['Precision', 'Recall', 'F1-Score']
    scores = [np.mean(precision), np.mean(recall), np.mean(f1)]
    plt.figure(figsize=(6, 5))
    bars = plt.bar(metrics, scores, color=['skyblue', 'lightgreen', 'salmon'])
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{score:.3f}', 
                 ha='center', va='bottom', fontsize=10)
    plt.ylim(0, 1)
    plt.title(f"Dataset performance for {label}")
    plt.ylabel("Score")
    plt.grid(axis='y')
    plt.show()

# pair visualization
def visualize_pairs(images, scores, y_true, threshold=0.2, max_pairs=10):
    selected = [(img1, img2, score, label) for (img1, img2), score, label in zip(images, scores, y_true)]
    selected.sort(key=lambda x: -x[2])  
    selected = selected[:max_pairs]

    fig, axes = plt.subplots(2, max_pairs, figsize=(2 * max_pairs, 4))

    for idx, (img1, img2, score, label) in enumerate(selected):
        pred = 1 if score >= threshold else 0
        for ax, img in zip([axes[0, idx], axes[1, idx]], [img1, img2]):
            img_np = img.permute(1, 2, 0).numpy()
            ax.imshow(img_np)
            ax.axis('off')
            if pred == 0 and label == 1:
                ax.add_patch(patches.Rectangle((0, 0), 223, 223, linewidth=3, edgecolor='r', facecolor='none'))
        axes[0, idx].set_title(f"S: {score:.2f}\nP: {pred}, T: {int(label)}", fontsize=9)

    plt.suptitle(f"Top {max_pairs} Pairs (Red border = FN @ t={threshold})", fontsize=14)
    plt.tight_layout()
    plt.show()

def run_pipeline(train_val_dir, test_dir, epochs=50, batch_size=32, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = load_data(train_val_dir, test_dir, batch_size)
    model = SimpleNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    thresholds = np.linspace(0.01, 0.4, 10)
    precision_list = np.zeros((len(thresholds),))
    recall_list = np.zeros((len(thresholds),))
    f1_list = np.zeros((len(thresholds),))

    for epoch in range(epochs):
        loss = train(model, train_loader, optimizer, criterion, device)
        acc, y_true, y_pred_scores, images = evaluate(model, val_loader, device)
        train_losses.append(loss)

        for t_idx, t in enumerate(thresholds):
            y_pred = [1 if score > t else 0 for score in y_pred_scores]
            precision_list[t_idx] += precision_score(y_true, y_pred)
            recall_list[t_idx] += recall_score(y_true, y_pred)
            f1_list[t_idx] += f1_score(y_true, y_pred)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Val Acc: {acc:.4f}")

    # avg metrics
    precision_list /= epochs
    recall_list /= epochs
    f1_list /= epochs

    test_acc, y_true, y_pred_scores, images = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")

    # plots
    plot_loss(train_losses)
    plot_line_chart(thresholds, precision_list, recall_list, f1_list)
    print_metric_table(precision_list, recall_list, f1_list, thresholds)
    plot_bar_chart(precision_list, recall_list, f1_list, label="Ours DB")
    visualize_pairs(images, y_pred_scores, y_true, threshold=0.2)

if __name__ == "__main__":
    run_pipeline(
        train_val_dir=r"C:\Users\misie\Desktop\praca_magisterska\programy\pointing_04_ready",
        test_dir=r"C:\Users\misie\Desktop\praca_magisterska\programy\Front",
        epochs=50,
        batch_size=32,
        lr=0.001
    )
