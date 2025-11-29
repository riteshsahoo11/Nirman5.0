import os
import random
import kagglehub
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
path = "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
train = "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
valid = "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"
test  = "/kaggle/input/new-plant-diseases-dataset/test"
Diseases_classes = os.listdir(train)
def display_disease_samples(data_dir, plants=None, num_cols=5):
    disease_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

    if plants is not None:
        disease_folders = [f for f in disease_folders if any(p in f for p in plants)]

    num_diseases = len(disease_folders)
    num_rows = (num_diseases + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))
    axes = axes.flatten() if num_rows > 1 else axes

    for i, disease_folder in enumerate(disease_folders):
        folder_path = os.path.join(data_dir, disease_folder)

        img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if img_files:
            img_path = os.path.join(folder_path, random.choice(img_files))
            img = Image.open(img_path).convert('RGB')

            disease_name = disease_folder.replace('_', ' ')

            axes[i].imshow(img)
            axes[i].set_title(disease_name, fontsize=12)
            axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

print("🌿 Sample images from different plant disease categories:")
display_disease_samples(test)
Root_dir = path
train_dir = Root_dir + "/train"
valid_dir = Root_dir + "/valid"
test_dir = test
Diseases_classes = os.listdir(train_dir)
print(Fore.GREEN +str(Diseases_classes))
print("\nTotal number of classes are: ", len(Diseases_classes))
plt.figure(figsize=(60,60), dpi=200)
cnt = 0
plant_names = []
tot_images = 0

for i in Diseases_classes:
    cnt += 1
    plant_names.append(i)
    plt.subplot(7,7,cnt)

    image_path = os.listdir(train_dir + "/" + i)
    print(Fore.GREEN)
    print("The Number of Images in " +i+ ":", len(image_path), end= " ")
    tot_images += len(image_path)

    img_show = plt.imread(train_dir + "/" + i + "/" + image_path[0])

    plt.imshow(img_show)
    plt.xlabel(i,fontsize=30)
    plt.xticks([])
    plt.yticks([])


print("\nTotal Number of Images in Directory: ", tot_images)
class PlantDiseaseDataset(Dataset):
    """Custom Dataset for loading plant disease images"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

      
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
import torch
import torch.nn as nn
from torchvision import models

class PlantDiseaseModel(nn.Module):
    """
    EfficientNet-B0 based model for high-accuracy plant disease detection.
    Uses Transfer Learning from ImageNet.
    """
    def __init__(self, num_classes, dropout_rate=0.5):
        super(PlantDiseaseModel, self).__init__()
        
  
        self.base_model = models.efficientnet_b0(weights='DEFAULT')
        
        
        in_features = self.base_model.classifier[1].in_features
        
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


try:
    from torchinfo import summary
    print(summary(PlantDiseaseModel(38), input_size=(1, 3, 224, 224)))
except ImportError:
    print("Install torchinfo for summary")
class EarlyStopping:
    """Early stopping handler to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0.001, save_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save the best model
            torch.save(model.state_dict(), self.save_path)
            print(f"[INFO] Model checkpoint saved to {self.save_path}")
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("[INFO] Early stopping triggered.")
                return True
        return False
