import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
from sklearn.model_selection import train_test_split
import shutil

# Ustawienia ścieżek do danych
data_dir = 'COVID-19_Radiography_Dataset'
covid_images_dir = os.path.join(data_dir, 'COVID')  # Ścieżka do obrazów COVID
normal_images_dir = os.path.join(data_dir, 'Normal')  # Ścieżka do obrazów Normal

# Przygotowanie katalogów do treningu i walidacji
train_dir = 'train_data'
val_dir = 'val_data'

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# Podział danych na treningowe i walidacyjne (80% dane treningowe, 20% walidacyjne)
for category, category_dir in [('COVID', covid_images_dir), ('Normal', normal_images_dir)]:
    images = os.listdir(category_dir)
    train_images, val_images = train_test_split(images, test_size=0.2)

    # Przenoszenie danych treningowych
    train_category_dir = os.path.join(train_dir, category)
    os.makedirs(train_category_dir, exist_ok=True)
    for img in train_images:
        img_path = os.path.join(category_dir, img)
        if os.path.isfile(img_path):  # Upewnienie się, że to plik, a nie katalog
            shutil.copy(img_path, os.path.join(train_category_dir, img))

    # Przenoszenie danych walidacyjnych
    val_category_dir = os.path.join(val_dir, category)
    os.makedirs(val_category_dir, exist_ok=True)
    for img in val_images:
        img_path = os.path.join(category_dir, img)
        if os.path.isfile(img_path):  # Upewnienie się, że to plik, a nie katalog
            shutil.copy(img_path, os.path.join(val_category_dir, img))

# Sprawdzenie dostępności GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformacje obrazów (zmiana rozmiaru, normalizacja, zamiana na tensor)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Ładowanie danych treningowych i walidacyjnych
image_datasets = {x: datasets.ImageFolder(os.path.join(f'{x}_data'), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Ładowanie pretrenowanego modelu ResNet18
model = models.resnet18(pretrained=True)

# Zamrożenie parametrów (nie trenujemy wcześniejszych warstw ResNet)
for param in model.parameters():
    param.requires_grad = False

# Zmiana ostatniej warstwy na dwie klasy (COVID-19 i Normal)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 klasy: COVID-19 i Normal
model = model.to(device)

# Funkcja kosztu i optymalizator
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Funkcja do trenowania modelu
def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Każda epoka składa się z fazy treningu i walidacji
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Ustawienie modelu w tryb treningowy
            else:
                model.eval()   # Ustawienie modelu w tryb ewaluacyjny (walidacyjny)

            running_loss = 0.0
            running_corrects = 0

            # Iteracja po danych
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Obliczenia w fazie treningowej
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backpropagation i optymalizacja w fazie treningowej
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statystyki
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

# Trenuj model
model = train_model(model, criterion, optimizer, num_epochs=3)

# Zapisz wytrenowany model do pliku
torch.save(model, 'covid_model.pth')
print("Model zapisany jako covid_model.pth")
