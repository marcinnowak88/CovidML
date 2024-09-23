import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
from PIL import ImageOps

# Konfiguracja aplikacji Flask
app = Flask(__name__)

# Ścieżka do folderu, w którym będą przechowywane przesłane obrazy
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Załaduj wytrenowany model
model = torch.load('covid_model.pth', map_location=torch.device('cpu'))
model.eval()  # Ustaw model w tryb ewaluacyjny (testowy)

# Transformacje obrazu – dopasowanie do tego, co było użyte podczas trenowania
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Zmiana rozmiaru na 224x224 pikseli
    transforms.ToTensor(),  # Zamiana na tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalizacja
])


# Strona główna – formularz do przesyłania obrazu
@app.route('/')
def index():
    return render_template('index.html')


# Przetwarzanie przesłanego obrazu
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Zapisz przesłany obraz
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Klasyfikacja obrazu
        label = classify_image(filepath)

        return render_template('result.html', label=label, image_file=filepath)


# Funkcja do klasyfikacji obrazu
def classify_image(filepath):
    # Otwórz obraz
    img = Image.open(filepath)

    # Jeśli obraz ma tylko jeden kanał (np. grayscale), konwertuj go na 3-kanałowy (RGB)
    if img.mode != 'RGB':
        img = ImageOps.grayscale(img).convert("RGB")

    # Zastosuj transformacje
    img = transform(img).unsqueeze(0)  # Dodaj wymiar batch

    # Przeprowadź predykcję
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    # Wynik klasyfikacji
    classes = ['COVID','Normal']
    return classes[predicted.item()]


# Funkcja sprawdzająca, czy przesłany plik jest obrazem
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Uruchomienie aplikacji
if __name__ == '__main__':
    app.run(debug=True)

