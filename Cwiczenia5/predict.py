
import argparse  # Import modułu argparse (typ: module) - parsowanie argumentów z linii komend
import numpy as np  # Import biblioteki NumPy (typ: module) - obliczenia na tablicach
import tensorflow as tf  # Import TensorFlow (typ: module) - ładowanie modelu i inferencja
from PIL import Image, ImageOps  # Import klasy Image i modułu ImageOps z biblioteki Pillow (typ: class, module) - przetwarzanie obrazów
import sys  # Import modułu sys (typ: module) - konfiguracja systemu

# Konfiguracja kodowania wyjścia na UTF-8, aby polskie znaki w printach działały poprawnie na Windows
sys.stdout.reconfigure(encoding='utf-8')

# Lista etykiet klas dla zbioru Fashion MNIST (typ: list[str])
# Indeksy odpowiadają wyjściom modelu (0-9)
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def preprocess_image(image_path):
    """
    Wczytuje obraz, zmienia rozmiar na 28x28, konwertuje na skalę szarości,
    wykonuje negatyw (białe tło -> czarne tło) i normalizuje.
    Argument image_path to ścieżka do pliku (typ: str).
    Zwraca przetworzony obraz jako tablicę NumPy lub None w przypadku błędu.
    """
    try:
        # 1. Wczytaj obraz z pliku
        img = Image.open(image_path)  # (typ: PIL.Image.Image)
        
        # 2. Konwersja do skali szarości ('L' - Luminance)
        # Fashion MNIST to obrazy czarno-białe
        img = img.convert("L")  # (typ: PIL.Image.Image)
        
        # 3. Negatyw (inwersja kolorów)
        # Ważne: Typowe zdjęcie 'koszulki na białym tle' po konwersji na szarość ma jasne tło (wartości bliskie 255).
        # Zbiór Fashion MNIST ma czarne tło (wartości bliskie 0) i jasny obiekt.
        # Dlatego musimy odwrócić kolory: pixel = 255 - pixel
        img = ImageOps.invert(img)  # (typ: PIL.Image.Image)
        
        # 4. Zmiana rozmiaru na 28x28 pikseli
        # Taki rozmiar wejściowy ma wytrenowana sieć
        img = img.resize((28, 28))  # (typ: PIL.Image.Image)
        
        # 5. Konwersja obiektu obrazu na tablicę NumPy
        img_array = np.array(img)  # (typ: np.ndarray, shape=(28, 28))
        
        # 6. Normalizacja wartości pikseli (0-255 -> 0.0-1.0)
        # Model był uczony na danych znormalizowanych
        img_array = img_array.astype("float32") / 255.0  # (typ: np.ndarray)
        
        # 7. Dodanie wymiarów (batch_size, height, width, channels)
        # Model oczekuje wejścia 4D: (N, 28, 28, 1)
        # expand_dims(axis=0) dodaje wymiar batch (1, 28, 28)
        img_array = np.expand_dims(img_array, axis=0) 
        # expand_dims(axis=-1) dodaje wymiar kanału (1, 28, 28, 1)
        img_array = np.expand_dims(img_array, axis=-1)  # (typ: np.ndarray, shape=(1, 28, 28, 1))
        
        return img_array
    except Exception as e:
        # Obsługa wyjątków (np. brak pliku, błędny format)
        print(f"Błąd przetwarzania obrazu: {e}")  # (typ: None)
        return None

def main():
    # Konfiguracja parsera argumentów
    parser = argparse.ArgumentParser(description="Klasyfikator ubrań (Fashion MNIST)")  # (typ: argparse.ArgumentParser)
    # Dodanie argumentu pozycyjnego 'image' (wymagany)
    parser.add_argument("image", help="Ścieżka do pliku obrazka")  # (typ: None)
    # Parsowanie argumentów z linii komend
    args = parser.parse_args()  # (typ: argparse.Namespace)
    
    # Przetwarzanie obrazu wejściowego
    input_data = preprocess_image(args.image)  # (typ: np.ndarray | None)
    
    # Jeśli wystąpił błąd w preprocessingu, zakończ działanie
    if input_data is None:
        return

    # Ładowanie wytrenowanego modelu
    try:
        # Wczytanie modelu z pliku .keras
        model = tf.keras.models.load_model("fashion_model.keras")  # (typ: keras.engine.sequential.Sequential)
    except OSError:
        print("Błąd: Nie znaleziono modelu 'fashion_model.keras'. Uruchom najpierw train.py.")  # (typ: None)
        return

    # Predykcja
    # verbose=0 wyłącza pasek postępu
    predictions = model.predict(input_data, verbose=0)  # (typ: np.ndarray, shape=(1, 10))
    
    # Pobranie wyników dla pierwszego (i jedynego) obrazka w batchu
    probs = predictions[0]  # (typ: np.ndarray, shape=(10,))
    
    # Znalezienie indeksu klasy z najwyższym prawdopodobieństwem
    predicted_class_idx = np.argmax(probs)  # (typ: int64)
    # Pobranie wartości pewności (prawdopodobieństwa) dla wybranej klasy
    confidence = probs[predicted_class_idx]  # (typ: float32)
    
    # Pobranie nazwy klasy z listy tekstowej
    class_name = CLASS_NAMES[predicted_class_idx]  # (typ: str)
    
    # Wyświetlenie wyników
    print("-" * 30)
    print(f"Wynik klasyfikacji:")
    print(f"Klasa:   {class_name} (ID: {predicted_class_idx})")
    # Formatowanie procentowe (.2%)
    print(f"Pewność: {confidence:.2%}")
    print("-" * 30)
    print("Rozkład prawdopodobieństwa:")
    # Iteracja po wszystkich klasach i wyświetlenie ich prawdopodobieństw
    for i, p in enumerate(probs):
        # Formatowanie: nazwa klasy wyrównana do lewej (12 znaków), prawdopodobieństwo do 4 miejsc po przecinku
        print(f"  {CLASS_NAMES[i]:<12}: {p:.4f}")

# Standardowy punkt wejścia programu
if __name__ == "__main__":
    main()
