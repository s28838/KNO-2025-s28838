import argparse
import sys

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Wymuszenie kodowania UTF-8 dla standardowego wyjścia (konsola Windows).
# Zapobiega to błędom UnicodeEncodeError przy próbie wypisania polskich znaków.
sys.stdout.reconfigure(encoding='utf-8')  # Konfiguruje standardowe wyjście na UTF-8.

# Mapa nazw klas (index -> nazwa).
# Kolejność jest zgodna z definicją zbioru danych Fashion MNIST.
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def preprocess_image(image_path):  # Przygotowuje obraz z dysku do inferencji przez sieć neuronową.
    try:  # Rozpoczyna blok try-except.
        # Wczytanie obrazu za pomocą biblioteki PIL (Pillow).
        img = Image.open(image_path)  # Otwiera obraz ze ścieżki.
        
        # Konwersja do skali szarości.
        # Kolor nie niesie kluczowej informacji dla kształtu ubrania w tym zbiorze.
        img = img.convert("L")  # Konwertuje obraz na odcienie szarości.
        
        # Inwersja kolorów (Negative).
        # Krytyczny krok dla zdjęć np. czarnej koszulki na białym tle.
        # Sieć oczekuje jasnego obiektu na ciemnym tle (wartości pikseli tła ~0).
        img = ImageOps.invert(img)  # Odwraca kolory (negatyw).
        
        # Zmiana rozmiaru na 28x28 (standard MNIST).
        img = img.resize((28, 28))  # Skaluje obraz do wymiarów wejściowych sieci.
        
        # Konwersja na tablicę NumPy i normalizacja.
        # Dzielenie przez 255.0 sprowadza wartości pikseli (0-255) do przedziału (0.0-1.0).
        img_array = np.array(img).astype("float32") / 255.0  # Normalizuje wartości pikseli.
        
        # Rozszerzenie wymiarów.
        # Keras oczekuje tensora 4D: (Batch_Size, Height, Width, Channels).
        # Oryginalnie mamy (28, 28).
        # Po expand_dims(axis=0) mamy (1, 28, 28).
        img_array = np.expand_dims(img_array, axis=0)  # Dodaje wymiar batcha.
        # Po expand_dims(axis=-1) mamy (1, 28, 28, 1).
        img_array = np.expand_dims(img_array, axis=-1)  # Dodaje wymiar kanału.
        
        return img_array  # Zwraca przygotowany tensor.
    except Exception as e:  # Obsługuje wyjątki podczas przetwarzania.
        print(f"Błąd przetwarzania obrazu: {e}")  # Wypisuje błąd.
        return None  # Zwraca None w przypadku błędu.


def main():  # Główna funkcja programu. Parsuje argumenty, ładuje model i wyświetla wynik predykcji.
    parser = argparse.ArgumentParser(description="Klasyfikator ubrań (Fashion MNIST)")  # Tworzy parser argumentów.
    parser.add_argument("image", help="Ścieżka do pliku obrazka")  # Dodaje argument ścieżki do obrazu.
    args = parser.parse_args()  # Parsuje argumenty.
    
    # Preprocessing danych wejściowych.
    input_data = preprocess_image(args.image)  # Przetwarza obraz wejściowy.
    if input_data is None:  # Sprawdza czy przetwarzanie się powiodło.
        return  # Kończy działanie w przypadku błędu.

    # Ładowanie modelu.
    try:  # Blok try dla ładowania modelu.
        model = tf.keras.models.load_model("fashion_model.keras")  # Ładuje model z pliku.
    except OSError:  # Obsługa błędu braku pliku modelu.
        print("Błąd: Nie znaleziono modelu 'fashion_model.keras'. Uruchom najpierw train.py.")  # Komunikat dla użytkownika.
        return  # Kończy działanie.

    # Inferencja.
    # verbose=0 zapobiega wypisywaniu logów (np. "1/1 [================]") na stdout.
    predictions = model.predict(input_data, verbose=0)  # Wykonuje predykcję.
    
    # Wyciągnięcie wektora prawdopodobieństw dla pierwszego elementu batcha.
    probs = predictions[0]  # Pobiera wynik dla pojedynczego obrazu.
    
    # Identyfikacja klasy zwycięskiej.
    predicted_class_idx = np.argmax(probs)  # Znajduje indeks klasy z max prawdopodobieństwem.
    confidence = probs[predicted_class_idx]  # Pobiera wartość pewności.
    class_name = CLASS_NAMES[predicted_class_idx]  # Pobiera nazwę klasy.
    
    # Prezentacja wyników.
    print("-" * 30)  # Separator.
    print(f"Wynik klasyfikacji:")  # Nagłówek.
    print(f"Klasa:   {class_name} (ID: {predicted_class_idx})")  # Wypisuje wynik.
    print(f"Pewność: {confidence:.2%}")  # Wypisuje pewność.
    print("-" * 30)  # Separator.
    print("Rozkład prawdopodobieństwa:")  # Nagłówek rozkładu.
    
    for i, p in enumerate(probs):  # Iteruje po prawdopodobieństwach wszystkich klas.
        # Formatowanie: nazwa wyrównana do lewej (12 znaków), prawdopodobieństwo (4 cyfry po przecinku).
        print(f"  {CLASS_NAMES[i]:<12}: {p:.4f}")  # Wypisuje sformatowaną linię.


if __name__ == "__main__":  # Punkt wejścia skryptu.
    main()  # Uruchamia funkcję main.
