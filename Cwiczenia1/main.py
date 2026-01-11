import sys  # Import modułu sys (typ: module) - umożliwia dostęp do parametrów wiersza poleceń i funkcji systemowych
import os  # Import modułu os (typ: module) - umożliwia interakcję z systemem operacyjnym (np. sprawdzanie plików)
import csv  # Import modułu csv (typ: module) - obsługuje odczyt i zapis plików w formacie CSV
import tensorflow as tf  # Import biblioteki TensorFlow (typ: module) - do uczenia maszynowego i sieci neuronowych, alias 'tf'
import numpy as np  # Import biblioteki NumPy (typ: module) - do operacji na macierzach i tablicach, alias 'np'
from PIL import Image  # Import klasy Image z biblioteki Pillow (typ: class) - do wczytywania i przetwarzania obrazów
import matplotlib.pyplot as plt  # Import modułu pyplot z biblioteki Matplotlib (typ: module) - do tworzenia wykresów, alias 'plt'

MODEL_PATH = "model.keras"  # Zmienna stała (typ: str) - przechowuje nazwę pliku, w którym zapisany będzie model

# Sekcja ładowania lub trenowania modelu
if os.path.exists(MODEL_PATH):  # Sprawdzenie, czy plik modelu już istnieje (zwraca typ: bool)
    print(f"Loading model from {MODEL_PATH}...")  # Wyświetlenie komunikatu (funkcja print, argument typ: str)
    model = tf.keras.models.load_model(MODEL_PATH)  # Wczytanie gotowego modelu z pliku (zwraca typ: keras.engine.training.Model)
else:  # Jeśli plik modelu nie istnieje
    print("Training new model...")  # Wyświetlenie komunikatu o rozpoczęciu treningu (argument typ: str)
    mnist = tf.keras.datasets.mnist  # Przypisanie referencji do zbioru danych MNIST z Keras (typ: module)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # Pobranie danych: x to obrazy (typ: np.ndarray), y to etykiety (typ: np.ndarray)
    
    # Normalizacja danych: dzielenie przez 255.0, aby wartości pikseli były w zakresie 0-1 (typ: np.ndarray, float64)
    x_train, x_test = x_train / 255.0, x_test / 255.0  

    model = tf.keras.models.Sequential(  # Utworzenie sekwencyjnego modelu sieci neuronowej (typ: keras.engine.sequential.Sequential)
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),  # Warstwa spłaszczająca wejście 28x28 do wektora jednowymiarowego (typ: keras.layers.core.flatten.Flatten)
            tf.keras.layers.Dense(128, activation="relu"),  # Warstwa gęsta (w pełni połączona) ze 128 neuronami i funkcją aktywacji ReLU (typ: keras.layers.core.dense.Dense)
            tf.keras.layers.Dropout(0.2),  # Warstwa Dropout - wyłącza losowo 20% neuronów, zapobiega przeuczeniu (typ: keras.layers.regularization.dropout.Dropout)
            tf.keras.layers.Dense(10, activation="softmax"),  # Warstwa wyjściowa z 10 neuronami (klasy 0-9) i aktywacją Softmax (prawdopodobieństwa) (typ: keras.layers.core.dense.Dense)
        ]
    )
    
    model.compile(  # Konfiguracja procesu uczenia modelu
        optimizer="adam",  # Wybór optymalizatora Adam (typ: str)
        loss="sparse_categorical_crossentropy",  # Funkcja straty dla klasyfikacji wieloklasowej (typ: str)
        metrics=["accuracy"]  # Metryka do monitorowania - dokładność (typ: list[str])
    )
    
    # Trenowanie modelu na danych treningowych przez 5 epok, z walidacją na danych testowych (zwraca typ: keras.callbacks.History)
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    
    model.evaluate(x_test, y_test)  # Ocena modelu na danych testowych po treningu (zwraca listę wyników: strata, dokładność)
    model.save(MODEL_PATH)  # Zapisanie wytrenowanego modelu do pliku .keras
    print(f"Model saved to {MODEL_PATH}")  # Wyświetlenie informacji o zapisaniu modelu (typ: str)

    # Zapis krzywej uczenia do pliku CSV
    with open("learning_curve.csv", "w", newline="") as f:  # Otwarcie pliku w trybie zapisu (typ kontekstu: io.TextIOWrapper)
        writer = csv.writer(f)  # Utworzenie obiektu do zapisu CSV (typ: _csv.writer)
        writer.writerow(history.history.keys())  # Zapisanie nagłówków kolumn (klucze słownika historii, typ: dict_keys)
        writer.writerows(zip(*history.history.values()))  # Zapisanie wierszy z danymi (transpozycja wartości słownika, typ: zip)
    print("Learning curve saved to learning_curve.csv")  # Komunikat o zapisaniu danych (typ: str)

# Obsługa predykcji lub rysowania wykresu na podstawie argumentów wiersza poleceń
if len(sys.argv) > 1:  # Sprawdzenie, czy podano dodatkowe argumenty (sys.argv to lista, len zwraca int > 1)
    image_path = sys.argv[1]  # Pobranie pierwszego argumentu jako ścieżki lub opcji (typ: str)

    if image_path == "--plot":  # Sprawdzenie, czy argument to flaga "--plot" (porównanie stringów)
        # Rysowanie krzywej uczenia
        with open("learning_curve.csv", "r") as f:  # Otwarcie pliku CSV do odczytu (typ kontekstu: io.TextIOWrapper)
            reader = csv.DictReader(f)  # Utworzenie czytnika słownikowego CSV (typ: csv.DictReader)
            data = list(reader)  # Konwersja iteratora do listy słowników (typ: list[dict])

        epochs = range(1, len(data) + 1)  # Utworzenie zakresu epok od 1 do liczby wierszy (typ: range)
        loss = [float(row["loss"]) for row in data]  # Wyciągnięcie listy strat treningowych, konwersja na float (typ: list[float])
        val_loss = [float(row["val_loss"]) for row in data]  # Wyciągnięcie listy strat walidacyjnych (typ: list[float])
        accuracy = [float(row["accuracy"]) for row in data]  # Wyciągnięcie listy dokładności treningowych (typ: list[float])
        val_accuracy = [float(row["val_accuracy"]) for row in data]  # Wyciągnięcie listy dokładności walidacyjnych (typ: list[float])

        plt.figure(figsize=(12, 4))  # Utworzenie nowego okna wykresu o wymiarach 12x4 cali (typ: matplotlib.figure.Figure)

        plt.subplot(1, 2, 1)  # Utworzenie pierwszego podwykresu w siatce 1x2 (typ: matplotlib.axes._subplots.AxesSubplot)
        plt.plot(epochs, loss, label="Training Loss")  # Rysowanie wykresu straty treningowej (oś X: epochs, oś Y: loss)
        plt.plot(epochs, val_loss, label="Validation Loss")  # Rysowanie wykresu straty walidacyjnej
        plt.xlabel("Epoch")  # Etykieta osi X (typ: str)
        plt.ylabel("Loss")  # Etykieta osi Y (typ: str)
        plt.legend()  # Wyświetlenie legendy

        plt.subplot(1, 2, 2)  # Utworzenie drugiego podwykresu
        plt.plot(epochs, accuracy, label="Training Accuracy")  # Wykres dokładności treningowej
        plt.plot(epochs, val_accuracy, label="Validation Accuracy")  # Wykres dokładności walidacyjnej
        plt.xlabel("Epoch")  # Etykieta osi X
        plt.ylabel("Accuracy")  # Etykieta osi Y
        plt.legend()  # Legenda

        plt.tight_layout()  # Automatyczne dopasowanie odstępów między wykresami
        plt.show()  # Wyświetlenie okna z wykresami (blokuje wykonanie programu do zamknięcia okna)
    else:
        # Wczytanie i przetworzenie obrazu do predykcji
        img = Image.open(image_path).convert("L")  # Otwarcie obrazu i konwersja na skalę szarości ('L') (typ: PIL.Image.Image)
        img = img.resize((28, 28))  # Zmiana rozmiaru obrazu na 28x28 pikseli zgodnie z wejściem modelu (zwraca typ: PIL.Image.Image)
        img_array = np.array(img) / 255.0  # Konwersja obrazu na tablicę NumPy i normalizacja (typ: np.ndarray, float64)
        img_array = np.expand_dims(img_array, axis=0)  # Dodanie wymiaru partii (batch dimension), kształt (1, 28, 28) (typ: np.ndarray)

        # Wykonanie predykcji
        predictions = model.predict(img_array, verbose=0)  # Uruchomienie modelu na obrazie (zwraca typ: np.ndarray z prawdopodobieństwami)
        digit = np.argmax(predictions[0])  # Znalezienie indeksu z największą wartością (cyfra o największym prawdopodobieństwie, typ: int64)

        print(f"Predicted digit: {digit}")  # Wyświetlenie wyniku predykcji (f-string)
