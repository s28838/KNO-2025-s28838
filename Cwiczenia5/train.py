
import os  # Import modułu os (typ: module) - operacje systemowe
import json  # Import modułu json (typ: module) - zapis/odczyt plików JSON
import numpy as np  # Import biblioteki NumPy (typ: module) - obliczenia numeryczne
import tensorflow as tf  # Import TensorFlow (typ: module) - uczenie maszynowe
import keras_tuner as kt  # Import Keras Tuner (typ: module) - strojenie hiperparametrów
from sklearn.metrics import confusion_matrix  # Import funkcji macierzy pomyłek (typ: function)
from models import build_model  # Import własnej funkcji budującej model (typ: function)
import sys  # Import modułu sys (typ: module) - operacje systemowe

# Konfiguracja UTF-8 dla konsoli Windows, aby polskie znaki wyświetlały się poprawnie
sys.stdout.reconfigure(encoding='utf-8')

# Stałe konfiguracyjne
BATCH_SIZE = 64  # Rozmiar partii danych podczas treningu (typ: int)
EPOCHS_SEARCH = 5  # Liczba epok w fazie poszukiwania najlepszych parametrów (typ: int)
EPOCHS_FINAL = 15  # Liczba epok dla finalnego treningu najlepszego modelu (typ: int)
MODEL_PATH = "fashion_model.keras"  # Ścieżka do zapisu wytrenowanego modelu (typ: str)
METRICS_PATH = "metrics.json"  # Ścieżka do zapisu metryk w formacie JSON (typ: str)

def main():
    print("Ładowanie danych Fashion MNIST...")  # (typ: None)
    # 1. Załaduj dane
    # Funkcja ładowania zbioru Fashion MNIST zwraca dwie krotki z tablicami NumPy (typ: function)
    # x_train, x_test: obrazy uint8 o kształcie (60000, 28, 28) i (10000, 28, 28)
    # y_train, y_test: etykiety uint8 (0-9)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # 2. Preprocessing
    # Skalowanie wartości pikseli z zakresu 0-255 do 0-1 (normalizacja)
    # Konwersja na float32 dla lepszej wydajności obliczeniowej (typ: np.ndarray)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape do (N, 28, 28, 1) - dodanie wymiaru kanału
    # Jest to wymagane przez warstwy Conv2D w Keras, które oczekują wejścia 4D
    # (samples, height, width, channels)
    x_train = np.expand_dims(x_train, -1)  # (typ: np.ndarray, shape=(60000, 28, 28, 1))
    x_test = np.expand_dims(x_test, -1)  # (typ: np.ndarray, shape=(10000, 28, 28, 1))

    print(f"Dane treningowe: {x_train.shape}")  # Wyświetlenie kształtu danych (typ: None)
    print(f"Dane testowe: {x_test.shape}")  # (typ: None)

    # 3. Keras Tuner
    print("Start strojenia hiperparametrów...")  # (typ: None)
    # Konfiguracja tunera RandomSearch (losowe przeszukiwanie przestrzeni parametrów)
    tuner = kt.RandomSearch(
        build_model,  # Funkcja konstruująca model (zdefiniowana w models.py)
        objective="val_accuracy",  # Cel optymalizacji: dokładność na zbiorze walidacyjnym (typ: str)
        max_trials=10,  # Maksymalna liczba różnych kombinacji do sprawdzenia (typ: int)
        executions_per_trial=1,  # Ile razy wytrenować każdą kombinację (tutaj 1 raz) (typ: int)
        directory="kt_dir",  # Katalog roboczy tunera (typ: str)
        project_name="fashion_mnist_tuning",  # Nazwa projektu (podkatalogu w kt_dir) (typ: str)
        overwrite=True  # Czy nadpisać poprzednie wyniki tuningu (typ: bool)
    )  # (typ: keras_tuner.tuners.randomsearch.RandomSearch)

    # Uruchomienie procesu wyszukiwania
    # validation_split=0.2 oznacza, że 20% danych treningowych zostanie użyte jako walidacja
    tuner.search(x_train, y_train, 
                 epochs=EPOCHS_SEARCH, 
                 validation_split=0.2, 
                 verbose=1)  # (metoda zwraca None)

    # 4. Pobieranie najlepszego modelu
    # Pobranie obiektu HyperParameters najlepszej próby
    # num_trials=1 oznacza pobranie top 1 wyniku
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]  # (typ: keras_tuner.engine.hyperparameters.HyperParameters)
    
    print("\nZnaleziono najlepsze hiperparametry:")  # (typ: None)
    print(f"Typ modelu: {best_hps.get('model_type')}")  # Wyświetlenie typu modelu (typ: str)
    print(f"Learning rate: {best_hps.get('lr')}")  # Wyświetlenie LR (typ: float)
    
    # 5. Pełny trening
    print("Trenowanie najlepszego modelu...")  # (typ: None)
    # Zbudowanie modelu na nowo przy użyciu najlepszych parametrów
    model = tuner.hypermodel.build(best_hps)  # (typ: keras.engine.sequential.Sequential)
    
    # Trenowanie modelu na pełnym zbiorze danych (lub z podziałem, tutaj używamy x_test jako validation data dla podglądu)
    history = model.fit(x_train, y_train,
                        epochs=EPOCHS_FINAL,  # Dłuższy trening dla lepszej zbieżności
                        validation_data=(x_test, y_test),
                        batch_size=BATCH_SIZE)  # (typ: keras.callbacks.History)

    # 6. Ewaluacja i Zapis
    # Ocena modelu na danych testowych (zwraca stratę i dokładność)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)  # (typ: tuple[float, float])
    print(f"\nFinal Test Loss: {loss:.4f}")  # (typ: None)
    print(f"Final Test Accuracy: {acc:.4f}")  # (typ: None)

    # Zapisanie całego modelu do pliku .keras
    model.save(MODEL_PATH)  # (metoda zwraca None)
    print(f"Model zapisano do {MODEL_PATH}")  # (typ: None)

    # Macierz pomyłek
    # Wykonanie predykcji dla zbioru testowego (zwraca prawdopodobieństwa)
    y_pred_probs = model.predict(x_test)  # (typ: np.ndarray, shape=(10000, 10))
    # Wybranie indeksu klasy z najwyższym prawdopodobieństwem
    y_pred = np.argmax(y_pred_probs, axis=1)  # (typ: np.ndarray, shape=(10000,))
    
    # Obliczenie macierzy pomyłek (porównanie etykiet prawdziwych z przewidzianymi)
    cm = confusion_matrix(y_test, y_pred)  # (typ: np.ndarray, shape=(10, 10))
    
    # Przygotowanie słownika z metrykami do zapisu
    metrics = {
        "loss": loss,  # (typ: float)
        "accuracy": acc,  # (typ: float)
        "best_params": best_hps.values,  # Wartości najlepszych hiperparametrów (typ: dict)
        "confusion_matrix": cm.tolist()  # Konwersja ndarray na listę dla formatu JSON (typ: list)
    }
    
    # Zapis słownika do pliku JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)  # (funkcja zwraca None)
        
    print(f"Metryki zapisano do {METRICS_PATH}")  # (typ: None)

# Standardowy blok uruchomieniowy
if __name__ == "__main__":
    main()
