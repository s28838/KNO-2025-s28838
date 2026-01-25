import json
import sys

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from models import build_model

# Konfiguracja kodowania wyjścia dla Windows (obsługa polskich znaków).
sys.stdout.reconfigure(encoding='utf-8')  # Ustawia kodowanie UTF-8 dla konsoli.

# Parametry globalne eksperymentu.
BATCH_SIZE = 64              # Liczba próbek przetwarzana w jednym kroku gradientowym.
EPOCHS_SEARCH = 5            # Krótki trening w fazie poszukiwania najlepszych parametrów.
EPOCHS_FINAL = 15            # Dłuższy trening dla wybranej, najlepszej konfiguracji.
MODEL_PATH = "fashion_model.keras"  # Ścieżka do zapisu finalnego modelu.
METRICS_PATH = "metrics.json"  # Ścieżka do zapisu metryk.


def main():  # Główna funkcja sterująca procesem treningu.
    print("Ładowanie danych Fashion MNIST...")  # Logowanie postępu.
    
    # 1. Pobranie datasetu.
    # Zbiór składa się z 60k obrazów treningowych i 10k testowych.
    # Obrazy są w skali szarości, 28x28 pikseli.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()  # Pobiera dane Fashion MNIST.

    # 2. Preprocessing danych.
    # Normalizacja: przeskalowanie wartości pikseli z zakresu [0, 255] do [0.0, 1.0].
    # Pomaga to w szybszej zbieżności algorytmu optymalizacyjnego.
    x_train = x_train.astype("float32") / 255.0  # Normalizuje zbiór treningowy.
    x_test = x_test.astype("float32") / 255.0  # Normalizuje zbiór testowy.

    # Reshape (zmiana kształtu).
    # Dodanie czwartego wymiaru (kanału), aby dane pasowały do warstw konwolucyjnych (Conv2D).
    # Oczekiwany format: (N, 28, 28, 1).
    x_train = np.expand_dims(x_train, -1)  # Dodaje wymiar kanału do danych treningowych.
    x_test = np.expand_dims(x_test, -1)  # Dodaje wymiar kanału do danych testowych.

    print(f"Dane treningowe: {x_train.shape}")  # Wypisuje kształt danych treningowych.
    print(f"Dane testowe: {x_test.shape}")  # Wypisuje kształt danych testowych.

    # 3. Strojenie hiperparametrów (Keras Tuner).
    print("Start strojenia hiperparametrów...")  # Loguje start tuningu.
    
    # Inicjalizacja algorytmu Random Search.
    # Będzie on losowo wybierał konfiguracje zdefiniowane w models.py (build_model)
    # i sprawdzał ich skuteczność na zbiorze walidacyjnym.
    tuner = kt.RandomSearch(  # Konfiguruje tuner RandomSearch.
        build_model,                     # Funkcja budująca model.
        objective="val_accuracy",        # Metryka do maksymalizacji (dokładność walidacyjna).
        max_trials=10,                   # Liczba różnych konfiguracji do przetestowania.
        executions_per_trial=1,          # Liczba powtórzeń treningu dla jednej konfiguracji (dla uśrednienia).
        directory="kt_dir",              # Katalog na tymczasowe wyniki tunera.
        project_name="fashion_mnist_tuning",  # Nazwa projektu tunera.
        overwrite=True                   # Startujemy od czysta przy każdym uruchomieniu.
    )

    # Uruchomienie przeszukiwania.
    # validation_split=0.2 automatycznie wydziela 20% danych treningowych jako zbiór walidacyjny.
    tuner.search(x_train, y_train,   # Uruchamia przeszukiwanie.
                 epochs=EPOCHS_SEARCH,   # Liczba epok dla każdego trialu.
                 validation_split=0.2,   # Podział walidacyjny wewnątrz tunera.
                 verbose=1)  # Poziom logowania.

    # 4. Wybór najlepszej konfiguracji.
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]  # Pobiera najlepsze hiperparametry.
    
    print("\nZnaleziono najlepsze hiperparametry:")  # Wypisuje wynik tuningu.
    print(f"Typ modelu: {best_hps.get('model_type')}")  # Wypisuje najlepszy typ modelu.
    print(f"Learning rate: {best_hps.get('lr')}")  # Wypisuje najlepszy learning rate.
    
    # 5. Trening finalny.
    print("Trenowanie najlepszego modelu...")  # Loguje start finalnego treningu.
    # Odbudowa modelu z najlepszymi znalezionymi parametrami.
    model = tuner.hypermodel.build(best_hps)  # Buduje optymalny model.
    
    # Trening na pełnym zbiorze (lub z tym samym podziałem walidacyjnym).
    # Tutaj używamy x_test jako walidacji tylko do podglądu postępów (nie wpływa na trening).
    history = model.fit(x_train, y_train,  # Uruchamia trening modelu.
                        epochs=EPOCHS_FINAL,  # Liczba epok finalnych.
                        validation_data=(x_test, y_test),  # Dane do podglądu walidacji.
                        batch_size=BATCH_SIZE)  # Rozmiar batcha.

    # 6. Ewaluacja końcowa.
    loss, acc = model.evaluate(x_test, y_test, verbose=0)  # Ocenia model na zbiorze testowym.
    print(f"\nFinal Test Loss: {loss:.4f}")  # Wypisuje końcową stratę.
    print(f"Final Test Accuracy: {acc:.4f}")  # Wypisuje końcową dokładność.

    # Zapis modelu.
    model.save(MODEL_PATH)  # Zapisuje model do pliku.
    print(f"Model zapisano do {MODEL_PATH}")  # Potwierdza zapis.

    # 7. Generowanie metryk dodatkowych (Macierz Pomyłek).
    # Predykcja na zbiorze testowym.
    y_pred_probs = model.predict(x_test)  # Wykonuje predykcję na x_test.
    y_pred = np.argmax(y_pred_probs, axis=1)  # Wyznacza przewidziane klasy.
    
    cm = confusion_matrix(y_test, y_pred)  # Oblicza macierz pomyłek.
    
    # Zapis metryk do pliku JSON w celu łatwej analizy lub automatycznego parsowania.
    metrics = {  # Przygotowuje słownik metryk.
        "loss": loss,  # Wartość straty.
        "accuracy": acc,  # Wartość dokładności.
        "best_params": best_hps.values,  # Słownik z najlepszymi wartościami hyperparametrów.
        "confusion_matrix": cm.tolist()  # Macierz pomyłek jako lista list.
    }
    
    with open(METRICS_PATH, "w") as f:  # Otwiera plik JSON do zapisu.
        json.dump(metrics, f, indent=4)  # Zapisuje metryki w formacie JSON.
        
    print(f"Metryki zapisano do {METRICS_PATH}")  # Potwierdza zapis metryk.


if __name__ == "__main__":  # Punkt wejścia skryptu.
    main()  # Uruchamia funkcję main.
