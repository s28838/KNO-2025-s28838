import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "model.keras"  # Definiuje stałą ze ścieżką do pliku modelu, używana w całym skrypcie do zapisu i ponownego odczytu wytrenowanego modelu.

# Sprawdzenie czy model już istnieje, aby uniknąć niepotrzebnego ponownego treningu.
# Jeśli plik modelu zostanie znaleziony, zostanie załadowany. W przeciwnym razie rozpocznie się proces uczenia.
if os.path.exists(MODEL_PATH):  # Sprawdza fizyczne istnienie pliku modelu na dysku, decyduje w przepływie sterowania czy trenować czy ładować.
    print(f"Loading model from {MODEL_PATH}...")  # Wyświetla komunikat informacyjny dla użytkownika, informuje o rozpoczęciu ładowania istniejącego modelu.
    model = tf.keras.models.load_model(MODEL_PATH)  # Wczytuje kompletny model (architektura + wagi) z pliku, inicjalizuje zmienną model do użycia w predykcji.
else:  # Blok else wykonuje się, gdy plik modelu nie istnieje, inicjuje procedurę treningu od zera.
    print("Training new model...")  # Wyświetla komunikat informacyjny dla użytkownika, sygnalizuje rozpoczęcie nowego procesu treningowego.
    # Pobranie zbioru MNIST z repozytorium Keras.
    # Zbiór składa się z 60 000 obrazów treningowych i 10 000 testowych.
    mnist = tf.keras.datasets.mnist  # Przypisuje moduł zbioru danych MNIST z Keras do zmiennej, umożliwia dostęp do metody load_data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # Pobiera i rozpakowuje dane do krotek treningowych i testowych, dane te są podstawą uczenia i walidacji.

    # Normalizacja wartości pikseli.
    # Domyślne wartości pikseli są w zakresie 0-255 (uint8).
    # Skalowanie ich do przedziału 0.0-1.0 (float) znacznie przyspiesza zbieżność algorytmu 
    # optymalizacyjnego (Gradient Descent) i poprawia stabilność uczenia.
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Dzieli wartości pikseli przez 255.0, normalizuje dane wejściowe do zakresu [0, 1] dla lepszej wydajności sieci.

    # Definicja modelu sekwencyjnego using Keras API.
    model = tf.keras.models.Sequential(  # Inicjalizuje pusty model sekwencyjny Keras, kontener na stos warstw neuronalnych.
        [
            # Warstwa Flatten przekształca macierz 2D (28x28) w wektor 1D (784 elementy).
            # Jest to konieczne, ponieważ warstwy Dense oczekują płaskiego wektora cech.
            tf.keras.layers.Flatten(input_shape=(28, 28)),  # Dodaje warstwę spłaszczającą wejście, przygotowuje dane obrazu 2D dla warstw gęstych.
            
            # Warstwa gęsta (Dense) z funkcją aktywacji ReLU.
            # 128 neuronów pozwala modelowi nauczyć się nieliniowych relacji złożonych cech.
            tf.keras.layers.Dense(128, activation="relu"),  # Dodaje warstwę ukrytą z 128 neuronami i aktywacją ReLU, główna warstwa ucząca się cech.
            
            # Warstwa Dropout (rzucanie monetą dla każdego neuronu).
            # Losowo "wyłącza" 20% neuronów w trakcie każdego kroku treningowego.
            # Zmusza to sieć do nauki redundantnych reprezentacji, co zapobiega overfittingowi (przeuczeniu).
            tf.keras.layers.Dropout(0.2),  # Dodaje warstwę Dropout z prawdopodobieństwem 0.2, mechanizm regularyzacji zapobiegający przeuczeniu podczas treningu.
            
            # Warstwa wyjściowa z 10 neuronami (dla cyfr 0-9).
            # Funkcja Softmax zamienia surowe wyniki (logits) na rozkład prawdopodobieństwa sumujący się do 1.
            tf.keras.layers.Dense(10, activation="softmax"),  # Dodaje warstwę wyjściową z 10 neuronami i aktywacją Softmax, generuje prawdopodobieństwa dla każdej klasy cyfr.
        ]
    )

    # Kompilacja modelu.
    # Adam to adaptacyjny algorytm optymalizacji, który zazwyczaj działa dobrze bez strojenia.
    # sparse_categorical_crossentropy jest odpowiednia, gdy etykiety są liczbami całkowitymi (a nie one-hot).
    model.compile(  # Konfiguruje proces uczenia modelu, definiuje optymalizator, funkcję straty i metryki.
        optimizer="adam",  # Wybiera optymalizator Adam, algorytm aktualizacji wag w procesie uczenia.
        loss="sparse_categorical_crossentropy",  # Ustawia funkcję straty dla klas całkowitoliczbowych, miara błędu modelu do minimalizacji.
        metrics=["accuracy"],  # Definiuje monitorowane metryki, w tym przypadku dokładność (accuracy) do oceny jakości modelu.
    )

    # Rozpoczęcie procesu uczenia.
    # epochs=5: Model przejdzie przez cały zbiór treningowy 5 razy.
    # validation_data: Użycie zbioru testowego do monitorowania postępów na danych, których model nie widzi.
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))  # Uruchamia pętlę treningową na 5 epok, zwraca obiekt historii z metrykami procesu uczenia.

    # Ostateczna ewaluacja na zbiorze testowym po zakończeniu treningu.
    model.evaluate(x_test, y_test)  # Przeprowadza ewaluację modelu na danych testowych, wypisuje ostateczne wskaźniki skuteczności.
    
    # Zapis modelu do pliku w formacie Keras, co pozwala na jego późniejsze użycie bez retrenowania.
    model.save(MODEL_PATH)  # Zapisuje stan modelu i wagi do pliku na dysku, umożliwia trwałość wytrenowanej sieci.
    print(f"Model saved to {MODEL_PATH}")  # Informuje użytkownika o pomyślnym zapisie modelu, potwierdzenie zakończenia operacji IO.

    # Eksport historii uczenia do CSV.
    # Pozwala to na zewnętrzną analizę lub wizualizację przebiegu funkcji straty i dokładności.
    with open("learning_curve.csv", "w", newline="") as f:  # Otwiera plik learning_curve.csv do zapisu, używany jako uchwyt do eksportu danych statystycznych.
        writer = csv.writer(f)  # Tworzy obiekt writera CSV powiązany z plikiem, służy do zapisu wierszy danych.
        writer.writerow(history.history.keys())  # Zapisuje nagłówek CSV z nazwami metryk, definiuje strukturę pliku wynikowego.
        writer.writerows(zip(*history.history.values()))  # Zapisuje wartości metryk epoka po epoce, transponuje dane słownika historii do formatu wierszowego.
    print("Learning curve saved to learning_curve.csv")  # Informuje użytkownika o zapisaniu logów uczenia, potwierdzenie operacji eksportu danych.

# ==============================================================================
# Obsługa Argumentów Linii Poleceń (CLI)
# ==============================================================================
# Jeśli skrypt został uruchomiony z dodatkowymi argumentami, wykonujemy specyficzne akcje:
# 1. "--plot": Rysuje wykresy uczenia.
# 2. <ścieżka_do_pliku>: Traktuje argument jako ścieżkę do obrazu i wykonuje predykcję.

if len(sys.argv) > 1:  # Sprawdza czy przekazano argumenty wywołania skryptu, decyduje o uruchomieniu trybu interaktywnego/CLI.
    image_path = sys.argv[1]  # Pobiera pierwszy argument po nazwie skryptu, przypisuje go do zmiennej image_path jako potencjalną ścieżkę lub komendę.

    if image_path == "--plot":  # Sprawdza czy argument to flaga "--plot", steruje wyborem trybu wizualizacji wykresów.
        # Tryb wizualizacji: Odczyt danych z CSV i generowanie wykresów Matplotlib.
        with open("learning_curve.csv", "r") as f:  # Otwiera plik z historią uczenia do odczytu, źródło danych do wykresów.
            reader = csv.DictReader(f)  # Tworzy czytnik CSV mapujący wiersze na słowniki, ułatwia dostęp do kolumn przez nazwy.
            data = list(reader)  # Wczytuje wszystkie wiersze z pliku do listy, ładuje dane do pamięci operacyjnej.

        epochs = range(1, len(data) + 1)  # Generuje zakres numerów epok na podstawie ilości danych, oś X wykresów.
        loss = [float(row["loss"]) for row in data]  # Ekstrahuje i konwertuje wartości straty treningowej, seria danych dla wykresu straty.
        val_loss = [float(row["val_loss"]) for row in data]  # Ekstrahuje wartości straty walidacyjnej, seria porównawcza dla wykresu straty.
        accuracy = [float(row["accuracy"]) for row in data]  # Ekstrahuje wartości dokładności treningowej, seria danych dla wykresu dokładności.
        val_accuracy = [float(row["val_accuracy"]) for row in data]  # Ekstrahuje wartości dokładności walidacyjnej, seria porównawcza dla wykresu dokładności.

        plt.figure(figsize=(12, 4))  # Inicjalizuje nowe okno wykresu o wymiarach 12x4 cala, kontener dla subplotów.

        # Wykres funkcji straty (Loss).
        # Spadek straty świadczy o tym, że model uczy się minimalizować błąd.
        plt.subplot(1, 2, 1)  # Tworzy pierwszy subplot w układzie 1x2, miejsce na wykres straty.
        plt.plot(epochs, loss, label="Training Loss")  # Rysuje linię straty treningowej, wizualizacja błędu na zbiorze uczącym.
        plt.plot(epochs, val_loss, label="Validation Loss")  # Rysuje linię straty walidacyjnej, wizualizacja błędu na zbiorze testowym.
        plt.xlabel("Epoch")  # Podpisuje oś X jako "Epoch", informacja o jednostce czasu treningu.
        plt.ylabel("Loss")  # Podpisuje oś Y jako "Loss", informacja o mierzonej wartości.
        plt.legend()  # Wyświetla legendę wykresu, pozwala rozróżnić serie danych.

        # Wykres dokładności (Accuracy).
        # Wzrost dokładności świadczy o poprawnym generalizowaniu wiedzy.
        plt.subplot(1, 2, 2)  # Tworzy drugi subplot w układzie 1x2, miejsce na wykres dokładności.
        plt.plot(epochs, accuracy, label="Training Accuracy")  # Rysuje linię dokładności treningowej, wizualizacja skuteczności na zbiorze uczącym.
        plt.plot(epochs, val_accuracy, label="Validation Accuracy")  # Rysuje linię dokładności walidacyjnej, wizualizacja skuteczności na zbiorze testowym.
        plt.xlabel("Epoch")  # Podpisuje oś X jako "Epoch", informacja o jednostce czasu treningu.
        plt.ylabel("Accuracy")  # Podpisuje oś Y jako "Accuracy", informacja o mierzonej wartości.
        plt.legend()  # Wyświetla legendę wykresu, pozwala rozróżnić serie danych.

        plt.tight_layout()  # Automatycznie dopasowuje odstępy między wykresami, poprawia czytelność layoutu.
        plt.show()  # Wyświetla okno z wykresami, blokuje wykonanie do momentu zamknięcia okna.
    else:  # Blok else wykonuje się, gdy argument nie jest flagą "--plot", zakłada, że podano ścieżkę do pliku obrazu.
        # Tryb predykcji: Przetwarzanie pojedynczego obrazu.
        
        # Wczytanie obrazu i konwersja na skalę szarości ("L").
        # Jest to kluczowe, ponieważ model był uczony na obrazach jednokanałowych.
        img = Image.open(image_path).convert("L")  # Otwiera plik graficzny i konwertuje na odcienie szarości, przygotowanie wstępne danych wejściowych.
        
        # Skalowanie do wymiaru 28x28, zgodnego z wejściem sieci.
        img = img.resize((28, 28))  # Zmienia rozmiar obrazu na 28x28 pikseli, dopasowanie do warstwy wejściowej modelu.
        
        # Konwersja na macierz NumPy i normalizacja (0-1).
        img_array = np.array(img) / 255.0  # Konwertuje obiekt obrazu na tablicę i normalizuje wartości, przygotowanie numeryczne danych.
        
        # Dodanie wymiaru batcha (axis=0).
        # Model oczekuje wejścia w formacie (Batch_Size, Height, Width).
        # Tutaj tworzymy batch jednoelementowy: (1, 28, 28).
        img_array = np.expand_dims(img_array, axis=0)  # Dodaje dodatkowy wymiar na początku tablicy, tworzy batch o rozmiarze 1 dla modelu.

        # Wykonanie inferencji.
        predictions = model.predict(img_array, verbose=0)  # Uruchamia model na przygotowanym obrazie, zwraca surowe wyniki predykcji (prawdopodobieństwa).
        
        # Wynik to tablica 10 prawdopodobieństw.
        # argmax zwraca indeks największej wartości, który odpowiada przewidzianej cyfrze.
        digit = np.argmax(predictions[0])  # Znajduje indeks elementu z najwyższym prawdopodobieństwem, interpretuje wynik jako konkretną cyfrę.

        print(f"Predicted digit: {digit}")  # Wypisuje wynik predykcji na konsolę, informacja końcowa dla użytkownika.
