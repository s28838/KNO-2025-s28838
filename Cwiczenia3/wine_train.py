# wine_train.py

import numpy as np  # Import biblioteki NumPy (typ: module) - do operacji na macierzach i tablicach numerycznych
import pandas as pd  # Import biblioteki pandas (typ: module) - do wczytywania i manipulacji danymi w formie tabelarycznej (DataFrame)
import tensorflow as tf  # Import biblioteki TensorFlow (typ: module) - do uczenia maszynowego i głębokiego
from tensorflow.keras import Sequential  # Import klasy Sequential (typ: class) - do tworzenia modeli sekwencyjnych (warstwa po warstwie)
from tensorflow.keras.layers import Dense  # Import klasy Dense (typ: class) - warstwa w pełni połączona (każdy neuron z każdym)
from tensorflow.keras.optimizers import Adam  # Import optymalizatora Adam (typ: class) - algorytm optymalizacji gradientowej
from tensorflow.keras.callbacks import TensorBoard  # Import callbacka TensorBoard (typ: class) - do logowania metryk uczenia
import matplotlib.pyplot as plt  # Import modułu pyplot (typ: module) - do rysowania wykresów
import datetime  # Import modułu datetime (typ: module) - do obsługi daty i czasu (np. unikalne nazwy logów)
import os  # Import modułu os (typ: module) - do operacji systemowych (ścieżki plików)
import csv  # Import modułu csv (typ: module) - do zapisu plików CSV

# Wczytanie danych (ręcznie z CSV)
# Definiowanie listy nazw kolumn dla zbioru danych Wine (typ: list[str])
cols = [
    "class",                 # Kolumna 0: Klasa wina (etykieta)
    "alcohol",               # Kolumna 1: Zawartość alkoholu
    "malic_acid",            # Kolumna 2: Kwas jabłkowy
    "ash",                   # Kolumna 3: Popiół
    "alcalinity_of_ash",     # Kolumna 4: Zasadowość popiołu
    "magnesium",             # Kolumna 5: Magnez
    "total_phenols",         # Kolumna 6: Całkowite fenole
    "flavanoids",            # Kolumna 7: Flawonoidy
    "nonflavanoid_phenols",  # Kolumna 8: Fenole nieflawonoidowe
    "proanthocyanins",       # Kolumna 9: Proantocyjanidyny
    "color_intensity",       # Kolumna 10: Intensywność koloru
    "hue",                   # Kolumna 11: Odcień
    "od280_od315",           # Kolumna 12: OD280/OD315 win rozcieńczonych
    "proline",               # Kolumna 13: Prolina
]

# Wczytanie danych z pliku CSV bez nagłówka, przypisanie nazw kolumn z listy cols
# data to obiekt DataFrame biblioteki pandas
data = pd.read_csv("wine.csv", header=None, names=cols)  # (typ: pandas.core.frame.DataFrame)

# Tasowanie
# Losowe przemieszanie wierszy (frac=1.0 = 100% danych), z ustalonym ziarnem losowości (42) dla powtarzalności
# reset_index(drop=True) resetuje indeksy w przetasowanej ramce danych
data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)  # (typ: pandas.core.frame.DataFrame)

# Podział X / y
# Usunięcie kolumny "class" i przekształcenie pozostałych kolumn na tablicę numpy float32 (cechy wejściowe)
X = data.drop("class", axis=1).values.astype("float32")  # (typ: np.ndarray, kształt=(178, 13), dtype=float32)

# Wyodrębnienie kolumny "class" jako tablicy numpy int32 (etykiety: 1, 2, 3)
y_int = data["class"].values.astype("int32")  # (typ: np.ndarray, kształt=(178,), dtype=int32)

# One-hot (gorąca jedynka) dla 3 klas
# Liczba klas do klasyfikacji (wino 1, 2, 3)
num_classes = 3  # (typ: int)

# Konwersja etykiet liczb całkowitych (1,2,3) na format one-hot: [1,0,0], [0,1,0], [0,0,1]
# Odejmowanie 1 (y_int - 1), aby zmienić zakres klas z [1,2,3] na [0,1,2] (indeksowanie od zera)
y = tf.keras.utils.to_categorical(y_int - 1, num_classes=num_classes)  # (typ: np.ndarray, kształt=(178, 3), dtype=float32)

# Podział na train / test (80/20)
# Pobranie liczby wszystkich próbek (wierszy)
n = X.shape[0]  # (typ: int)

# Obliczenie indeksu podziału (80% danych dla treningu)
split = int(0.8 * n)  # (typ: int)

# Podział cech X na zbiór treningowy (pierwsze 80% wierszy) i testowy (pozostałe 20%)
X_train, X_test = X[:split], X[split:]  # (typ: np.ndarray, np.ndarray)

# Podział etykiet y na zbiór treningowy i testowy w tym samym punkcie
y_train, y_test = y[:split], y[split:]  # (typ: np.ndarray, np.ndarray)

# Standaryzacja cech na podstawie zbioru treningowego
# Obliczenie średniej dla każdej cechy (kolumny) w zbiorze treningowym
mean = X_train.mean(axis=0)  # (typ: np.ndarray, kształt=(13,), dtype=float32)

# Obliczenie odchylenia standardowego dla każdej cechy w zbiorze treningowym
std = X_train.std(axis=0)  # (typ: np.ndarray, kształt=(13,), dtype=float32)

# Standaryzacja zbioru treningowego: (wartość - średnia) / odchylenie standardowe
# Operacja jest wykonywana element po elemencie (broadcasting)
X_train = (X_train - mean) / std  # (typ: np.ndarray)

# Standaryzacja zbioru testowego używając parametrów (średniej i odchylenia) ze zbioru treningowego
# Ważne: nie obliczamy średniej/std na zbiorze testowym, aby uniknąć wycieku danych!
X_test = (X_test - mean) / std  # (typ: np.ndarray)

# Zapis średniej do pliku binarnego NumPy (potrzebne później do predykcji na nowych danych)
np.save("wine_mean.npy", mean)  # (zapisuje plik .npy)
# Zapis odchylenia standardowego do pliku binarnego NumPy
np.save("wine_std.npy", std)  # (zapisuje plik .npy)

# Parametry uczenia
# Liczba epok uczenia - ile razy model przejdzie przez cały zbiór treningowy
EPOCHS = 100  # (typ: int)
# Rozmiar batcha - liczba próbek przetwarzanych w jednej iteracji przed aktualizacją wag
BATCH_SIZE = 16  # (typ: int)
# Współczynnik uczenia - determinuje wielkość kroku w algorytmie gradientowym
LEARNING_RATE = 0.001  # (typ: float)

# Utworzenie ścieżki do katalogu logów z aktualną datą i czasem (dla TensorBoard)
log_dir = os.path.join(
    "logs",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Tworzy string np. "20230101-120000" (typ: str)
) # (typ: str)

# Utworzenie callback'a TensorBoard do zapisywania metryk uczenia w czasie rzeczywistym
tensorboard_cb = TensorBoard(log_dir=log_dir)  # (typ: tensorflow.keras.callbacks.TensorBoard)

# ==============================================================================
# Model 1 – ReLU + HeUniform
# ==============================================================================
# Utworzenie pustego modelu sekwencyjnego
model1 = Sequential(name="Model_dense_relu")  # (typ: tensorflow.keras.Sequential)

# Dodanie pierwszej warstwy ukrytej:
# - 64 neurony (units=64)
# - Funkcja aktywacji ReLU (activation="relu")
# - Inicjalizacja wag metodą He Uniform (kernel_initializer="he_uniform") - dobra dla ReLU
# - input_shape=(13,) definiuje kształt danych wejściowych (13 cech)
model1.add(Dense(64, activation="relu", kernel_initializer="he_uniform", name="hidden_1", input_shape=(13,))) # (typ: None)

# Dodanie drugiej warstwy ukrytej: 32 neurony, ReLU, He Uniform
model1.add(Dense(32, activation="relu", kernel_initializer="he_uniform", name="hidden_2")) # (typ: None)

# Dodanie warstwy wyjściowej:
# - 3 neurony (num_classes=3) - po jednym dla każdej klasy
# - Funkcja aktywacji Softmax, która zwraca rozkład prawdopodobieństwa
# - Inicjalizacja Glorot Uniform
model1.add(Dense(num_classes, activation="softmax", kernel_initializer="glorot_uniform", name="output")) # (typ: None)

# Kompilacja modelu 1 - konfiguracja procesu treningu
model1.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE), # Optymalizator Adam (typ: tensorflow.keras.optimizers.Adam)
    loss="categorical_crossentropy",             # Funkcja kosztu dla klasyfikacji wieloklasowej (typ: str)
    metrics=["accuracy"],                        # Metryka do monitorowania (dokładność) (typ: list[str])
)

# Trenowanie modelu 1 na danych treningowych
history1 = model1.fit(
    X_train, y_train,                  # Dane treningowe (cechy i etykiety)
    validation_data=(X_test, y_test),  # Dane walidacyjne (do oceny modelu po każdej epoce)
    epochs=EPOCHS,                     # Liczba epok
    batch_size=BATCH_SIZE,             # Rozmiar partii danych
    callbacks=[tensorboard_cb],        # Callbacki (tutaj logowanie do TensorBoard)
    verbose=0                          # Wyłączenie standardowego paska postępu w konsoli (0=silent)
) # Zwraca obiekt History (typ: tensorflow.keras.callbacks.History)

# ==============================================================================
# Model 2 – tanh + GlorotNormal
# ==============================================================================
# Utworzenie drugiego modelu sekwencyjnego
model2 = Sequential(name="Model_tanh_style")  # (typ: tensorflow.keras.Sequential)

# Pierwsza warstwa ukryta:
# - 128 neuronów
# - Funkcja aktywacji tanh (tangens hiperboliczny)
# - Inicjalizacja wag metodą Glorot Normal (Xavier) - dobra dla tanh/sigmoid
model2.add(Dense(128, activation="tanh", kernel_initializer="glorot_normal", name="hidden_1", input_shape=(13,))) # (typ: None)

# Druga warstwa ukryta: 32 neurony, tanh, Glorot Normal
model2.add(Dense(32, activation="tanh", kernel_initializer="glorot_normal", name="hidden_2")) # (typ: None)

# Warstwa wyjściowa: 3 neurony (klasy), Softmax
model2.add(Dense(num_classes, activation="softmax", kernel_initializer="glorot_uniform", name="output")) # (typ: None)

# Kompilacja modelu 2 (te same parametry co Model 1 dla porównania)
model2.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE), # (typ: tensorflow.keras.optimizers.Adam)
    loss="categorical_crossentropy",             # (typ: str)
    metrics=["accuracy"],                        # (typ: list[str])
)

# Trenowanie modelu 2
history2 = model2.fit(
    X_train, y_train,                  # Dane treningowe
    validation_data=(X_test, y_test),  # Dane walidacyjne
    epochs=EPOCHS,                     # Liczba epok
    batch_size=BATCH_SIZE,             # Rozmiar batcha
    callbacks=[tensorboard_cb],        # Ten sam callback (logi trafią do tego samego katalogu, ale pod różnymi nazwami modeli)
    verbose=0                          # (typ: int)
) # (typ: tensorflow.keras.callbacks.History)

# Ewaluacja (Ocena końcowa)
# Ocena modelu 1 na zbiorze testowym - zwraca [strata, dokładność]
test_loss1, test_acc1 = model1.evaluate(X_test, y_test, verbose=0)  # (typ: float, float)
# Ocena modelu 2 na zbiorze testowym
test_loss2, test_acc2 = model2.evaluate(X_test, y_test, verbose=0)  # (typ: float, float)

# Wyświetlenie wyników modelu 1 w konsoli
# model1.name to nazwa nadana przy tworzeniu ("Model_dense_relu")
print("Model 1:", model1.name, "acc =", round(test_acc1, 4)) # (typ: None)
# Wyświetlenie wyników modelu 2
print("Model 2:", model2.name, "acc =", round(test_acc2, 4)) # (typ: None)

# Wybór lepszego modelu i zapis
# Porównanie dokładności na zbiorze testowym
if test_acc1 >= test_acc2:
    best_model = model1  # Przypisanie referencji do lepszego modelu (typ: tensorflow.keras.Model)
else:
    best_model = model2  # (typ: tensorflow.keras.Model)

# Zapisanie całego modelu (architektura + wagi + konfiguracja optymalizatora) do pliku .keras
best_model.save("wine_best_model.keras")  # (zapis przy użyciu biblioteki Keras)
print("Zapisano najlepszy model jako wine_best_model.keras")  # (typ: None)

# Zapis krzywych uczenia do pliku CSV (do analizy poza Pythonem)
# Otwarcie pliku w trybie zapisu tekstu ('w')
with open("learning_curve.csv", "w", newline="") as f: # (typ kontekstu: io.TextIOWrapper)
    writer = csv.writer(f)  # Utworzenie obiektu piszącego CSV (typ: _csv.writer)
    # Zapis nagłówka kolumn
    writer.writerow(["accuracy", "loss", "val_accuracy", "val_loss"]) # (typ: None)
    
    # Iteracja przez wszystkie epoki
    for i in range(EPOCHS):
        # Pobranie wartości metryk dla danej epoki ze słownika history.history
        # history1.history to słownik list (typ: dict[str, list[float]])
        a = history1.history["accuracy"][i]       # Dokładność treningowa (typ: float)
        l = history1.history["loss"][i]           # Strata treningowa (typ: float)
        va = history1.history["val_accuracy"][i]  # Dokładność walidacyjna (typ: float)
        vl = history1.history["val_loss"][i]      # Strata walidacyjna (typ: float)
        
        # Zapis wiersza do pliku
        writer.writerow([a, l, va, vl]) # (typ: None)

# Funkcja pomocnicza do rysowania wykresów
def plot_history(hist, title_prefix):
    # Rysuje wykresy dokładności i straty dla obiektu historii
    # hist: obiekt History zwrócony przez model.fit
    # title_prefix: prefiks tytułu wykresu (typ: str)
    
    # Wykres Dokładności (Accuracy)
    plt.figure()  # Utworzenie nowego okna/obszaru wykresu (typ: matplotlib.figure.Figure)
    plt.plot(hist.history["accuracy"], label="train_acc")  # Linia dokładności treningowej (typ: list[float])
    plt.plot(hist.history["val_accuracy"], label="val_acc")  # Linia dokładności walidacyjnej (typ: list[float])
    plt.xlabel("Epoka")  # Opis osi X (typ: str)
    plt.ylabel("Dokładność")  # Opis osi Y (typ: str)
    plt.title(f"{title_prefix} - accuracy")  # Tytuł wykresu (typ: str)
    plt.legend()  # Legenda (typ: None)
    plt.tight_layout()  # Dopasowanie układu (typ: None)

    # Wykres Straty (Loss)
    plt.figure()  # Nowe okno dla drugiego wykresu (typ: matplotlib.figure.Figure)
    plt.plot(hist.history["loss"], label="train_loss")  # Linia straty treningowej
    plt.plot(hist.history["val_loss"], label="val_loss")  # Linia straty walidacyjnej
    plt.xlabel("Epoka")  # Opis osi X
    plt.ylabel("Strata")  # Opis osi Y
    plt.title(f"{title_prefix} - loss")  # Tytuł wykresu
    plt.legend()  # Legenda
    plt.tight_layout()  # Dopasowanie układu
    plt.show()  # Wyświetlenie wykresów (typ: None)

# Wywołanie funkcji rysowania dla historii modelu 1
plot_history(history1, "Model 1") # (typ: None)
# Wywołanie funkcji rysowania dla historii modelu 2
plot_history(history2, "Model 2") # (typ: None)
