import csv
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ==============================================================================
# 1. Wczytanie i przygotowanie danych
# ==============================================================================

# Definiowanie listy nazw kolumn dla zbioru danych Wine
cols = [
    "class",                 # Kolumna 0: Klasa wina (etykieta, target).
    "alcohol",               # Kolumna 1: Zawartość alkoholu.
    "malic_acid",            # Kolumna 2: Kwas jabłkowy.
    "ash",                   # Kolumna 3: Popiół.
    "alcalinity_of_ash",     # Kolumna 4: Zasadowość popiołu.
    "magnesium",             # Kolumna 5: Magnez.
    "total_phenols",         # Kolumna 6: Całkowite fenole.
    "flavanoids",            # Kolumna 7: Flawonoidy.
    "nonflavanoid_phenols",  # Kolumna 8: Fenole nieflawonoidowe.
    "proanthocyanins",       # Kolumna 9: Proantocyjanidyny.
    "color_intensity",       # Intensywność koloru.
    "hue",                   # Kolumna 11: Odcień.
    "od280_od315",           # Kolumna 12: OD280/OD315 win rozcieńczonych.
    "proline",               # Kolumna 13: Prolina.
]

# Wczytanie danych z pliku CSV bez nagłówka.
data = pd.read_csv("wine.csv", header=None, names=cols)  # Wczytuje plik 'wine.csv' do DataFrame, nadając nazwy kolumn.

# Tasowanie (Shuffling)
# Losowe przemieszanie wierszy (frac=1.0 = 100% danych).
# Ustawienie ziarna losowości (random_state=42) zapewnia powtarzalność eksperymentu.
data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)  # Miesza dane losowo i resetuje indeksy.

# Podział na cechy (X) i etykiety (y)
# X: wszystkie kolumny poza "class", konwertowane na float32.
X = data.drop("class", axis=1).values.astype("float32")  # Wyodrębnia macierz cech, usuwając kolumnę klasy.

# y: kolumna "class", wartości 1, 2, 3.
y_int = data["class"].values.astype("int32")  # Wyodrębnia wektor etykiet klas jako liczby całkowite.

# Kodowanie One-hot dla etykiet
# Zmieniamy zakres klas z [1, 2, 3] na [0, 1, 2] poprzez odjęcie 1.
# Wynik to macierz, gdzie np. klasa 1 to [1, 0, 0].
num_classes = 3  # Określa liczbę klas w problemie klasyfikacji.
y = tf.keras.utils.to_categorical(y_int - 1, num_classes=num_classes)  # Konwertuje etykiety na format one-hot encoding.

# Podział na zbiór treningowy i testowy (80% / 20%)
n = X.shape[0]  # Pobiera całkowitą liczbę próbek w zbiorze danych.
split = int(0.8 * n)  # Oblicza indeks podziału (80% danych treningowych).

X_train, X_test = X[:split], X[split:]  # Dzieli dane wejściowe na zbiór treningowy i testowy.
y_train, y_test = y[:split], y[split:]  # Dzieli etykiety na zbiór treningowy i testowy.

# Standaryzacja cech (Feature Scaling)
# Obliczamy średnią i odchylenie standardowe tylko na zbiorze treningowym,
# aby uniknąć wycieku informacji ze zbioru testowego.
mean = X_train.mean(axis=0)  # Oblicza średnią dla każdej cechy w zbiorze treningowym.
std = X_train.std(axis=0)  # Oblicza odchylenie standardowe dla każdej cechy w zbiorze treningowym.

# Standaryzacja: (wartość - średnia) / odchylenie standardowe.
X_train = (X_train - mean) / std  # Normalizuje zbiór treningowy.
X_test = (X_test - mean) / std  # Normalizuje zbiór testowy tymi samymi parametrami.

# Zapisanie parametrów standaryzacji do późniejszego wykorzystania przy predykcji.
np.save("wine_mean.npy", mean)  # Zapisuje wektor średnich do pliku .npy.
np.save("wine_std.npy", std)  # Zapisuje wektor odchyleń standardowych do pliku .npy.


# ==============================================================================
# 2. Konfiguracja procesu uczenia
# ==============================================================================

EPOCHS = 100  # Definiuje liczbę epok treningu.
BATCH_SIZE = 16  # Definiuje rozmiar batcha (liczba próbek przetwarzanych naraz).
LEARNING_RATE = 0.001  # Definiuje współczynnik uczenia dla optymalizatora.

# Konfiguracja TensorBoard
# Tworzenie unikalnego katalogu dla logów na podstawie daty i czasu.
log_dir = os.path.join(  # Tworzy ścieżkę do katalogu logów.
    "logs",  # Katalog główny logów.
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Podkatalog z aktualną datą i czasem.
)
tensorboard_cb = TensorBoard(log_dir=log_dir)  # Inicjalizuje callback TensorBoard z przygotowaną ścieżką.


# ==============================================================================
# 3. Model 1 – ReLU + HeUniform
# ==============================================================================

model1 = Sequential(name="Model_dense_relu")  # Tworzy pierwszy model sekwencyjny o nazwie 'Model_dense_relu'.

# Warstwa ukryta 1: 64 neurony, ReLU, inicjalizacja He (optymalna dla ReLU).
model1.add(Dense(64, activation="relu", kernel_initializer="he_uniform", name="hidden_1", input_shape=(13,)))  # Dodaje pierwszą warstwę gęstą.

# Warstwa ukryta 2: 32 neurony.
model1.add(Dense(32, activation="relu", kernel_initializer="he_uniform", name="hidden_2"))  # Dodaje drugą warstwę gęstą.

# Warstwa wyjściowa: 3 neurony (klasy), Softmax (prawdopodobieństwa).
model1.add(Dense(num_classes, activation="softmax", kernel_initializer="glorot_uniform", name="output"))  # Dodaje warstwę wyjściową.

model1.compile(  # Kompiluje model 1.
    optimizer=Adam(learning_rate=LEARNING_RATE),  # Ustawia optymalizator Adam.
    loss="categorical_crossentropy",  # Ustawia funkcję straty dla one-hot encoding.
    metrics=["accuracy"],  # Monitoruje dokładność.
)

# Trenowanie Modelu 1
history1 = model1.fit(  # Rozpoczyna trening modelu 1.
    X_train, y_train,  # Dane treningowe i etykiety.
    validation_data=(X_test, y_test),  # Dane walidacyjne.
    epochs=EPOCHS,  # Liczba epok.
    batch_size=BATCH_SIZE,  # Rozmiar batcha.
    callbacks=[tensorboard_cb],  # Użyte callbacki (TensorBoard).
    verbose=0  # Wyłącza pasek postępu (tryb cichy).
)


# ==============================================================================
# 4. Model 2 – tanh + GlorotNormal
# ==============================================================================

model2 = Sequential(name="Model_tanh_style")  # Tworzy drugi model sekwencyjny.

# Warstwa ukryta 1: 128 neuronów, tanh, inicjalizacja Glorot Normal (Xavier).
model2.add(Dense(128, activation="tanh", kernel_initializer="glorot_normal", name="hidden_1", input_shape=(13,)))  # Dodaje pierwszą warstwę modelu 2.

# Warstwa ukryta 2: 32 neurony.
model2.add(Dense(32, activation="tanh", kernel_initializer="glorot_normal", name="hidden_2"))  # Dodaje drugą warstwę modelu 2.

# Warstwa wyjściowa.
model2.add(Dense(num_classes, activation="softmax", kernel_initializer="glorot_uniform", name="output"))  # Dodaje warstwę wyjściową modelu 2.

model2.compile(  # Kompiluje model 2.
    optimizer=Adam(learning_rate=LEARNING_RATE),  # Ustawia optymalizator Adam.
    loss="categorical_crossentropy",  # Ustawia funkcję straty.
    metrics=["accuracy"],  # Monitoruje dokładność.
)

# Trenowanie Modelu 2
history2 = model2.fit(  # Rozpoczyna trening modelu 2.
    X_train, y_train,  # Dane treningowe.
    validation_data=(X_test, y_test),  # Dane walidacyjne.
    epochs=EPOCHS,  # Liczba epok.
    batch_size=BATCH_SIZE,  # Rozmiar batcha.
    callbacks=[tensorboard_cb],  # Callbacki.
    verbose=0  # Tryb cichy.
)


# ==============================================================================
# 5. Ewaluacja i Zapis
# ==============================================================================

# Ocena modeli na zbiorze testowym
test_loss1, test_acc1 = model1.evaluate(X_test, y_test, verbose=0)  # Ocenia model 1 na zbiorze testowym.
test_loss2, test_acc2 = model2.evaluate(X_test, y_test, verbose=0)  # Ocenia model 2 na zbiorze testowym.

print("Model 1:", model1.name, "acc =", round(test_acc1, 4))  # Wyświetla dokładność modelu 1.
print("Model 2:", model2.name, "acc =", round(test_acc2, 4))  # Wyświetla dokładność modelu 2.

# Wybór i zapis lepszego modelu
if test_acc1 >= test_acc2:  # Porównuje dokładności modeli.
    best_model = model1  # Wybiera model 1 jeśli lepszy lub równy.
else:
    best_model = model2  # Wybiera model 2 jeśli lepszy.

best_model.save("wine_best_model.keras")  # Zapisuje najlepszy model do pliku.
print("Zapisano najlepszy model jako wine_best_model.keras")  # Informuje o zapisie.

# Zapis krzywych uczenia do pliku CSV
with open("learning_curve.csv", "w", newline="") as f:  # Otwiera plik CSV do zapisu logów.
    writer = csv.writer(f)  # Tworzy obiekt writera CSV.
    writer.writerow(["accuracy", "loss", "val_accuracy", "val_loss"])  # Zapisuje nagłówek pliku CSV.

    for i in range(EPOCHS):  # Iteruje po epokach.
        a = history1.history["accuracy"][i]  # Pobiera dokładność treningową z historii.
        l = history1.history["loss"][i]  # Pobiera stratę treningową z historii.
        va = history1.history["val_accuracy"][i]  # Pobiera dokładność walidacyjną.
        vl = history1.history["val_loss"][i]  # Pobiera stratę walidacyjną.
        writer.writerow([a, l, va, vl])  # Zapisuje wiersz danych do pliku CSV.


def plot_history(hist, title_prefix):  # Definiuje funkcję do rysowania wykresów historii.
    # Wykres Dokładności (Accuracy)
    plt.figure()  # Tworzy nową figurę dla wykresu.
    plt.plot(hist.history["accuracy"], label="train_acc")  # Rysuje linię dokładności treningowej.
    plt.plot(hist.history["val_accuracy"], label="val_acc")  # Rysuje linię dokładności walidacyjnej.
    plt.xlabel("Epoka")  # Opisuje oś X.
    plt.ylabel("Dokładność")  # Opisuje oś Y.
    plt.title(f"{title_prefix} - accuracy")  # Nadaje tytuł wykresowi.
    plt.legend()  # Dodaje legendę.
    plt.tight_layout()  # Dopasowuje układ.

    # Wykres Straty (Loss)
    plt.figure()  # Tworzy nową figurę.
    plt.plot(hist.history["loss"], label="train_loss")  # Rysuje linię straty treningowej.
    plt.plot(hist.history["val_loss"], label="val_loss")  # Rysuje linię straty walidacyjnej.
    plt.xlabel("Epoka")  # Opisuje oś X.
    plt.ylabel("Strata")  # Opisuje oś Y.
    plt.title(f"{title_prefix} - loss")  # Nadaje tytuł.
    plt.legend()  # Dodaje legendę.
    plt.tight_layout()  # Dopasowuje układ.
    plt.show()  # Wyświetla wykresy.


# Wyświetlenie wykresów
plot_history(history1, "Model 1")  # Rysuje wykresy dla modelu 1.
plot_history(history2, "Model 2")  # Rysuje wykresy dla modelu 2.
