import os

import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Normalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Lista nazw kolumn (zgodna z dokumentacją zbioru danych).
COLUMNS = [
    "class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315", "proline"
]

# Definicja cech (wszystkie kolumny oprócz pierwszej "class").
FEATURES = COLUMNS[1:]  # Wybiera wszystkie elementy listy poza pierwszym, określając cechy wejściowe modelu.

# Dynamiczne ustalenie ścieżki do pliku CSV (z tego samego katalogu co skrypt).
csv_path = os.path.join(os.path.dirname(__file__), "wine.csv")  # Łączy ścieżkę katalogu skryptu z nazwą pliku, tworząc relatywną ścieżkę do danych.

# Wczytanie danych z CSV.
df = pd.read_csv(csv_path, header=None, names=COLUMNS)  # Wczytuje plik CSV do obiektu DataFrame, przypisując zdefiniowane nazwy kolumn.

# Tasowanie danych.
# Randomizacja kolejności próbek jest kluczowa dla stochastycznego spadku wzdłuż gradientu (SGD).
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # Miesza losowo wiersze ramki danych i resetuje indeksowanie.

# Ekstrakcja macierzy cech X.
X = df.drop("class", axis=1).values.astype("float32")  # Usuwa kolumnę 'class' i konwertuje resztę na macierz float32.

# Przygotowanie etykiet y.
# Konwersja z zakresu 1-3 na 0-2 dla one-hot encoding.
y = df["class"].values.astype("int32") - 1  # Pobiera kolumnę 'class' i przesuwa indeksację o -1 (zakres 0-2).
y = to_categorical(y, 3)  # Zamienia liczby całkowite na wektory one-hot encoding.

# Podział na zbiór treningowy (80%) i walidacyjny (20%).
val_split = int(0.8 * len(X))  # Oblicza punkt podziału zbioru danych (80% próbek).
X_train, X_val = X[:val_split], X[val_split:]  # Dzieli macierz cech na część treningową i walidacyjną.
y_train, y_val = y[:val_split], y[val_split:]  # Dzieli macierz etykiet na część treningową i walidacyjną.

# Utworzenie warstwy normalizacyjnej.
# W tym podejściu normalizacja jest częścią samego grafu modelu,
# co ułatwia wdrażanie (model "sam wie" jak znormalizować surowe dane).
normalizer = Normalization()  # Inicjalizuje warstwę normalizacyjną Keras.

# Adaptacja normalizatora do statystyk zbioru treningowego (obliczenie mean i variance).
normalizer.adapt(X_train)  # Oblicza średnią i wariancję z danych treningowych, kalibrując normalizację.

# Definicja architektury modelu (Baseline MLP).
model = Sequential([  # Inicjalizuje sekwencyjny model Keras.
    # Warstwa wejściowa - automatyczna normalizacja.
    normalizer,  # Dodaje warstwę normalizacji jako pierwszą operację w modelu.
    
    # Warstwa ukryta 1: 64 neurony, aktywacja ReLU.
    # Inicjalizator 'he_uniform' jest zalecany dla ReLU.
    Dense(64, activation='relu', kernel_initializer='he_uniform'),  # Dodaje warstwę gęstą z 64 neuronami i inicjalizacją He.
    
    # Warstwa ukryta 2: 32 neurony, aktywacja ReLU.
    Dense(32, activation='relu', kernel_initializer='he_uniform'),  # Dodaje drugą warstwę gęstą z 32 neuronami.
    
    # Warstwa wyjściowa: 3 klasy, aktywacja Softmax (rozkład prawdopodobieństwa).
    Dense(3, activation='softmax')  # Dodaje warstwę wyjściową z 3 neuronami i funkcją softmax.
])

# Kompilacja modelu.
model.compile(  # Konfiguruje proces uczenia modelu.
    optimizer=Adam(learning_rate=0.001),  # Ustawia optymalizator Adam ze stałym współczynnikiem uczenia.
    loss='categorical_crossentropy',  # Wybiera funkcję straty dla klasyfikacji wieloklasowej.
    metrics=['accuracy']  # Nakazuje monitorowanie dokładności podczas treningu.
)

print("Training Baseline Model...")  # Wypisuje komunikat o rozpoczęciu treningu.

# Trening modelu.
history = model.fit(  # Uruchamia pętlę treningową.
    X_train, y_train,  # Przekazuje dane treningowe.
    validation_data=(X_val, y_val),  # Przekazuje dane walidacyjne do oceny postępów.
    epochs=100,  # Ustawia liczbę epok na 100.
    batch_size=16,  # Ustawia rozmiar batcha na 16.
    verbose=0  # Wyłącza logowanie postępów w konsoli.
)

# Ewaluacja na zbiorze walidacyjnym.
loss, acc = model.evaluate(X_val, y_val, verbose=0)  # Oblicza stratę i dokładność na zbiorze walidacyjnym.
print(f"Baseline Accuracy: {acc:.4f}")  # Wypisuje osiągniętą dokładność modelu.
print(f"Baseline Loss: {loss:.4f}")  # Wypisuje osiągniętą stratę modelu.

# Zapisanie modelu w formacie Keras.
model.save("wine_baseline.keras")  # Zapisuje wytrenowany model do pliku.
print("Saved baseline model to wine_baseline.keras")  # Potwierdza zapisanie modelu.

# Zapis wyników do pliku tekstowego dla łatwego porównania.
with open("baseline_result.txt", "w") as f:  # Otwiera plik tekstowy do zapisu wyników.
    f.write(f"Accuracy: {acc:.4f}\nLoss: {loss:.4f}")  # Zapisuje metryki w formacie tekstowym.
