
import pandas as pd  # Import biblioteki pandas (typ: module) - do analizy danych
import numpy as np  # Import biblioteki NumPy (typ: module) - do obliczeń numerycznych
import tensorflow as tf  # Import biblioteki TensorFlow (typ: module) - framework uczenia maszynowego
from tensorflow.keras import Sequential  # Import klasy Sequential (typ: class) - prosty stos warstw
from tensorflow.keras.layers import Dense, Normalization  # Import warstw Dense i Normalization (typ: class, class)
from tensorflow.keras.optimizers import Adam  # Import optymalizatora Adam (typ: class)
from tensorflow.keras.utils import to_categorical  # Import funkcji do one-hot encoding (typ: function)
import os  # Import modułu os (typ: module) - operacje na systemie plików

# 1. Load Data
# Definicja listy nazw wszystkich kolumn w pliku CSV (typ: list[str])
COLUMNS = [
    "class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315", "proline"
]

# Definicja cech (wszystkie kolumny oprócz pierwszej "class") (typ: list[str])
# Używamy tego samego nazewnictwa co w wine_predict.py dla spójności
FEATURES = COLUMNS[1:]

# Ustalenie ścieżki do pliku CSV względem bieżącego skryptu
csv_path = os.path.join(os.path.dirname(__file__), "wine.csv")  # (typ: str)

# Wczytanie danych z CSV do DataFrame (typ: pandas.DataFrame)
df = pd.read_csv(csv_path, header=None, names=COLUMNS)

# Przetasowanie danych losowo i reset indeksu
# frac=1.0 oznacza wzięcie 100% wierszy
# reset_index(drop=True) usuwa stary indeks
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # (typ: pandas.DataFrame)

# Oddzielenie cech (X) od etykiet (y)
# drop("class") usuwa kolumnę z klasą
# .values konwertuje na tablicę NumPy
X = df.drop("class", axis=1).values.astype("float32")  # (typ: np.ndarray, dtype=float32)

# Przygotowanie etykiet
# Odejmowanie 1, aby klasy 1,2,3 zamienić na 0,1,2
y = df["class"].values.astype("int32") - 1  # (typ: np.ndarray, dtype=int32)
# Konwersja na one-hot encoding (np. 0 -> [1, 0, 0])
y = to_categorical(y, 3)  # (typ: np.ndarray, dtype=float32)

# Podział na zbiór treningowy i walidacyjny (80% / 20%)
val_split = int(0.8 * len(X))  # Obliczenie indeksu podziału (typ: int)
X_train, X_val = X[:val_split], X[val_split:]  # Slicing tablicy X (typ: np.ndarray, np.ndarray)
y_train, y_val = y[:val_split], y[val_split:]  # Slicing tablicy y (typ: np.ndarray, np.ndarray)

# 2. Normalization Layer
# Utworzenie warstwy normalizacyjnej, która będzie częścią modelu
normalizer = Normalization()  # (typ: keras.layers.preprocessing.normalization.Normalization)
# Dopasowanie (adaptacja) warstwy do danych treningowych (obliczenie średniej i odchylenia)
normalizer.adapt(X_train)  # (metoda zwraca None)

# 3. Create Model
# Definicja modelu sekwencyjnego
model = Sequential([
    normalizer,  # Pierwsza warstwa to normalizacja (aplikowana automatycznie na wejściu)
    Dense(64, activation='relu', kernel_initializer='he_uniform'),  # Warstwa ukryta 64 neurony, ReLu, inicjalizacja He
    Dense(32, activation='relu', kernel_initializer='he_uniform'),  # Warstwa ukryta 32 neurony
    Dense(3, activation='softmax')  # Warstwa wyjściowa 3 neurony (klasy), Softmax (prawdopodobieństwo)
])  # (typ: keras.engine.sequential.Sequential)

# Kompilacja modelu
model.compile(optimizer=Adam(learning_rate=0.001),  # Optymalizator Adam z LR=0.001
              loss='categorical_crossentropy',  # Funkcja straty dla klasyfikacji wieloklasowej
              metrics=['accuracy'])  # Metryka do monitorowania

# 4. Train
print("Training Baseline Model...")  # (typ: None)
# Trening modelu
history = model.fit(X_train, y_train,  # Dane treningowe
                    validation_data=(X_val, y_val),  # Dane walidacyjne
                    epochs=100,  # Liczba epok
                    batch_size=16,  # Rozmiar partii
                    verbose=0)  # Wyłączenie logowania w konsoli (typ: keras.callbacks.History)

# Ocena modelu na zbiorze walidacyjnym
loss, acc = model.evaluate(X_val, y_val, verbose=0)  # (zwraca tuple: float, float)
print(f"Baseline Accuracy: {acc:.4f}")  # Wyświetlenie dokładności (typ: str)
print(f"Baseline Loss: {loss:.4f}")  # Wyświetlenie straty (typ: str)

# Zapisanie modelu (ważne dla Cwiczenia4/wine_predict.py)
model.save("wine_baseline.keras")
print("Saved baseline model to wine_baseline.keras")

# Zapis wyników bazowych do pliku tekstowego
with open("baseline_result.txt", "w") as f:  # (kontekst menedżera pliku)
    f.write(f"Accuracy: {acc:.4f}\nLoss: {loss:.4f}")  # Zapis danych (typ: int - liczba znaków)
