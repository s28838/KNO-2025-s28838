
import pandas as pd  # Import pandas (typ: module) - analiza danych tabelarycznych
import numpy as np  # Import NumPy (typ: module) - operacje macierzowe
import tensorflow as tf  # Import TensorFlow (typ: module) - framework ML
import keras_tuner as kt  # Import Keras Tuner (typ: module) - biblioteka do optymalizacji hiperparametrów
from tensorflow.keras import Sequential  # Import klasy Sequential (typ: class)
from tensorflow.keras.layers import Dense, Normalization  # Import warstw neuronowych i normalizacyjnych (typ: class, class)
from tensorflow.keras.optimizers import Adam  # Import optymalizatora (typ: class)
from tensorflow.keras.utils import to_categorical  # Import utility do kodowania etykiet (typ: function)
from sklearn.metrics import confusion_matrix, classification_report  # Import metryk ewaluacji (typ: function, function)
import os  # Import modułu os (typ: module) - obsługa ścieżek

# 1. Load Data
# Definicja nazw kolumn datasetu (typ: list[str])
cols = [
    "class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315", "proline"
]

# Budowanie ścieżki do pliku csv niezależnie od katalogu roboczego (typ: str)
csv_path = os.path.join(os.path.dirname(__file__), "wine.csv")
# Wczytanie pliku CSV (typ: pandas.DataFrame)
df = pd.read_csv(csv_path, header=None, names=cols)
# Losowe tasowanie danych (frac=1.0) i reset indeksu (typ: pandas.DataFrame)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Konwersja cech na float32 (typ: np.ndarray)
X = df.drop("class", axis=1).values.astype("float32")
# Przesunięcie etykiet z 1-3 na 0-2 (typ: np.ndarray)
y = df["class"].values.astype("int32") - 1
# One-hot encoding etykiet (typ: np.ndarray)
y = to_categorical(y, 3)

# Obliczenie punktu podziału (80% trening, 20% walidacja) (typ: int)
val_split = int(0.8 * len(X))
# Podział danych (typ: np.ndarray, np.ndarray)
X_train, X_val = X[:val_split], X[val_split:]
# Podział etykiet (typ: np.ndarray, np.ndarray)
y_train, y_val = y[:val_split], y[val_split:]

# 2. Prepare Normalization
# Inicjalizacja warstwy normalizacji (typ: tensorflow.keras.layers.Normalization)
normalizer = Normalization()
# Dopasowanie normalizatora do danych treningowych (obliczenie średniej i odchylenia)
normalizer.adapt(X_train)

# 3. Model Builder
# Funkcja budująca model, przyjmuje hiperparametry (hp)
def build_model(hp):
    # Inicjalizacja pustego modelu sekwencyjnego (typ: tensorflow.keras.Sequential)
    model = Sequential()
    # Dodanie warstwy normalizacji jako pierwszej (typ: None)
    model.add(normalizer)
    
    # Hiperparametr: Liczba warstw ukrytych (int od 1 do 3)
    # Pętla dodająca dynamicznie warstwy Dense
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(
            # Hiperparametr: liczba neuronów w warstwie i-tej (od 16 do 128 co 16)
            units=hp.Int(f'units_{i}', min_value=16, max_value=128, step=16),
            # Hiperparametr: funkcja aktywacji (wybór między 'relu' a 'tanh')
            activation=hp.Choice('activation', ['relu', 'tanh']),
            # Dynamiczny dobór inicjalizatora wag w zależności od funkcji aktywacji
            # He dla ReLU, Glorot (Xavier) dla innych (np. tanh)
            kernel_initializer='he_uniform' if hp.get('activation') == 'relu' else 'glorot_uniform'
        ))
    
    # Warstwa wyjściowa stała: 3 klasy, softmax (typ: None)
    model.add(Dense(3, activation='softmax'))
    
    # Hiperparametr: współczynnik uczenia (learning rate)
    # Logarytmiczne próbkowanie od 0.0001 do 0.01
    lr = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')
    
    # Kompilacja modelu z wylosowanym LR (typ: None)
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model # Zwraca skompilowany model Keras (typ: tensorflow.keras.Model)

# 4. Tuner
# Konfiguracja tunera RandomSearch
tuner = kt.RandomSearch(
    build_model,                     # Funkcja budująca model
    objective='val_accuracy',        # Metryka do optymalizacji (dokładność walidacyjna)
    max_trials=20,                   # Maksymalna liczba różnych kombinacji hiperparametrów do sprawdzenia
    executions_per_trial=1,          # Liczba uruchomień dla każdej kombinacji (aby uśrednić wynik)
    directory='kt_dir',              # Katalog na wyniki tunera
    project_name='wine_tuning_simple' # Nazwa projektu (podkatalogu)
)

# Wyświetlenie podsumowania przestrzeni poszukiwań (typ: None)
tuner.search_space_summary()

print("\nStarting search...") # (typ: None)
# Uruchomienie procesu przeszukiwania (typ: None)
tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), verbose=0)

# 5. Results
# Pobranie najlepszych hiperparametrów (zwraca listę, bierzemy pierwszy/najlepszy zestaw)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0] # (typ: keras_tuner.HyperParameters)

print("\nBest Hyperparameters:") # (typ: None)
# Wyświetlenie znalezionych najlepszych wartości
print(f"  Num Layers: {best_hps.get('num_layers')}") # (typ: None)
print(f"  Activation: {best_hps.get('activation')}") # (typ: None)
print(f"  Learning Rate: {best_hps.get('lr')}") # (typ: None)

# Retrain best model
print("\nRetraining best model...") # (typ: None)
# Zbudowanie modelu na nowo z najlepszymi parametrami (typ: tensorflow.keras.Model)
best_model = tuner.hypermodel.build(best_hps)
# Ponowne trenowanie najlepszego modelu, tym razem dłużej (100 epok)
history = best_model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=0) # (typ: keras.callbacks.History)

# Evaluate
# Ostateczna ocena na zbiorze walidacyjnym (typ: tuple[float, float])
loss, acc = best_model.evaluate(X_val, y_val, verbose=0)
print(f"\nBest Model Accuracy: {acc:.4f}") # (typ: None)

# Save
# Zapis modelu do pliku .keras (typ: None)
best_model.save("wine_tuned.keras")
print("Saved best model to wine_tuned.keras") # (typ: None)

# Confusion Matrix
# Predykcja na zbiorze walidacyjnym (zwraca prawdopodobieństwa) (typ: np.ndarray)
y_pred = best_model.predict(X_val)
# Wybór klasy o najwyższym prawdopodobieństwie (typ: np.ndarray)
y_pred_classes = np.argmax(y_pred, axis=1)
# Konwersja etykiet one-hot na numery klas (typ: np.ndarray)
y_true_classes = np.argmax(y_val, axis=1)

# Obliczenie macierzy pomyłek (typ: np.ndarray)
cm = confusion_matrix(y_true_classes, y_pred_classes)
print("\nConfusion Matrix:") # (typ: None)
print(cm) # (typ: None)

print("\nClassification Report:") # (typ: None)
# Generowanie raportu klasyfikacji (precyzja, czułość, F1) (typ: str)
print(classification_report(y_true_classes, y_pred_classes))
