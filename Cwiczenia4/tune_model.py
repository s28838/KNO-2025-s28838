import os

import keras_tuner as kt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Normalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Definicja nazw kolumn datasetu.
cols = [
    "class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315", "proline"
]

# Budowanie ścieżki do pliku csv.
csv_path = os.path.join(os.path.dirname(__file__), "wine.csv")  # Łączy ścieżkę bieżącą z nazwą pliku danych, lokalizacja zasobu.

# Wczytanie i przygotowanie danych (analogicznie do baseline.py).
df = pd.read_csv(csv_path, header=None, names=cols)  # Wczytuje dane z pliku CSV do ramki danych Pandas, nadając nazwy kolumnom.
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # Tasuje dane losowo dla zapewnienia reprezentatywności w podziałach.

X = df.drop("class", axis=1).values.astype("float32")  # Wyodrębnia cechy do macierzy X, usuwając kolumnę klasy.
y = df["class"].values.astype("int32") - 1  # Wyodrębnia etykiety, konwertując je na zakres od 0.
y = to_categorical(y, 3)  # Konwertuje etykiety liczbowe na reprezentację one-hot.

# Podział na trening i walidację (80/20).
val_split = int(0.8 * len(X))  # Oblicza indeks podziału zbioru na treningowy i walidacyjny.
X_train, X_val = X[:val_split], X[val_split:]  # Dzieli macierz cech na podzbiory.
y_train, y_val = y[:val_split], y[val_split:]  # Dzieli macierz etykiet na podzbiory.

# Normalizacja zintegrowana z modelem.
normalizer = Normalization()  # Inicjalizuje warstwę normalizacyjną.
normalizer.adapt(X_train)  # Kalibruje normalizator na podstawie danych treningowych.


def build_model(hp):  # Definiuje funkcję budującą model z hiperparametrami z obiektu hp.
    model = Sequential()  # Inicjalizuje pusty model sekwencyjny.
    
    # Warstwa normalizacyjna jako pierwsza.
    model.add(normalizer)  # Dodaje warstwę normalizacyjną na wejściu sieci.
    
    # Dynamiczne dodawanie warstw ukrytych.
    # Tuner zdecyduje ile razy pętla się wykona (1, 2 lub 3 razy).
    for i in range(hp.Int('num_layers', 1, 3)):  # Iteruje przez liczbę warstw wybraną przez tuner (od 1 do 3).
        
        # Wybór funkcji aktywacji.
        activation_choice = hp.Choice('activation', ['relu', 'tanh'])  # Losuje funkcję aktywacji z podanych opcji.
        
        # Dobór odpowiedniego inicjalizatora wag.
        # He dla ReLU, Glorot dla Tanh.
        if activation_choice == 'relu':  # Sprawdza czy wylosowano ReLU.
            init = 'he_uniform'  # Ustawia inicjalizator He Uniform dla ReLU.
        else:
            init = 'glorot_uniform'  # Ustawia inicjalizator Glorot Uniform dla Tanh.
            
        model.add(Dense(  # Dodaje warstwę gęstą do modelu.
            # Liczba neuronów w i-tej warstwie (step=16 oznacza kroki co 16).
            units=hp.Int(f'units_{i}', min_value=16, max_value=128, step=16),
            activation=activation_choice,  # Ustawia wybraną funkcję aktywacji.
            kernel_initializer=init  # Ustawia dobrany inicjalizator wag.
        ))
    
    # Warstwa wyjściowa (3 klasy).
    model.add(Dense(3, activation='softmax'))  # Dodaje warstwę wyjściową z softmax dla klasyfikacji.
    
    # Strojenie learning rate (skala logarytmiczna sprawdza rzędy wielkości).
    lr = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')  # Losuje współczynnik uczenia w skali logarytmicznej.
    
    model.compile(optimizer=Adam(learning_rate=lr),  # Kompiluje model z wylosowanym LR.
                  loss='categorical_crossentropy',  # Ustawia funkcję straty.
                  metrics=['accuracy'])  # Ustawia metrykę oceny.
    return model  # Zwraca skompilowany model.


# Konfiguracja algorytmu przeszukiwania (Random Search).
# Wykona 20 losowych prób, każdą trenując raz.
tuner = kt.RandomSearch(  # Inicjalizuje tuner RandomSearch.
    build_model,  # Przekazuje funkcję budującą model.
    objective='val_accuracy',  # Cel optymalizacji to dokładność walidacyjna.
    max_trials=20,  # Liczba próbnych konfiguracji.
    executions_per_trial=1,  # Liczba treningów dla każdej konfiguracji.
    directory='kt_dir',  # Katalog roboczy tunera.
    project_name='wine_tuning_simple',  # Nazwa projektu tunera.
    overwrite=True  # Nadpisuje poprzednie wyniki w katalogu.
)

tuner.search_space_summary()  # Wyświetla podsumowanie przestrzeni przeszukiwania.

print("\nStarting search...")  # Loguje rozpoczęcie procesu szukania.
# Start przeszukiwania. verbose=0 ukrywa logi każdej epoki.
tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), verbose=0)  # Uruchamia przeszukiwanie przestrzeni hiperparametrów.

# Pobranie najlepszych hiperparametrów.
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]  # Pobiera najlepszy zestaw hiperparametrów.

print("\nBest Hyperparameters:")  # Nagłówek sekcji wyników.
print(f"  Num Layers: {best_hps.get('num_layers')}")  # Wypisuje znalezioną liczbę warstw.
print(f"  Activation: {best_hps.get('activation')}")  # Wypisuje wybraną aktywację.
print(f"  Learning Rate: {best_hps.get('lr')}")  # Wypisuje znaleziony learning rate.

print("\nRetraining best model...")  # Loguje rozpoczęcie finalnego treningu.
# Ponowne zbudowanie modelu z najlepszymi parametrami.
best_model = tuner.hypermodel.build(best_hps)  # Buduje model używając najlepszych parametrów.

# Pełny trening najlepszego modelu (więcej epok dla zbieżności).
best_model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=0)  # Trenuje najlepszy model przez 100 epok.

# Ewaluacja.
loss, acc = best_model.evaluate(X_val, y_val, verbose=0)  # Ocenia model na zbiorze walidacyjnym.
print(f"\nBest Model Accuracy: {acc:.4f}")  # Wypisuje końcową dokładność.

# Zapis.
best_model.save("wine_tuned.keras")  # Zapisuje najlepszy model do pliku.
print("Saved best model to wine_tuned.keras")  # Potwierdza zapis.

# ==============================================================================
# Analiza Wyników
# ==============================================================================

# Generowanie macierzy pomyłek i raportu klasyfikacji.
y_pred = best_model.predict(X_val)  # Wykonuje predykcję na zbiorze walidacyjnym.
y_pred_classes = np.argmax(y_pred, axis=1)  # Konwertuje prawdopodobieństwa na indeksy klas.
y_true_classes = np.argmax(y_val, axis=1)  # Konwertuje one-hot na indeksy klas rzeczywistych.

cm = confusion_matrix(y_true_classes, y_pred_classes)  # Tworzy macierz pomyłek.
print("\nConfusion Matrix:")  # Nagłówek macierzy.
print(cm)  # Wypisuje macierz pomyłek.

print("\nClassification Report:")  # Nagłówek raportu klasyfikacji.
print(classification_report(y_true_classes, y_pred_classes))  # Generuje i wypisuje raport klasyfikacji.
