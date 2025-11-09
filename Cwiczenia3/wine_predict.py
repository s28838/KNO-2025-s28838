# wine_predict.py
# Przykład użycia:
# python wine_predict.py --alcohol 13.2 --malic_acid 1.7 --ash 2.3 --alcalinity_of_ash 16.8 --magnesium 100 --total_phenols 2.4 --flavanoids 2.0 --nonflavanoid_phenols 0.3 --proanthocyanins 1.7 --color_intensity 5.0 --hue 1.0 --od280_od315 3.0 --proline 1100

import argparse
import numpy as np
import tensorflow as tf

# Lista nazw wszystkich cech (features) używanych przez model
# Musi być w tej samej kolejności co podczas treningu
FEATURES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315",
    "proline",
]

# Definicja funkcji do parsowania argumentów z linii poleceń
def parse_args():
    # Utworzenie obiektu parsera z opisem programu
    parser = argparse.ArgumentParser(
        description="Klasyfikacja wina na podstawie cech chemicznych (UCI Wine)."
    )
    # Iteracja przez wszystkie cechy z listy FEATURES
    for feat in FEATURES:
        # Dodanie argumentu dla każdej cechy (np. --alcohol, --malic_acid)
        # type=float oznacza, że wartość musi być liczbą zmiennoprzecinkową
        # required=True oznacza, że argument jest obowiązkowy
        parser.add_argument(f"--{feat}", type=float, required=True)
    # Parsowanie argumentów z linii poleceń i zwrócenie obiektu z wartościami
    return parser.parse_args()

# Definicja głównej funkcji programu
def main():
    # Wywołanie funkcji parsującej argumenty i zapisanie wyników
    args = parse_args()
    # Utworzenie tablicy 2D (1 wiersz, 13 kolumn) z wartościami cech podanymi przez użytkownika
    # getattr(args, f) pobiera wartość atrybutu o nazwie f z obiektu args
    # Przykład: getattr(args, "alcohol") zwraca wartość argumentu --alcohol
    x = np.array([[getattr(args, f) for f in FEATURES]], dtype="float32")

    # Wczytanie średniej z pliku (obliczonej podczas treningu)
    mean = np.load("wine_mean.npy")
    # Wczytanie odchylenia standardowego z pliku (obliczonego podczas treningu)
    std = np.load("wine_std.npy")
    # Standaryzacja danych wejściowych używając tych samych parametrów co podczas treningu
    # (wartość - średnia) / odchylenie standardowe
    x = (x - mean) / std

    # Wczytanie wytrenowanego modelu z pliku
    model = tf.keras.models.load_model("wine_best_model.keras")
    # Wykonanie predykcji - model.predict zwraca tablicę prawdopodobieństw dla każdej klasy
    # verbose=0 wyłącza wyświetlanie paska postępu
    # [0] pobiera pierwszy (i jedyny) wiersz wyników, bo predykcja jest dla jednej próbki
    probs = model.predict(x, verbose=0)[0]
    # Znalezienie indeksu klasy z najwyższym prawdopodobieństwem (0, 1 lub 2)
    # Dodanie 1, aby przekonwertować indeks [0,1,2] na klasę [1,2,3]
    predicted_class = int(np.argmax(probs)) + 1

    # Wyświetlenie prawdopodobieństw dla wszystkich trzech klas
    print("Prawdopodobieństwa klas:", probs)
    # Wyświetlenie przewidywanej klasy wina (1, 2 lub 3)
    print("Przewidywana klasa wina:", predicted_class)

# Standardowy wzorzec Pythona - uruchomienie main() tylko gdy skrypt jest wykonywany bezpośrednio
# (nie gdy jest importowany jako moduł)
if __name__ == "__main__":
    # Wywołanie głównej funkcji programu
    main()