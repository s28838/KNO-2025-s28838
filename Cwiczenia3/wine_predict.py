# wine_predict.py
# Przykład użycia:
# python wine_predict.py --alcohol 13.2 --malic_acid 1.7 --ash 2.3 --alcalinity_of_ash 16.8 --magnesium 100 --total_phenols 2.4 --flavanoids 2.0 --nonflavanoid_phenols 0.3 --proanthocyanins 1.7 --color_intensity 5.0 --hue 1.0 --od280_od315 3.0 --proline 1100

import argparse  # Import modułu argparse (typ: module) - do parsowania argumentów z wiersza poleceń
import numpy as np  # Import modułu NumPy (typ: module) - do operacji na tablicach
import tensorflow as tf  # Import TensorFlow (typ: module) - do ładowania modelu i predykcji

# Lista nazw wszystkich cech (features) używanych przez model
# Musi być w tej samej kolejności co podczas treningu, aby dane trafiły do odpowiednich wejść sieci
FEATURES = [
    "alcohol",               # (typ: str)
    "malic_acid",            # (typ: str)
    "ash",                   # (typ: str)
    "alcalinity_of_ash",     # (typ: str)
    "magnesium",             # (typ: str)
    "total_phenols",         # (typ: str)
    "flavanoids",            # (typ: str)
    "nonflavanoid_phenols",  # (typ: str)
    "proanthocyanins",       # (typ: str)
    "color_intensity",       # (typ: str)
    "hue",                   # (typ: str)
    "od280_od315",           # (typ: str)
    "proline",               # (typ: str)
]

# Definicja funkcji do parsowania argumentów z linii poleceń
def parse_args():
    # Utworzenie obiektu parsera z opisem programu
    parser = argparse.ArgumentParser(
        description="Klasyfikacja wina na podstawie cech chemicznych (UCI Wine)."
    ) # (typ: argparse.ArgumentParser)
    
    # Iteracja przez wszystkie cechy z listy FEATURES w celu dynamicznego dodania argumentów
    for feat in FEATURES:
        # Dodanie argumentu dla każdej cechy (np. --alcohol, --malic_acid)
        # f"--{feat}" tworzy nazwę flagi (typ: str)
        # type=float wymusza konwersję wejścia na liczbę zmiennoprzecinkową
        # required=True oznacza, że użytkownik musi podać ten parametr
        parser.add_argument(f"--{feat}", type=float, required=True) # (typ: None)
        
    # Parsowanie argumentów z linii poleceń i zwrócenie obiektu namespace z wartościami
    return parser.parse_args() # (typ: argparse.Namespace)

# Definicja głównej funkcji programu
def main():
    # Wywołanie funkcji parsującej argumenty i zapisanie wyników
    args = parse_args() # (typ: argparse.Namespace)
    
    # Utworzenie tablicy NumPy 2D z wartościami cech
    # Pobieramy wartości z obiektu args używając nazw z listy FEATURES
    # getattr(args, f) to dynamiczny dostęp do atrybutu o nazwie f
    # Lista składana tworzy listę wartości [val1, val2, ...]
    # [ ... ] wokół listy składanej tworzy listę list, co daje kształt (1, 13) po konwersji na array
    x = np.array([[getattr(args, f) for f in FEATURES]], dtype="float32") # (typ: np.ndarray, kształt=(1, 13))

    # Wczytanie średniej z pliku binarnego (obliczonej i zapisanej podczas treningu)
    mean = np.load("wine_mean.npy") # (typ: np.ndarray, kształt=(13,))
    
    # Wczytanie odchylenia standardowego z pliku
    std = np.load("wine_std.npy") # (typ: np.ndarray, kształt=(13,))
    
    # Standaryzacja danych wejściowych (z-score normalization)
    # Używamy tych samych parametrów co przy treningu sieci, aby dane były w tej samej skali
    # Broadcasting NumPy umożliwia operację na (1,13) i (13,)
    x = (x - mean) / std # (typ: np.ndarray, kształt=(1, 13))

    # Wczytanie wytrenowanego modelu z pliku .keras
    # Model musi być w tej samej lokalizacji lub podana pełna ścieżka
    model = tf.keras.models.load_model("wine_best_model.keras") # (typ: keras.engine.sequential.Sequential)
    
    # Wykonanie predykcji
    # model.predict zwraca tablicę prawdopodobieństw dla każdej klasy (shape (1, 3))
    # verbose=0 wyłącza logowanie postępu (przydatne w skryptach CLI)
    predictions = model.predict(x, verbose=0) # (typ: np.ndarray)
    
    # Pobranie pierwszego (i jedynego) wiersza wyników
    probs = predictions[0] # (typ: np.ndarray, kształt=(3,))
    
    # Znalezienie indeksu klasy z najwyższym prawdopodobieństwem (0, 1 lub 2)
    predicted_index = np.argmax(probs) # (typ: int64)
    
    # Przekonwertowanie indeksu [0,1,2] na numer klasy [1,2,3] (zgodnie z datasetem Wine)
    predicted_class = int(predicted_index) + 1 # (typ: int)

    # Wyświetlenie wektora prawdopodobieństw dla wszystkich trzech klas
    print("Prawdopodobieństwa klas:", probs) # (typ: None)
    
    # Wyświetlenie ostatecznej predykcji
    print("Przewidywana klasa wina:", predicted_class) # (typ: None)

# Standardowy blok uruchamiający funkcję main() tylko przy bezpośrednim wywołaniu skryptu
if __name__ == "__main__":
    main()