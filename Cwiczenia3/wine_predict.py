import argparse

import numpy as np
import tensorflow as tf

# Definicja listy cech (features), które model przyjmuje na wejściu.
# Kolejność musi być ściśle zachowana i zgodna z kolejnością kolumn w zbiorze treningowym,
# aby wagi neuronów odpowiadały właściwym parametrom.
FEATURES = [
    "alcohol",  # Zawartość alkoholu.
    "malic_acid",  # Zawartość kwasu jabłkowego.
    "ash",  # Zawartość popiołu.
    "alcalinity_of_ash",  # Zasadowość popiołu.
    "magnesium",  # Zawartość magnezu.
    "total_phenols",  # Całkowita zawartość fenoli.
    "flavanoids",  # Zawartość flawonoidów.
    "nonflavanoid_phenols",  # Zawartość fenoli nieflawonoidowych.
    "proanthocyanins",  # Zawartość proantocyjanidyn.
    "color_intensity",  # Intensywność koloru.
    "hue",  # Odcień wina.
    "od280_od315",  # Stosunek OD280/OD315 win rozcieńczonych.
    "proline",  # Zawartość proliny.
]


def parse_args():  # Funkcja parsująca argumenty przekazane przy uruchomieniu skryptu.
    parser = argparse.ArgumentParser(  # Tworzy obiekt parsera argumentów.
        description="Klasyfikacja wina na podstawie cech chemicznych (UCI Wine)."  # Ustawia opis programu widoczny w pomocy.
    )

    # Automatyczne dodawanie argumentów dla każdej cechy chemicznej.
    # Wymuszenie typu float zapewnia, że dane są liczbowe.
    # Ustawienie required=True gwarantuje, że użytkownik poda wszystkie niezbędne dane.
    for feat in FEATURES:  # Iteruje przez listę zdefiniowanych cech win.
        parser.add_argument(f"--{feat}", type=float, required=True)  # Dodaje wymagany argument (flota) dla danej cechy.

    return parser.parse_args()  # Parsuje i zwraca argumenty podane przez użytkownika.


def main():  # Główna funkcja programu, wykonuje całą logikę predykcji.
    args = parse_args()  # Pobiera sparsowane argumenty z linii poleceń.

    # Konstrukcja wektora wejściowego.
    # Tworzymy tablicę NumPy o kształcie (1, 13) - jeden wiersz, 13 kolumn.
    # Jest to format oczekiwany przez warstwę wejściową modelu Keras.
    x = np.array([[getattr(args, f) for f in FEATURES]], dtype="float32")  # Tworzy tablicę NumPy z wartości argumentów, typ float32.

    # Wczytanie parametrów statystycznych zbioru treningowego.
    # Model był uczony na danych znormalizowanych (Z-score), więc nowe dane
    # muszą zostać przekształcone w ten sam sposób przed podaniem na wejście.
    try:  # Rozpoczyna blok obsługi wyjątków dla operacji wczytywania plików.
        mean = np.load("wine_mean.npy")  # Wczytuje średnie wartości cech zapisane podczas treningu.
        std = np.load("wine_std.npy")  # Wczytuje odchylenia standardowe cech zapisane podczas treningu.
    except FileNotFoundError:  # Obsługuje sytuację, gdy pliki normalizacji nie istnieją.
        print("Błąd: Nie znaleziono plików wine_mean.npy lub wine_std.npy.")  # Wyświetla komunikat o braku plików.
        print("Uruchom najpierw skrypt wine_train.py, aby wygenerować parametry normalizacji.")  # Instruuje użytkownika co robić.
        return  # Przerywa działanie programu.

    # Standaryzacja (Z-score normalization): odejmujemy średnią i dzielimy przez odchylenie standardowe.
    # Dzięki broadcastingowi NumPy operacja odjęcia wektora średnich od macierzy danych
    # wykonuje się automatycznie dla każdego elementu.
    x = (x - mean) / std  # Normalizuje dane wejściowe używając wczytanych średnich i odchyleń.

    # Ładowanie zapisanego modelu .keras.
    try:  # Rozpoczyna blok obsługi wyjątków dla ładowania modelu.
        model = tf.keras.models.load_model("wine_best_model.keras")  # Ładuje wytrenowany model sieci neuronowej z pliku.
    except OSError:  # Obsługuje błąd, gdy plik modelu nie istnieje lub jest uszkodzony.
        print("Błąd: Nie znaleziono modelu wine_best_model.keras.")  # Wyświetla komunikat o błędzie.
        return  # Przerywa działanie programu.

    # Wykonanie inferencji.
    # Argument verbose=0 wycisza paski postępu, co jest pożądane w aplikacjach CLI.
    # Wynikiem jest tablica prawdopodobieństw przynależności do każdej z 3 klas.
    predictions = model.predict(x, verbose=0)  # Wykonuje predykcję na znormalizowanych danych, zwraca prawdopodobieństwa.

    # Wyciągnięcie pierwszego (i jedynego) wyniku z batcha.
    probs = predictions[0]  # Pobiera wynik dla pierwszej próbki (tablica prawdopodobieństw).

    # Wybór klasy zwycięskiej.
    # Funkcja argmax zwraca indeks największej wartości w tablicy prawdopodobieństw.
    predicted_index = np.argmax(probs)  # Znajduje indeks klasy z najwyższym prawdopodobieństwem.

    # Konwersja indeksu (0-2) na etykietę klasy zgodną z datasetem (1-3).
    predicted_class = int(predicted_index) + 1  # Przelicza indeks (0-2) na numer klasy (1-3).

    print("Prawdopodobieństwa klas:", probs)  # Wyświetla obliczone prawdopodobieństwa dla każdej klasy.
    print("Przewidywana klasa wina:", predicted_class)  # Wyświetla ostateczny wynik klasyfikacji.


if __name__ == "__main__":  # Sprawdza czy skrypt został uruchomiony bezpośrednio (a nie zaimportowany).
    main()  # Wywołuje główną funkcję programu.
