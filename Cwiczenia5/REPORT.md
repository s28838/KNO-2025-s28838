# Lab 5: Fashion MNIST Classification - Raport

## 1. Wstęp
Celem laboratorium było stworzenie systemu klasyfikacji ubrań na podstawie zbioru Fashion MNIST. System składa się z:
- Modułu budowania modeli (`models.py`) obsługującego architektury Dense i CNN.
- Skryptu trenującego (`train.py`) z wykorzystaniem Keras Tuner.
- Skryptu predykcji (`predict.py`) normalizującego zdjęcia rzeczywiste.

## 2. Architektura i Strojenie
Zastosowano `RandomSearch` do znalezienia najlepszych hiperparametrów. Przestrzeń poszukiwań obejmowała:
- **Typ modelu:** `Dense` (MLP) vs `CNN`.
- **Liczba warstw/bloków:** 1-3.
- **Liczba neuronów/filtrów:** Zmienna w zależności od warstwy.
- **Augmentacja danych:** Włączona/Wyłączona (RandomFlip, RandomRotation, RandomZoom).
- **Learning Rate:** 1e-4 do 1e-2 (skala log).

## 3. Wyniki
Znaleziono najlepszą konfigurację (na podstawie `metrics.json`):
- **Typ modelu:** CNN (Convolutional Neural Network)
- **Struktura:** 1 blok konwolucyjny, 16 filtrów.
- **Augmentacja:** False (w tym przebiegu prostszy model bez augmentacji okazał się lepszy lub szybciej zbieżny).
- **Learning Rate:** ~0.0026

**Metryki na zbiorze testowym:**
- **Dokładność (Accuracy):** 90.53%
- **Strata (Loss):** 0.4481

## 4. Macierz Pomyłek (Confusion Matrix)
Poniżej przedstawiono macierz pomyłek. Widać, że model najlepiej radzi sobie z:
- **Trouser (Spodnie):** 975 poprawnych / 1000
- **Sandal (Sandały):** 975 poprawnych / 1000
- **Bag (Torby):** 972 poprawnych / 1000

Największe trudności sprawia rozróżnianie:
- **Shirt (Koszula)** często mylona z **T-shirt/top** (klasy podobne wizualnie).
- **Pullover (Sweter)** mylony z **Coat (Płaszcz)**.

## 5. Augmentacja Danych (Bonus)
Zaimplementowano warstwę augmentacji zawierającą:
- `RandomFlip("horizontal")`: Odbicie lustrzane.
- `RandomRotation(0.1)`: Obrót o +/- 10%.
- `RandomZoom(0.1)`: Przybliżenie/oddalenie.

W finalnym (najlepszym) modelu tuner wybrał wariant **bez augmentacji**. Może to wynikać z faktu, że przy ograniczonej liczbie epok (5 podczas szukania) modele z augmentacją uczą się wolniej (trudniejsze zadanie), przez co wypadły gorzej w krótkim czasie. Jednak docelowo augmentacja zazwyczaj poprawia generalizację przy dłuższym treningu.

## 6. Predykcja (Test Syntetyczny)
Przeprowadzono test `predict.py` na wygenerowanym syntetycznie obrazie (biały prostokąt na czarnym tle, symulujący kształt koszulki).
- **Wynik:** Shirt (Koszula)
- **Pewność:** 79.40%
System poprawnie zinterpretował prosty kształt geometryczny jako element górnej garderoby.
