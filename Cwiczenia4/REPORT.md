# Lab 4: Optymalizacja Hiperparametrów - Raport

## 1. Wstęp
Celem laboratorium była optymalizacja modelu klasyfikacji win przy użyciu biblioteki `keras-tuner`.
Jako punkt odniesienia (baseline) przyjęto model wykorzystujący warstwę normalizacji `tf.keras.layers.Normalization`.

## 2. Baseline
Model bazowy (Normalization + 2 warstwy Dense ReLU) osiągnął na zbiorze walidacyjnym:
- **Dokładność (Accuracy):** 0.9722
- **Strata (Loss):** 0.0591

## 3. Optymalizacja (Keras Tuner)
Wykorzystano metodę **RandomSearch** z następującą konfiguracją:
- **Cel optymalizacji:** `val_accuracy`
- **Liczba prób (max_trials):** 20
- **Optymalizowane parametry:**
  - `num_layers` (Int): Liczba warstw ukrytych (1-3).
  - `units_i` (Int): Liczba neuronów w warstwie (16-128, krok 16).
  - `activation` (Choice): Funkcja aktywacji (`relu`, `tanh`).
  - `lr` (Float): Współczynnik uczenia (1e-4 do 1e-2, skala logarytmiczna).

## 4. Wyniki Strojenia
Najlepsze znalezione hiperparametry:
- **Liczba warstw:** 1
- **Aktywacja:** tanh
- **Learning Rate:** ~0.0045
- **Konfiguracja:** Model uproszczony (1 warstwa ukryta) z aktywacją tanh.

## 5. Wynik Końcowy
Po przetrenowaniu najlepszego modelu na pełną liczbę epok (100):
- **Dokładność (Accuracy):** 0.9722
- **Strata (Loss):** 0.0515

### Porównanie
| Model | Accuracy | Loss |
|-------|----------|------|
| Baseline | 0.9722 | 0.0591 |
| Tuned | 0.9722 | 0.0515 |

### Macierz Pomyłek (Tuned)
```
[[11  0  0]
 [ 0 14  0]
 [ 0  1 10]]
```
Klasyfikacja jest niemal idealna, z jednym błędnym przypisaniem (klasa 3 jako klasa 2).

## 6. Podsumowanie
Zastosowanie warstwy Normalization oraz optymalizacja hiperparametrów pozwoliła na zredukowanie funkcji straty (z 0.0591 do 0.0515) przy zachowaniu wysokiej dokładności. Tuner wybrał prostszą architekturę (1 warstwa ukryta) z funkcją aktywacji `tanh` i wyższym learning rate, co sugeruje, że model bazowy mógł być lekko nadmiarowy (over-parameterized) dla tego prostego zbioru danych.
