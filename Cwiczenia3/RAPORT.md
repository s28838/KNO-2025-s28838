## Dane i preprocessing
- Podział: 80% train / 20% test (tasowanie z seed=42)
- Standaryzacja: Z-score na podstawie zbioru treningowego (μ i σ zapisane do plików)
- Etykiety: one-hot (3 klasy)
- Wejście: 13 znormalizowanych cech float32

## Modele i trening
- Model 1 (ReLU + HeUniform): 13 → Dense(64, ReLU) → Dense(32, ReLU) → Dense(3, Softmax)
- Model 2 (tanh + GlorotNormal): 13 → Dense(128, tanh) → Dense(32, tanh) → Dense(3, Softmax)
- Hiperparametry: epoki=100, batch=16, lr=0.001, Adam, loss=categorical_crossentropy, metric=accuracy
- Logowanie: TensorBoard; zapis najlepszego modelu do wine_best_model.keras

## Wyniki (z learning_curve.csv)
- Postęp uczenia:
  - accuracy (train): 0.50 → 1.00 (płynny wzrost)
  - val_accuracy: 0.61 → 0.94 (stabilizacja ok. 0.94)
  - loss (train): 1.00 → ~0.0012 (spadek)
  - val_loss: 0.84 → ~0.13 (spadek z niewielkimi wahaniami)
- Interpretacja:
  - Szybka konwergencja do wysokiej jakości
  - Niewielka luka train–val → dobra generalizacja
  - Stabilna val_accuracy ≈ 0.94

## Który model jest lepszy?
Na podstawie zachowania walidacji (stabilizacja val_accuracy ≈ 0.94 i systematyczny spadek val_loss) oraz typowej przewagi ReLU dla takich danych, lepszy okazał się Model 1 (ReLU + HeUniform). Został zapisany jako wine_best_model.keras i używany w predykcji.

## Wdrożenie i użycie
- Model: wine_best_model.keras
- Normalizacja w predykcji: wine_mean.npy, wine_std.npy
- Predykcja (wine_predict.py): wejście 13 cech, wyjście prawdopodobieństwa klas i klasa 1..3

## Podsumowanie
System osiąga stabilną jakość klasyfikacji (val_accuracy ~0.94) bez oznak istotnego przeuczenia. Pipeline obejmuje przygotowanie danych, trening dwóch wariantów, automatyczny wybór lepszego modelu oraz spójne przetwarzanie w predykcji.