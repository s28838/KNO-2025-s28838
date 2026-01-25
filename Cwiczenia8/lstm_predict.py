import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def generate_data(n_points=1000):  # Generuje zbiór danych w postaci funkcji sinus.
    # Tworzenie osi X: generowanie 'n_points' liczb równomiernie rozłożonych od 0 do 50
    x = np.linspace(0, 50, n_points)  # Tworzy wektor czasu.
    # Obliczanie wartości funkcji sinus dla każdego punktu z osi X
    y = np.sin(x)  # Oblicza wartości sinusa.
    # Zwracanie krotki zawierającej tablice x (czas) i y (wartości)
    return x, y  # Zwraca dane.


def create_dataset(data, look_back=1, steps_ahead=1):  # Konwertuje szereg czasowy na zbiór danych do uczenia nadzorowanego.
    # Inicjalizacja pustej listy na dane wejściowe (cechy)
    X = []  # Bufor na cechy.
    # Inicjalizacja pustej listy na dane wyjściowe (etykiety/cele)
    Y = []  # Bufor na etykiety.
    # Pętla iterująca przez indeksy danych, zatrzymująca się odpowiednio wcześniej,
    # aby uniknąć wyjścia poza zakres tablicy (indeks końcowy musi uwzględniać look_back i steps_ahead)
    for i in range(len(data) - look_back - steps_ahead):  # Iteruje po danych z marginesem bezpieczeństwa.
        # Wycinanie fragmentu danych od indeksu 'i' do 'i + look_back' (to jest nasze "okno" historii)
        # Pobieramy tylko kolumnę 0, ponieważ data ma kształt (N, 1)
        a = data[i:(i + look_back), 0]  # Pobiera okno danych historycznych.
        # Dodawanie wyciętego fragmentu do listy X
        X.append(a)  # Dodaje okno do listy cech.
        # Pobieranie wartości docelowej, która znajduje się 'steps_ahead' kroków za oknem 'look_back'
        val = data[i + look_back + steps_ahead, 0]  # Pobiera wartość przyszłą (cel predykcji).
        # Dodawanie wartości docelowej do listy Y
        Y.append(val)  # Dodaje cel do listy etykiet.
    # Konwersja list X i Y na tablice NumPy i zwrócenie ich
    return np.array(X), np.array(Y)  # Zwraca gotowe tablice treningowe.


def main():  # Główna funkcja programu, steruje procesem eksperymentu.
    # --- Konfiguracja Eksperymentu ---
    # LOOK_BACK: Ważny parametr. Określa "pamięć" sieci - ile poprzednich punktów
    # model bierze pod uwagę przy zgadywaniu następnego.
    LOOK_BACK = 20  # Ustawia długość okna historycznego.
    
    # STEPS_AHEAD: Horyzont predykcji. Tutaj 0 oznacza następy krok bezpośrednio po sekwencji (t+1).
    # Z logicznego punktu widzenia w create_dataset, jeśli chcemy przewidzieć "następny", to offset jest 0
    # względem końca okna. Zmienimy to na 0 dla "następnego kroku".
    STEPS_AHEAD = 0   # Ustawia horyzont predykcji na najbliższy krok.
    
    # EPOCHS: Liczba epok trenowania, czyli ile razy algorytm przejdzie przez cały zbiór danych treningowych
    EPOCHS = 10         # Ustawia liczbę epok.
    # BATCH_SIZE: Liczba próbek przetwarzanych jednocześnie w jednej iteracji przed aktualizacją wag modelu
    BATCH_SIZE = 32     # Ustawia rozmiar batcha.

    # --- 1. Przygotowanie Danych (Data Preparation) ---
    # Wyświetlenie komunikatu o rozpoczęciu generowania danych
    print("Generowanie danych (Funkcja Sinus)...")  # Loguje generowanie danych.
    # Generowanie danych: 1500 punktów z funkcji sinus za pomocą wcześniej zdefiniowanej funkcji
    x_axis, y_axis = generate_data(1500)  # Generuje dane syntetyczne.
    
    # Keras wymaga, by dane miały kształt (samples, features).
    # Nasz sinus ma tylko 1 cechę (feature) - amplitudę.
    # Reshape z (-1, 1) zmienia tablicę 1D [1, 2, 3] na kolumnę [[1], [2], [3]].
    data = y_axis.reshape(-1, 1)  # Przekształca wektor danych na macierz kolumnową.

    # Podział na zbiór treningowy i testowy.
    # Obliczenie indeksu podziału: 70% danych przeznaczamy na trening
    train_size = int(len(data) * 0.70)  # Oblicza punkt podziału train/test.
    # Wycięcie danych treningowych: od początku do obliczonego indeksu
    train_data = data[0:train_size, :]  # Wyodrębnia zbiór treningowy.
    # Wycięcie danych testowych: od obliczonego indeksu do końca
    test_data = data[train_size:len(data), :]  # Wyodrębnia zbiór testowy.
    # Wyświetlenie informacji o liczbie próbek w zbiorach
    print(f"Liczba próbek treningowych: {len(train_data)}, testowych: {len(test_data)}")  # Wypisuje rozmiary zbiorów.

    # Tworzenie sekwencji (X -> Y)
    # Wyświetlenie komunikatu o tworzeniu datasetu z zadanymi parametrami
    print(f"Tworzenie sekwencji (Lookback: {LOOK_BACK}, Ahead: {STEPS_AHEAD})...")  # Loguje tworzenie sekwencji.
    # Tworzenie par treningowych (wejście, wyjście) za pomocą funkcji create_dataset
    X_train, y_train = create_dataset(train_data, LOOK_BACK, STEPS_AHEAD)  # Tworzy dataset treningowy.
    # Tworzenie par testowych (wejście, wyjście) za pomocą funkcji create_dataset
    X_test, y_test = create_dataset(test_data, LOOK_BACK, STEPS_AHEAD)  # Tworzy dataset testowy.

    # Sieci LSTM w Kerasie wymagają trójwymiarowego wejścia: [Samples, Time Steps, Features]
    # Zmiana kształtu X_train:
    # - X_train.shape[0]: liczba próbek (Samples)
    # - X_train.shape[1]: długość sekwencji (Time Steps / Lookback)
    # - 1: liczba cech (Features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # Dodaje wymiar cech do X_train.
    # Analogiczna zmiana kształtu dla zbioru testowego X_test
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Dodaje wymiar cech do X_test.

    # Wyświetlenie ostatecznego kształtu danych treningowych dla weryfikacji
    print(f"Kształt danych treningowych X: {X_train.shape}, Y: {y_train.shape}")  # Wypisuje kształty tensora wejściowego.

    # --- 2. Budowa Modelu (Model Construction) ---
    # Wyświetlenie komunikatu o budowaniu modelu
    print("Budowanie modelu LSTM...")  # Loguje budowę modelu.
    # Inicjalizacja modelu sekwencyjnego (stos warstw jedna po drugiej)
    model = keras.Sequential([  # Inicjalizuje model.
        # Warstwa wejściowa: definiuje oczekiwany kształt danych wejściowych (długość sekwencji, liczba cech)
        layers.Input(shape=(LOOK_BACK, 1)),  # Definiuje warstwę wejściową.
        
        # Warstwa LSTM: rekurencyjna warstwa pamięci długo-krótkoterminowej.
        # - 50: liczba neuronów/jednostek w warstwie (wymiarowość przestrzeni wyjściowej)
        # - activation='tanh': funkcja aktywacji hiperboliczna (standard dla LSTM)
        # Warstwa zwraca tylko ostatni stan wyjściowy (ostatni krok czasowy), bo return_sequences=False (domyślnie)
        layers.LSTM(50, activation='tanh'),  # Dodaje warstwę LSTM z 50 jednostkami.
        
        # Warstwa Dense (Gęsta): w pełni połączona warstwa wyjściowa
        # - 1: jeden neuron, ponieważ przewidujemy jedną wartość liczbową (regresja)
        layers.Dense(1)  # Dodaje warstwę wyjściową (regresja).
    ])

    # Kompilacja modelu (przygotowanie do trenowania):
    # - optimizer='adam': wydajny algorytm optymalizacji
    # - loss='mse': błąd średniokwadratowy jako funkcja straty (minimalizujemy różnicę kwadratów błędów)
    model.compile(optimizer='adam', loss='mse')  # Kompiluje model.
    # Wyświetlenie podsumowania architektury modelu w konsoli
    model.summary()  # Wypisuje podsumowanie modelu.

    # --- 3. Trenowanie (Training) ---
    # Wyświetlenie komunikatu o starcie treningu
    print("Rozpoczynam trening...")  # Loguje start treningu.
    # Uruchomienie procesu uczenia metodą fit():
    # - X_train, y_train: dane treningowe
    # - epochs=EPOCHS: liczba iteracji po całym zbiorze
    # - batch_size=BATCH_SIZE: rozmiar partii danych
    # - validation_split=0.1: wykorzystanie 10% danych treningowych jako zbioru walidacyjnego
    # - verbose=1: wyświetlanie paska postępu
    history = model.fit(  # Uruchamia trening.
        X_train, y_train,  # Dane treningowe.
        epochs=EPOCHS,  # Liczba epok.
        batch_size=BATCH_SIZE,  # Rozmiar batcha.
        validation_split=0.1,  # Podział walidacyjny.
        verbose=1  # Wyświetlanie postępów.
    )

    # --- 4. Ewaluacja i Predykcja (Evaluation) ---
    # Wygenerowanie predykcji dla zbioru treningowego w celu sprawdzenia dopasowania
    train_predict = model.predict(X_train)  # Wykonuje predykcję na danych treningowych.
    # Wygenerowanie predykcji dla zbioru testowego (sprawdzenie generalizacji)
    test_predict = model.predict(X_test)  # Wykonuje predykcję na danych testowych.

    # --- 5. Wizualizacja (Visualization) ---
    # Wyświetlenie komunikatu o generowaniu wykresów
    print("Generowanie wykresów...")  # Loguje generowanie wykresów.
    
    # Wykres 1: Krzywa uczenia (Loss)
    # Utworzenie nowej figury o rozmiarze 10x6 cali
    plt.figure(figsize=(10, 6))  # Ustawia rozmiar wykresu.
    # Wyrysowanie wartości straty na zbiorze treningowym z historii uczenia
    plt.plot(history.history['loss'], label='Strata Treningowa (Train Loss)')  # Rysuje stratę treningową.
    # Wyrysowanie wartości straty na zbiorze walidacyjnym
    plt.plot(history.history['val_loss'], label='Strata Walidacyjna (Val Loss)')  # Rysuje stratę walidacyjną.
    # Dodanie tytułu wykresu
    plt.title('Przebieg procesu uczenia (Model Loss)')  # Ustawia tytuł.
    # Opis osi Y
    plt.ylabel('Wartość funkcji straty (MSE)')  # Opisuje oś Y.
    # Opis osi X
    plt.xlabel('Epoka (Epoch)')  # Opisuje oś X.
    # Dodanie legendy
    plt.legend()  # Dodaje legendę.
    # Dodanie siatki pomocniczej
    plt.grid(True, linestyle='--', alpha=0.6)  # Włącza siatkę.
    # Zapisanie wykresu do pliku
    plt.savefig('learning_curve.png')  # Zapisuje wykres.
    # plt.show() # Opcjonalne wyświetlenie okna (zablokowane w środowisku bez GUI)

    # Wykres 2: Porównanie Rzeczywistość vs Predykcja
    # Utworzenie nowej figury o rozmiarze 15x6 cali
    plt.figure(figsize=(15, 6))  # Ustawia rozmiar drugiego wykresu.
    
    # Przygotowanie osi X dla predykcji treningowych:
    # Predykcje są "krótsze" o LOOK_BACK na początku, więc przesuwamy start.
    train_plot_x = np.arange(LOOK_BACK + STEPS_AHEAD, len(train_predict) + LOOK_BACK + STEPS_AHEAD)  # Tworzy oś X dla predykcji treningowej.
    
    # Przygotowanie osi X dla predykcji testowych:
    # Zaczynają się po zbiorze treningowym (+ przesunięcie LOOK_BACK)
    test_plot_x = np.arange(  # Tworzy oś X dla predykcji testowej.
        train_size + LOOK_BACK + STEPS_AHEAD,   # Start po treningu.
        train_size + LOOK_BACK + STEPS_AHEAD + len(test_predict)  # Koniec po teście.
    )

    # Wyrysowanie oryginalnych danych (cały przebieg sinusa) kolorem szarym
    plt.plot(y_axis, label='Prawdziwe dane (Sinus)', alpha=0.5, color='gray', linewidth=2)  # Rysuje dane referencyjne.
    # Wyrysowanie predykcji dla części treningowej na niebiesko
    plt.plot(train_plot_x, train_predict, label='Predykcja (Trening)', color='blue')  # Rysuje predykcję treningową.
    # Wyrysowanie predykcji dla części testowej na czerwono (linia przerywana)
    plt.plot(test_plot_x, test_predict, label='Predykcja (Test)', color='red', linestyle='--')  # Rysuje predykcję testową.
    
    # Dodanie tytułu z parametrem LOOK_BACK
    plt.title(f'Predykcja funkcji Sinus za pomocą LSTM (Lookback: {LOOK_BACK})')  # Ustawia tytuł wyniku.
    # Opis osi X
    plt.xlabel('Krok czasowy')  # Opisuje oś X.
    # Opis osi Y
    plt.ylabel('Wartość')  # Opisuje oś Y.
    # Dodanie legendy
    plt.legend()  # Dodaje legendę.
    # Dodanie siatki
    plt.grid(True, linestyle='--', alpha=0.6)  # Włącza siatkę.
    # Zapisanie wykresu do pliku
    plt.savefig('prediction_results.png')  # Zapisuje wykres końcowy.
    
    # Wyświetlenie komunikatu o zakończeniu zapisu
    print("Zapisano wykresy do plików: 'learning_curve.png' oraz 'prediction_results.png'")  # Potwierdza zapis plików.
    # plt.show()

# Sprawdzenie, czy skrypt jest uruchamiany bezpośrednio (a nie importowany)
if __name__ == "__main__":  # Sprawdza main.
    # Wywołanie głównej funkcji programu
    main()  # Uruchamia program.
