import math

import numpy as np
import tensorflow as tf

from main import rotate_point, solve_linear_system, solve_linear_system_cli


def test_rotate_point():  # Definiuje funkcję testową dla operacji obrotu punktu, weryfikuje poprawność geometryczną transformacji.
    print("===== TESTY ROTACJI PUNKTU =====")  # Wyświetla nagłówek sekcji testowej w konsoli, ułatwia analizę wyników.

    # Scenariusz 1: Obrót o 90 stopni (pi/2).
    angle1 = tf.constant(math.pi / 2, tf.float32)  # Tworzy stałą tensorową reprezentującą kąt 90 stopni, dane testowe.
    res1 = rotate_point(1.0, 0.0, angle1)  # Wywołuje funkcję rotate_point dla punktu (1,0), testuje obrót o 90 stopni.
    print("Test 1:", res1.numpy(), "→ oczekiwane [0, 1]")  # Wypisuje otrzymany wynik i wartość oczekiwaną, logowanie przebiegu testu.

    # Scenariusz 2: Obrót o 180 stopni (pi).
    angle2 = tf.constant(math.pi, tf.float32)  # Tworzy stałą tensorową reprezentującą kąt 180 stopni, dane testowe.
    res2 = rotate_point(1.0, 1.0, angle2)  # Wywołuje funkcję rotate_point dla punktu (1,1), testuje obrót o 180 stopni.
    print("Test 2:", res2.numpy(), "→ oczekiwane [-1, -1]")  # Wypisuje otrzymany wynik i wartość oczekiwaną, logowanie przebiegu testu.

    # Scenariusz 3: Obrót o 270 stopni (3pi/2).
    angle3 = tf.constant(3 * math.pi / 2, tf.float32)  # Tworzy stałą tensorową reprezentującą kąt 270 stopni, dane testowe.
    res3 = rotate_point(1.0, 0.0, angle3)  # Wywołuje funkcję rotate_point dla punktu (1,0), testuje obrót o 270 stopni.
    print("Test 3:", res3.numpy(), "→ oczekiwane [0, -1]")  # Wypisuje otrzymany wynik i wartość oczekiwaną, logowanie przebiegu testu.

    # Weryfikacja poprawności obliczeń z tolerancją na błędy numeryczne (atol=1e-6).
    assert np.allclose(res1, [0.0, 1.0], atol=1e-6)  # Sprawdza czy wynik Testu 1 jest bliski [0, 1], automatyczna weryfikacja poprawności.
    assert np.allclose(res2, [-1.0, -1.0], atol=1e-6)  # Sprawdza czy wynik Testu 2 jest bliski [-1, -1], automatyczna weryfikacja poprawności.
    assert np.allclose(res3, [0.0, -1.0], atol=1e-6)  # Sprawdza czy wynik Testu 3 jest bliski [0, -1], automatyczna weryfikacja poprawności.

    print("Wszystkie testy rotacji OK\n")  # Informuje o pomyślnym przejściu testów rotacji, potwierdzenie sukcesu.


def test_solve_linear_system():  # Definiuje funkcję testową dla solera układów równań, weryfikuje poprawność matematyczną rozwiązań.
    print("===== TESTY UKŁADU RÓWNAŃ (A*x=b) =====")  # Wyświetla nagłówek sekcji testowej w konsoli, ułatwia analizę wyników.

    # Scenariusz 1: Prosty układ 2x2.
    # 3x + 2y = 7
    # 1x + 2y = 5
    # Oczekiwane rozwiązanie: x=1, y=2.
    A1 = tf.constant([[3.0, 2.0], [1.0, 2.0]], tf.float32)  # Definiuje macierz współczynników A dla pierwszego układu, dane testowe.
    b1 = tf.constant([7.0, 5.0], tf.float32)  # Definiuje wektor wyrazów wolnych b dla pierwszego układu, dane testowe.

    # Rozwiązanie i spłaszczenie wyniku do wektora jednowymiarowego.
    x1 = tf.reshape(solve_linear_system(A1, b1), (-1,))  # Rozwiązuje układ i spłaszcza wynik do 1D, ułatwia porównanie.

    print("Test 1:", x1.numpy(), "→ oczekiwane [1, 2]")  # Wypisuje otrzymany wynik i wartość oczekiwaną, logowanie przebiegu testu.
    assert np.allclose(x1, [1.0, 2.0], atol=1e-6)  # Sprawdza czy wynik Testu 1 jest bliski [1, 2], automatyczna weryfikacja poprawności.

    # Scenariusz 2: Inny układ 2x2.
    A2 = tf.constant([[2.0, 1.0], [5.0, 3.0]], tf.float32)  # Definiuje macierz współczynników A dla drugiego układu, dane testowe.
    b2 = tf.constant([4.0, 11.0], tf.float32)  # Definiuje wektor wyrazów wolnych b dla drugiego układu, dane testowe.
    x2 = tf.reshape(solve_linear_system(A2, b2), (-1,))  # Rozwiązuje drugi układ i spłaszcza wynik, operacja obliczeniowa.
    print("Test 2:", x2.numpy(), "→ oczekiwane [1, 2]")  # Wypisuje otrzymany wynik i wartość oczekiwaną, logowanie przebiegu testu.
    assert np.allclose(x2, [1.0, 2.0], atol=1e-6)  # Sprawdza czy wynik Testu 2 jest bliski [1, 2], automatyczna weryfikacja poprawności.

    # Scenariusz 3: Układ 3x3.
    # Sprawdza działanie algorytmu dla większej liczby zmiennych.
    A3 = tf.constant(  # Definiuje macierz 3x3 dla trzeciego układu, test złożoności.
        [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]], tf.float32  # Wartości macierzy 3x3, dane testowe.
    )
    b3 = tf.constant([8.0, -11.0, -3.0], tf.float32)  # Definiuje wektor wyrazów wolnych b dla trzeciego układu, dane testowe.
    x3 = tf.reshape(solve_linear_system(A3, b3), (-1,))  # Rozwiązuje układ 3x3 i spłaszcza wynik, operacja obliczeniowa.
    print("Test 3:", x3.numpy(), "→ oczekiwane [2, 3, -1]")  # Wypisuje otrzymany wynik i wartość oczekiwaną, logowanie przebiegu testu.
    assert np.allclose(x3, [2.0, 3.0, -1.0], atol=1e-6)  # Sprawdza czy wynik Testu 3 jest bliski [2, 3, -1], automatyczna weryfikacja poprawności.

    print("Wszystkie testy układu równań OK\n")  # Informuje o pomyślnym przejściu testów układów równań, potwierdzenie sukcesu.


def test_solve_linear_system_cli():  # Definiuje funkcję testową dla wrappera CLI, weryfikuje obsługę płaskich list.
    print("===== TESTY WERSJI CLI (A_flat, b_vec) =====")  # Wyświetla nagłówek sekcji testowej CLI, ułatwia analizę wyników.

    # Scenariusz 1: Odpowiednik Test 1 dla danych wejściowych w formie list Pythonowych.
    A_flat1 = [3, 2, 1, 2]  # Definiuje spłaszczoną listę macierzy A, format wejściowy CLI.
    b_vec1 = [7, 5]  # Definiuje listę wektora b, format wejściowy CLI.

    x1 = solve_linear_system_cli(A_flat1, b_vec1)  # Wywołuje funkcję CLI z listami, testuje interfejs konsolowy.
    print("Test 1:", x1.numpy(), "→ oczekiwane [1, 2]")  # Wypisuje wynik i oczekiwane wartości, logowanie.
    assert np.allclose(x1, [1.0, 2.0], atol=1e-6)  # Weryfikuje poprawność wyniku z tolerancją błędu.

    # Scenariusz 2: Odpowiednik Test 2.
    A_flat2 = [2, 1, 5, 3]  # Definiuje spłaszczoną macierz dla drugiego przypadku, dane testowe.
    b_vec2 = [4, 11]  # Definiuje wektor b dla drugiego przypadku, dane testowe.
    x2 = solve_linear_system_cli(A_flat2, b_vec2)  # Wywołuje funkcję CLI, testuje inny zestaw danych.
    print("Test 2:", x2.numpy(), "→ oczekiwane [1, 2]")  # Wypisuje wynik i oczekiwane wartości, logowanie.
    assert np.allclose(x2, [1.0, 2.0], atol=1e-6)  # Weryfikuje poprawność wyniku.

    # Scenariusz 3: Odpowiednik Test 3 (3x3).
    A_flat3 = [2, 1, -1, -3, -1, 2, -2, 1, 2]  # Definiuje spłaszczoną macierz 3x3, dane testowe.
    b_vec3 = [8, -11, -3]  # Definiuje wektor b 3-elementowy, dane testowe.
    x3 = solve_linear_system_cli(A_flat3, b_vec3)  # Wywołuje funkcję CLI dla układu 3x3, test obsługi większych macierzy w CLI.
    print("Test 3:", x3.numpy(), "→ oczekiwane [2, 3, -1]")  # Wypisuje wynik i oczekiwane wartości, logowanie.
    assert np.allclose(x3, [2.0, 3.0, -1.0], atol=1e-6)  # Weryfikuje poprawność wyniku.

    print("Wszystkie testy CLI OK\n")  # Informuje o pomyślnym zakończeniu testów CLI.


def main():  # Główna funkcja uruchamiająca zestaw testów.
    print("===================================")  # Wypisuje separator graficzny na start.
    print("URUCHAMIANIE TESTÓW FUNKCJI")  # Wypisuje tytuł uruchamianego procesu.
    print("===================================")  # Wypisuje separator graficzny.
    test_rotate_point()  # Uruchamia testy rotacji punktu.
    test_solve_linear_system()  # Uruchamia testy solvera macierzowego.
    test_solve_linear_system_cli()  # Uruchamia testy wrappera CLI.
    print("Wszystkie testy przeszły pomyślnie!")  # Komunikat końcowy o sukcesie wszystkich testów.
    print("===================================\n")  # Wypisuje końcowy separator graficzny.


if __name__ == "__main__":  # Sprawdza czy plik uruchomiono jako skrypt główny.
    main()  # Wywołuje funkcję główną main.
