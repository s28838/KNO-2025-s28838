# ============================================================
#
#     python test_tasks.py
#
# ============================================================

import math  # Import modułu math (typ: module) - funkcje matematyczne
import numpy as np  # Import NumPy (typ: module) - obliczenia numeryczne, alias 'np'
import tensorflow as tf  # Import TensorFlow (typ: module) - obliczenia tensorowe, alias 'tf'

# Import testowanych funkcji z pliku main.py
# rotate_point: funkcja obrotu
# solve_linear_system: funkcja rozwiązywania układu równań
# solve_linear_system_cli: funkcja CLI do rozwiązywania układu równań
from main import rotate_point, solve_linear_system, solve_linear_system_cli


# -------------------------
# Zadania 1–2
# -------------------------
def test_rotate_point():
    # Funkcja testująca poprawność obrotu punktu
    print("===== TESTY ROTACJI PUNKTU =====") # (typ: None)

    # Test 1: Obrót punktu (1, 0) o 90 stopni (pi/2 radianów)
    # tf.constant tworzy tensor stały. Używamy float32 dla kąta.
    angle1 = tf.constant(math.pi / 2, tf.float32) # (typ: tf.Tensor, dtype=float32)
    # Wywołanie funkcji obrotu
    res1 = rotate_point(1.0, 0.0, angle1) # (typ: tf.Tensor)
    # .numpy() konwertuje tensor wyniku na tablicę NumPy w celu wyświetlenia
    print("Test 1:", res1.numpy(), "→ oczekiwane [0, 1]") # (typ: None)

    # Test 2: Obrót punktu (1, 1) o 180 stopni (pi radianów)
    angle2 = tf.constant(math.pi, tf.float32) # (typ: tf.Tensor, dtype=float32)
    res2 = rotate_point(1.0, 1.0, angle2) # (typ: tf.Tensor)
    print("Test 2:", res2.numpy(), "→ oczekiwane [-1, -1]") # (typ: None)

    # Test 3: Obrót punktu (1, 0) o 270 stopni (3pi/2 radianów)
    angle3 = tf.constant(3 * math.pi / 2, tf.float32) # (typ: tf.Tensor, dtype=float32)
    res3 = rotate_point(1.0, 0.0, angle3) # (typ: tf.Tensor)
    print("Test 3:", res3.numpy(), "→ oczekiwane [0, -1]") # (typ: None)

    # Asercje sprawdzające poprawność wyników z tolerancją błędu (atol=1e-6)
    # np.allclose zwraca True, jeśli dwie tablice są sobie równe w granicach tolerancji
    assert np.allclose(res1, [0.0, 1.0], atol=1e-6)  # Sprawdzenie wyniku Testu 1 (typ: bool)
    assert np.allclose(res2, [-1.0, -1.0], atol=1e-6) # Sprawdzenie wyniku Testu 2 (typ: bool)
    assert np.allclose(res3, [0.0, -1.0], atol=1e-6)  # Sprawdzenie wyniku Testu 3 (typ: bool)
    
    print("Wszystkie testy rotacji OK\n") # (typ: None)


# -------------------------------
# Zadanie 3
# -------------------------------
def test_solve_linear_system():
    # Funkcja testująca rozwiązywanie układów równań liniowych Ax=b
    print("===== TESTY UKŁADU RÓWNAŃ (A*x=b) =====") # (typ: None)

    # Test 1: Układ 2x2.
    # Macierz A zdefiniowana jako tensor stały float32
    A1 = tf.constant([[3.0, 2.0], [1.0, 2.0]], tf.float32) # (typ: tf.Tensor, kształt=(2,2), dtype=float32)
    # Wektor b zdefiniowany jako tensor stały float32
    b1 = tf.constant([7.0, 5.0], tf.float32) # (typ: tf.Tensor, kształt=(2,), dtype=float32)
    
    # Rozwiązanie układu i spłaszczenie wyniku do wektora jednowymiarowego
    # solve_linear_system zwraca (n,1), reshape zmienia na (n,)
    x1 = tf.reshape(solve_linear_system(A1, b1), (-1,)) # (typ: tf.Tensor, kształt=(2,), dtype=float32)
    
    print("Test 1:", x1.numpy(), "→ oczekiwane [1, 2]") # (typ: None)
    # Sprawdzenie poprawności wyniku: [1.0, 2.0]
    assert np.allclose(x1, [1.0, 2.0], atol=1e-6) # (typ: bool)

    # Test 2: Inny układ 2x2
    A2 = tf.constant([[2.0, 1.0], [5.0, 3.0]], tf.float32) # (typ: tf.Tensor, kształt=(2,2))
    b2 = tf.constant([4.0, 11.0], tf.float32) # (typ: tf.Tensor, kształt=(2,))
    x2 = tf.reshape(solve_linear_system(A2, b2), (-1,)) # (typ: tf.Tensor)
    print("Test 2:", x2.numpy(), "→ oczekiwane [1, 2]") # (typ: None)
    assert np.allclose(x2, [1.0, 2.0], atol=1e-6) # (typ: bool)

    # Test 3: Układ 3x3
    A3 = tf.constant(
        [[2.0, 1.0, -1.0],
         [-3.0, -1.0, 2.0],
         [-2.0, 1.0, 2.0]], tf.float32
    ) # (typ: tf.Tensor, kształt=(3,3))
    b3 = tf.constant([8.0, -11.0, -3.0], tf.float32) # (typ: tf.Tensor, kształt=(3,))
    x3 = tf.reshape(solve_linear_system(A3, b3), (-1,)) # (typ: tf.Tensor)
    print("Test 3:", x3.numpy(), "→ oczekiwane [2, 3, -1]") # (typ: None)
    assert np.allclose(x3, [2.0, 3.0, -1.0], atol=1e-6) # (typ: bool)

    print("Wszystkie testy układu równań OK\n") # (typ: None)


# -------------------------------------------------
# Zadanie 4
# -------------------------------------------------
def test_solve_linear_system_cli():
    # Funkcja testująca wersję CLI (przyjmującą płaskie listy)
    print("===== TESTY WERSJI CLI (A_flat, b_vec) =====") # (typ: None)

    # Test 1: Układ 2x2. Dane wejściowe to zwykłe listy pythona (nie tensory)
    A_flat1 = [3, 2, 1, 2] # (typ: list[int])
    b_vec1 = [7, 5]  # (typ: list[int])
    
    # Wywołanie funkcji CLI, która sama zajmie się konwersją na tensory
    x1 = solve_linear_system_cli(A_flat1, b_vec1) # (typ: tf.Tensor)
    print("Test 1:", x1.numpy(), "→ oczekiwane [1, 2]") # (typ: None)
    assert np.allclose(x1, [1.0, 2.0], atol=1e-6) # (typ: bool)

    # Test 2: Inny układ 2x2
    A_flat2 = [2, 1, 5, 3] # (typ: list[int])
    b_vec2 = [4, 11] # (typ: list[int])
    x2 = solve_linear_system_cli(A_flat2, b_vec2) # (typ: tf.Tensor)
    print("Test 2:", x2.numpy(), "→ oczekiwane [1, 2]") # (typ: None)
    assert np.allclose(x2, [1.0, 2.0], atol=1e-6) # (typ: bool)

    # Test 3: Układ 3x3
    A_flat3 = [2, 1, -1, -3, -1, 2, -2, 1, 2] # (typ: list[int])
    b_vec3 = [8, -11, -3] # (typ: list[int])
    x3 = solve_linear_system_cli(A_flat3, b_vec3) # (typ: tf.Tensor)
    print("Test 3:", x3.numpy(), "→ oczekiwane [2, 3, -1]") # (typ: None)
    assert np.allclose(x3, [2.0, 3.0, -1.0], atol=1e-6) # (typ: bool)

    print("Wszystkie testy CLI OK\n") # (typ: None)


def main():
    # Funkcja główna uruchamiająca wszystkie testy
    print("===================================") # (typ: None)
    print("URUCHAMIANIE TESTÓW FUNKCJI") # (typ: None)
    print("===================================") # (typ: None)
    test_rotate_point() # Uruchomienie testów rotacji
    test_solve_linear_system() # Uruchomienie testów układów równań
    test_solve_linear_system_cli() # Uruchomienie testów CLI
    print("Wszystkie testy przeszły pomyślnie!") # (typ: None)
    print("===================================\n") # (typ: None)


if __name__ == "__main__":
    main() # Uruchomienie funkcji main
