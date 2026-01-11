# ============================================================
#
#      Zadanie 1–2 — obrót punktu (x, y) o zadany kąt wokół (0,0)
#      Zadanie 3 — rozwiązanie układu równań A*x = b
#      Zadanie 4 — rozwiązanie Ax=b z parametrami z CLI
#
#
#     Obrót punktu:
#       python main.py rotate --x 1 --y 0 --angle 90 --degrees
#
#     Rozwiązanie układu:
#       python main.py solve --A 3 2 1 2 --b 5 5
#
#     Wersja CLI:
#       python main.py solve-cli --A 3 2 1 2 --b 5 5
#
# ============================================================

import argparse  # Import modułu argparse (typ: module) - do obsługi argumentów wiersza poleceń
import math  # Import modułu math (typ: module) - funkcje matematyczne (np. sqrt, radians)
import numpy as np  # Import NumPy (typ: module) - obliczenia numeryczne, alias 'np'
import tensorflow as tf  # Import TensorFlow (typ: module) - obliczenia tensorowe, alias 'tf'


# ============================================================
# Zadania 1–2 – Obrót punktu
# ============================================================
def rotate_point(x, y, angle_rad):
    # Funkcja wykonująca obrót punktu (x, y) o kąt w radianach
    # Argumenty:
    #   x: współrzędna x (typ: float lub int)
    #   y: współrzędna y (typ: float lub int)
    #   angle_rad: kąt obrotu w radianach (typ: float lub tf.Tensor)
    
    # Konwersja danych wejściowych na tensory typu float64
    angle = tf.cast(angle_rad, tf.float64)  # Rzutowanie kąta na tensor zmiennoprzecinkowy 64-bit (typ: tf.Tensor, dtype=float64)
    x = tf.cast(x, tf.float64)  # Rzutowanie x na tensor (typ: tf.Tensor, dtype=float64)
    y = tf.cast(y, tf.float64)  # Rzutowanie y na tensor (typ: tf.Tensor, dtype=float64)

    c = tf.cos(angle)  # Obliczenie cosinusa kąta (typ: tf.Tensor, dtype=float64)
    s = tf.sin(angle)  # Obliczenie sinusa kąta (typ: tf.Tensor, dtype=float64)

    # Budowanie macierzy rotacji R = [[cos, -sin], [sin, cos]]
    # Używamy tf.stack do łączenia tensorów skalarnych w tensor wyższego rzędu
    row1 = tf.stack([c, -s])  # Pierwszy wiersz macierzy (typ: tf.Tensor, kształt=(2,), dtype=float64)
    row2 = tf.stack([s, c])   # Drugi wiersz macierzy (typ: tf.Tensor, kształt=(2,), dtype=float64)
    R = tf.stack([row1, row2])  # Pełna macierz rotacji 2x2 wyższa (typ: tf.Tensor, kształt=(2,2), dtype=float64)
    
    p = tf.stack([x, y])  # Wektor punktu [x, y] (typ: tf.Tensor, kształt=(2,), dtype=float64)
    
    # Mnożenie macierzy przez wektor: R * p
    return tf.linalg.matvec(R, p)  # Wynik obrotu jako wektor (typ: tf.Tensor, kształt=(2,), dtype=float64)

# ============================================================
# Zadanie 3 – Rozwiązywanie układu A × x = b
# ============================================================
@tf.function  # Dekorator tf.function (typ: function) - kompiluje funkcję do grafu TensorFlow dla wydajności
def solve_linear_system(A, b):
    # Funkcja rozwiązująca układ równań liniowych A*x = b
    # Argumenty:
    #   A: macierz współczynników (typ: np.ndarray, lista lub tf.Tensor, n x n)
    #   b: wektor wyrazów wolnych (typ: np.ndarray, lista lub tf.Tensor, n)
    """
    Rozwiązuje układ równań A*x=b przy użyciu tf.linalg.solve.
    A – macierz (n×n)
    b – wektor (n)
    """
    A = tf.cast(A, tf.float32)  # Konwersja macierzy A na tensor float32 (typ: tf.Tensor, dtype=float32)
    b_cast = tf.cast(b, tf.float32) # Konwersja wektora b na tensor float32 (typ: tf.Tensor, dtype=float32)
    b = tf.reshape(b_cast, (-1, 1))  # Przekształcenie wektora b na wektor kolumnowy (n, 1) (typ: tf.Tensor, dtype=float32)
    return tf.linalg.solve(A, b)  # Rozwiązanie układu, zwraca wektor x (typ: tf.Tensor, kształt=(n, 1), dtype=float32)

# ============================================================
# Zadanie 4 – Wersja CLI (dane wprowadzone przez użytkownika)
# ============================================================
@tf.function  # Dekorator tf.function (typ: function) - optymalizacja wykonania w TensorFlow
def solve_linear_system_cli(A_flat, b_vec):
    # Funkcja pomocnicza dla CLI do rozwiązywania układu z płaskich list
    # Argumenty:
    #   A_flat: lista wszystkich elementów macierzy A (typ: list[float] lub tensor 1D)
    #   b_vec: lista elementów wektora b (typ: list[float] lub tensor 1D)
    """
    Rozwiązuje Ax=b, gdzie:
      A_flat – lista elementów macierzy A (płasko, wierszami)
      b_vec  – lista elementów wektora b
    Automatycznie oblicza rozmiar macierzy n×n.
    """
    # Obliczenie rozmiaru n macierzy kwadratowej (sqrt z liczby elementów)
    # Ponieważ tf.function działa na tensorach, len() może być użyte przy budowie grafu dla stałych kształtów,
    # ale logicznie n to integer.
    n = int(math.sqrt(len(A_flat)))  # Rozmiar macierzy n (typ: int)
    
    # Tworzenie tensora A z płaskiej listy i zmiana kształtu na (n, n)
    # np.array konwertuje listę na tablicę NumPy, reshape nadaje jej kształt
    A_np = np.array(A_flat, np.float32).reshape((n, n)) # (typ: np.ndarray, dtype=float32)
    A = tf.constant(A_np) # Utworzenie stałej (tessora) TensorFlow z tablicy NumPy (typ: tf.Tensor)
    
    # Tworzenie tensora b z listy
    b_np = np.array(b_vec, np.float32) # (typ: np.ndarray, dtype=float32)
    b = tf.constant(b_np) # Utworzenie stałej TensorFlow (typ: tf.Tensor)
    
    # Wywołanie funkcji rozwiązującej (zwraca kolumnę n x 1) i spłaszczenie wyniku do wektora 1D
    result = solve_linear_system(A, b) # (typ: tf.Tensor, kształt=(n, 1))
    return tf.reshape(result, (-1,)) # Zwraca wynik jako płaski wektor (typ: tf.Tensor, kształt=(n,))

# ============================================================
# Podkomendy programu
# ============================================================

def do_rotate(args):
    # Funkcja obsługująca polecenie 'rotate' z CLI
    # Argumenty:
    #   args: obiekt z argumentami z argparse (typ: argparse.Namespace)
    """Wykonuje obrót punktu."""
    
    # Obliczenie kąta w radianach: jeśli podano flagę --degrees, konwertuj ze stopni; inaczej użyj podanego
    angle = math.radians(args.angle) if args.degrees else args.angle # (typ: float)
    
    # Wywołanie funkcji rotate_point. Zamiana kąta (float) na stałą tensorową TensorFlow.
    # .numpy() konwertuje wynikowy tensor z powrotem na tablicę NumPy.
    t_angle = tf.constant(angle, tf.float64) # (typ: tf.Tensor)
    res_tensor = rotate_point(args.x, args.y, t_angle) # (typ: tf.Tensor)
    res = res_tensor.numpy() # (typ: np.ndarray)
    
    # Pobranie poszczególnych współrzędnych z tablicy NumPy
    rx, ry = float(res[0]), float(res[1]) # Konwersja na standardowy float pythona (typ: float, float)
    
    print(f"\n🔹 Punkt ({args.x}, {args.y}) po obrocie o {args.angle}{'°' if args.degrees else ' rad'}:") # (typ: None)
    print(f"   Wynik → ({rx:.6f}, {ry:.6f})\n") # Wyświetlenie wyniku (typ: None)

def do_solve(args):
    # Funkcja obsługująca polecenie 'solve' z CLI
    # Argumenty:
    #   args: obiekt z argumentami (ma atrybuty A i b jako listy stringów) (typ: argparse.Namespace)
    """Rozwiązuje prosty układ A*x=b."""
    
    # Konwersja listy stringów (args.A) na tablicę NumPy float32
    A = np.array([float(x) for x in args.A], dtype=np.float32) # (typ: np.ndarray)
    
    # Konwersja listy stringów (args.b) na tablicę NumPy float32
    b = np.array([float(x) for x in args.b], dtype=np.float32) # (typ: np.ndarray)
    
    # Wyliczenie wymiaru n pierwiastkując całkowitą liczbę elementów A
    n = int(math.sqrt(len(A)))  # (typ: int)
    
    # Zmiana kształtu A na macierz kwadratową (n, n)
    A = A.reshape((n, n)) # (typ: np.ndarray)
    
    # Zmiana kształtu b na wektor o długości n
    b = b.reshape((n,)) # (typ: np.ndarray)
    
    # Rozwiązanie układu (wynik to tensor, kolumna n x 1)
    x = solve_linear_system(A, b) # (typ: tf.Tensor)
    
    print("\n🔹 Rozwiązanie układu A x = b:") # (typ: None)
    # Wyświetlenie wyniku po spłaszczeniu i konwersji na NumPy
    print("   x =", tf.reshape(x, (-1,)).numpy(), "\n") # (typ: None)

def do_solve_cli(args):
    # Funkcja obsługująca polecenie 'solve-cli'
    # Argumenty:
    #   args: obiekt z argumentami (typ: argparse.Namespace)
    """Rozwiązuje Ax=b z parametrami CLI."""
    
    # Konwersja listy stringów na listę floatów dla A
    A_flat = [float(x) for x in args.A] # (typ: list[float])
    
    # Konwersja listy stringów na listę floatów dla b
    b_vec = [float(x) for x in args.b] # (typ: list[float])
    
    # Wywołanie funkcji rozwiązującej (zwraca tensor)
    x = solve_linear_system_cli(A_flat, b_vec) # (typ: tf.Tensor)
    
    print("\n🔹 Wynik (tryb CLI):", x.numpy(), "\n") # Wyświetlenie wyniku (typ: None)

# ============================================================
# Funkcja główna programu
# ============================================================
def main():
    # Główna funkcja konfigurująca parser argumentów
    """Tworzy parser argumentów i wywołuje odpowiednią funkcję."""
    
    # Utworzenie głównego parsera argumentów
    parser = argparse.ArgumentParser(
        description="Zadania 1–5: TensorFlow – obrót punktu i układy równań"
    ) # (typ: argparse.ArgumentParser)
    
    # Utworzenie podzbioru poleceń (subkomend)
    sub = parser.add_subparsers(dest="cmd", required=True) # (typ: argparse._SubParsersAction)

    # --- Konfiguracja podkomendy 'rotate' ---
    pr = sub.add_parser("rotate", help="Obrót punktu (x, y).") # (typ: argparse.ArgumentParser)
    pr.add_argument("--x", type=float, required=True, help="Współrzędna X punktu.") # (argument typu float)
    pr.add_argument("--y", type=float, required=True, help="Współrzędna Y punktu.") # (argument typu float)
    pr.add_argument("--angle", type=float, required=True, help="Kąt obrotu.") # (argument typu float)
    pr.add_argument("--degrees", action="store_true", help="Interpretuj kąt w stopniach.") # (flaga logiczna bool)
    pr.set_defaults(func=do_rotate) # Przypisanie funkcji obsługującej do komendy

    # --- Konfiguracja podkomendy 'solve' ---
    ps = sub.add_parser("solve", help="Rozwiązywanie układu Ax=b.") # (typ: argparse.ArgumentParser)
    ps.add_argument("--A", nargs="+", required=True, help="Elementy macierzy A (wierszami).") # (lista argumentów string)
    ps.add_argument("--b", nargs="+", required=True, help="Elementy wektora b.") # (lista argumentów string)
    ps.set_defaults(func=do_solve) # Przypisanie funkcji obsługującej do komendy

    # --- Konfiguracja podkomendy 'solve-cli' ---
    pc = sub.add_parser("solve-cli", help="Rozwiązywanie Ax=b z parametrami CLI.") # (typ: argparse.ArgumentParser)
    pc.add_argument("--A", nargs="+", required=True) # (lista argumentów string)
    pc.add_argument("--b", nargs="+", required=True) # (lista argumentów string)
    pc.set_defaults(func=do_solve_cli) # Przypisanie funkcji obsługującej do komendy

    # Parsowanie argumentów z linii poleceń
    args = parser.parse_args() # (typ: argparse.Namespace)
    
    # Wywołanie odpowiedniej funkcji przypisanej do podkomendy (do_rotate, do_solve, lub do_solve_cli)
    args.func(args) # (wywołanie funkcji dynamicznie)

# ============================================================
# Punkt wejścia programu
# ============================================================
if __name__ == "__main__":
    main() # Uruchomienie funkcji main, jeśli skrypt jest uruchamiany bezpośrednio
