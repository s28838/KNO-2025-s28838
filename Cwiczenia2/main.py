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

import argparse
import math
import numpy as np
import tensorflow as tf


# ============================================================
# Zadania 1–2 – Obrót punktu
# ============================================================
def rotate_point(x, y, angle_rad):
    # wszystko w float64
    angle = tf.cast(angle_rad, tf.float64)
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)

    c = tf.cos(angle)
    s = tf.sin(angle)

    # buduj macierz przez tf.stack (nie tf.constant z tensorów)
    R = tf.stack([tf.stack([c, -s]), tf.stack([s, c])])  # (2,2), float64
    p = tf.stack([x, y])                                 # (2,),   float64
    return tf.linalg.matvec(R, p)                        # (2,),   float64

# ============================================================
# Zadanie 3 – Rozwiązywanie układu A × x = b
# ============================================================
@tf.function
def solve_linear_system(A, b):
    """
    Rozwiązuje układ równań A*x=b przy użyciu tf.linalg.solve.
    A – macierz (n×n)
    b – wektor (n)
    """
    A = tf.cast(A, tf.float32)
    b = tf.reshape(tf.cast(b, tf.float32), (-1, 1))  # zamiana b na kolumnę
    return tf.linalg.solve(A, b)  # zwraca kolumnę (n×1)

# ============================================================
# Zadanie 4 – Wersja CLI (dane wprowadzone przez użytkownika)
# ============================================================
@tf.function
def solve_linear_system_cli(A_flat, b_vec):
    """
    Rozwiązuje Ax=b, gdzie:
      A_flat – lista elementów macierzy A (płasko, wierszami)
      b_vec  – lista elementów wektora b
    Automatycznie oblicza rozmiar macierzy n×n.
    """
    n = int(math.sqrt(len(A_flat)))
    A = tf.constant(np.array(A_flat, np.float32).reshape((n, n)))
    b = tf.constant(np.array(b_vec, np.float32))
    return tf.reshape(solve_linear_system(A, b), (-1,))

# ============================================================
# Podkomendy programu
# ============================================================

def do_rotate(args):
    """Wykonuje obrót punktu."""
    angle = math.radians(args.angle) if args.degrees else args.angle # funkcje trygonometryczne w TensorFlow używają radianów.
    res = rotate_point(args.x, args.y, tf.constant(angle, tf.float64)).numpy()
    rx, ry = float(res[0]), float(res[1]) # Konwersja tensora na zwykłe liczby zmiennoprzecinkowe typu float
    print(f"\n🔹 Punkt ({args.x}, {args.y}) po obrocie o {args.angle}{'°' if args.degrees else ' rad'}:")
    print(f"   Wynik → ({rx:.6f}, {ry:.6f})\n")

def do_solve(args):
    """Rozwiązuje prosty układ A*x=b."""
    A = np.array([float(x) for x in args.A], dtype=np.float32)
    b = np.array([float(x) for x in args.b], dtype=np.float32)
    n = int(math.sqrt(len(A)))  # wyliczenie rozmiaru macierzy
    A = A.reshape((n, n))
    b = b.reshape((n,))
    x = solve_linear_system(A, b)
    print("\n🔹 Rozwiązanie układu A x = b:")
    print("   x =", tf.reshape(x, (-1,)).numpy(), "\n")

def do_solve_cli(args):
    """Rozwiązuje Ax=b z parametrami CLI."""
    A_flat = [float(x) for x in args.A]
    b_vec = [float(x) for x in args.b]
    x = solve_linear_system_cli(A_flat, b_vec)
    print("\n🔹 Wynik (tryb CLI):", x.numpy(), "\n")

# ============================================================
# Funkcja główna programu
# ============================================================
def main():
    """Tworzy parser argumentów i wywołuje odpowiednią funkcję."""
    parser = argparse.ArgumentParser(
        description="Zadania 1–5: TensorFlow – obrót punktu i układy równań"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Obrót punktu (zadania 1–2)
    pr = sub.add_parser("rotate", help="Obrót punktu (x, y).")
    pr.add_argument("--x", type=float, required=True, help="Współrzędna X punktu.")
    pr.add_argument("--y", type=float, required=True, help="Współrzędna Y punktu.")
    pr.add_argument("--angle", type=float, required=True, help="Kąt obrotu.")
    pr.add_argument("--degrees", action="store_true", help="Interpretuj kąt w stopniach.")
    pr.set_defaults(func=do_rotate)

    # Rozwiązywanie układu A*x=b (zadanie 3)
    ps = sub.add_parser("solve", help="Rozwiązywanie układu Ax=b.")
    ps.add_argument("--A", nargs="+", required=True, help="Elementy macierzy A (wierszami).")
    ps.add_argument("--b", nargs="+", required=True, help="Elementy wektora b.")
    ps.set_defaults(func=do_solve)

    # Wersja CLI (zadanie 4)
    pc = sub.add_parser("solve-cli", help="Rozwiązywanie Ax=b z parametrami CLI.")
    pc.add_argument("--A", nargs="+", required=True)
    pc.add_argument("--b", nargs="+", required=True)
    pc.set_defaults(func=do_solve_cli)

    # Parsowanie argumentów
    args = parser.parse_args()
    args.func(args)

# ============================================================
# Punkt wejścia programu
# ============================================================
if __name__ == "__main__":
    main()
