# ============================================================
#
#     python test_tasks.py
#
# ============================================================

import math
import numpy as np
import tensorflow as tf

from main import rotate_point, solve_linear_system, solve_linear_system_cli


# -------------------------
# Zadania 1–2
# -------------------------
def test_rotate_point():
    print("===== TESTY ROTACJI PUNKTU =====")

    res1 = rotate_point(1.0, 0.0, tf.constant(math.pi / 2, tf.float32))  # 90°
    print("Test 1:", res1.numpy(), "→ oczekiwane [0, 1]")

    res2 = rotate_point(1.0, 1.0, tf.constant(math.pi, tf.float32))      # 180°
    print("Test 2:", res2.numpy(), "→ oczekiwane [-1, -1]")

    res3 = rotate_point(1.0, 0.0, tf.constant(3 * math.pi / 2, tf.float32))  # 270°
    print("Test 3:", res3.numpy(), "→ oczekiwane [0, -1]")

    assert np.allclose(res1, [0.0, 1.0], atol=1e-6)
    assert np.allclose(res2, [-1.0, -1.0], atol=1e-6)
    assert np.allclose(res3, [0.0, -1.0], atol=1e-6)
    print("Wszystkie testy rotacji OK\n")


# -------------------------------
# Zadanie 3
# -------------------------------
def test_solve_linear_system():
    print("===== TESTY UKŁADU RÓWNAŃ (A*x=b) =====")

    # Test 1: 2x2 z całkowitym rozwiązaniem [1, 2]
    # Uwaga: b zmienione na [7, 5], aby wynik był [1, 2]
    A1 = tf.constant([[3.0, 2.0], [1.0, 2.0]], tf.float32)
    b1 = tf.constant([7.0, 5.0], tf.float32)
    x1 = tf.reshape(solve_linear_system(A1, b1), (-1,))
    print("Test 1:", x1.numpy(), "→ oczekiwane [1, 2]")
    assert np.allclose(x1, [1.0, 2.0], atol=1e-6)

    # Test 2: inny układ 2x2 → [1, 2]
    A2 = tf.constant([[2.0, 1.0], [5.0, 3.0]], tf.float32)
    b2 = tf.constant([4.0, 11.0], tf.float32)
    x2 = tf.reshape(solve_linear_system(A2, b2), (-1,))
    print("Test 2:", x2.numpy(), "→ oczekiwane [1, 2]")
    assert np.allclose(x2, [1.0, 2.0], atol=1e-6)

    # Test 3: klasyczny 3x3 → [2, 3, -1]
    A3 = tf.constant(
        [[2.0, 1.0, -1.0],
         [-3.0, -1.0, 2.0],
         [-2.0, 1.0, 2.0]], tf.float32
    )
    b3 = tf.constant([8.0, -11.0, -3.0], tf.float32)
    x3 = tf.reshape(solve_linear_system(A3, b3), (-1,))
    print("Test 3:", x3.numpy(), "→ oczekiwane [2, 3, -1]")
    assert np.allclose(x3, [2.0, 3.0, -1.0], atol=1e-6)

    print("Wszystkie testy układu równań OK\n")


# -------------------------------------------------
# Zadanie 4
# -------------------------------------------------
def test_solve_linear_system_cli():
    print("===== TESTY WERSJI CLI (A_flat, b_vec) =====")

    # Test 1: 2x2 z wynikiem [1, 2]
    A_flat1 = [3, 2, 1, 2]
    b_vec1 = [7, 5]  # zmienione z [5,5], by wynik był [1,2]
    x1 = solve_linear_system_cli(A_flat1, b_vec1)
    print("Test 1:", x1.numpy(), "→ oczekiwane [1, 2]")
    assert np.allclose(x1, [1.0, 2.0], atol=1e-6)

    # Test 2: inny 2x2 → [1, 2]
    A_flat2 = [2, 1, 5, 3]
    b_vec2 = [4, 11]
    x2 = solve_linear_system_cli(A_flat2, b_vec2)
    print("Test 2:", x2.numpy(), "→ oczekiwane [1, 2]")
    assert np.allclose(x2, [1.0, 2.0], atol=1e-6)

    # Test 3: 3x3 → [2, 3, -1]
    A_flat3 = [2, 1, -1, -3, -1, 2, -2, 1, 2]
    b_vec3 = [8, -11, -3]
    x3 = solve_linear_system_cli(A_flat3, b_vec3)
    print("Test 3:", x3.numpy(), "→ oczekiwane [2, 3, -1]")
    assert np.allclose(x3, [2.0, 3.0, -1.0], atol=1e-6)

    print("Wszystkie testy CLI OK\n")


def main():
    print("===================================")
    print("URUCHAMIANIE TESTÓW FUNKCJI")
    print("===================================")
    test_rotate_point()
    test_solve_linear_system()
    test_solve_linear_system_cli()
    print("Wszystkie testy przeszły pomyślnie!")
    print("===================================\n")


if __name__ == "__main__":
    main()
