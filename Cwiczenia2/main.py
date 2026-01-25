import argparse
import math

import numpy as np
import tensorflow as tf


def rotate_point(x, y, angle_rad):  # Definiuje funkcję rotate_point przyjmującą współrzędne i kąt, służy do obliczania obrotu punktu.
    # Rzutowanie na typ float64 zapewnia wyższą precyzję obliczeń trygonometrycznych.
    angle = tf.cast(angle_rad, tf.float64)  # Rzutuje kąt na typ float64, zapewnia precyzję niezbędną dla funkcji trygonometrycznych.
    x = tf.cast(x, tf.float64)  # Rzutuje współrzędną x na float64, ujednolica typy danych w obliczeniach.
    y = tf.cast(y, tf.float64)  # Rzutuje współrzędną y na float64, ujednolica typy danych w obliczeniach.

    c = tf.cos(angle)  # Oblicza cosinus kąta obrotu, składowa macierzy rotacji.
    s = tf.sin(angle)  # Oblicza sinus kąta obrotu, składowa macierzy rotacji.

    # Konstrukcja macierzy rotacji 2x2.
    # tf.stack łączy skalary w tensory.
    row1 = tf.stack([c, -s])  # Tworzy pierwszy wiersz macierzy rotacji [cos, -sin], element transformacji.
    row2 = tf.stack([s, c])  # Tworzy drugi wiersz macierzy rotacji [sin, cos], element transformacji.
    R = tf.stack([row1, row2])  # Łączy wiersze w pełną macierz rotacji 2x2, tensor reprezentujący obrót.

    p = tf.stack([x, y])  # Tworzy wektor punktu [x, y], reprezentuje punkt do przekształcenia.

    # Mnożenie macierzy R przez wektor kolumnowy p.
    # Wynikiem jest wektor po transformacji liniowej.
    return tf.linalg.matvec(R, p)  # Mnoży macierz rotacji R przez wektor p, zwraca nowy obrócony punkt.


@tf.function  # Dekorator tf.function kompiluje funkcję do grafu, optymalizuje wydajność obliczeń.
def solve_linear_system(A, b):  # Definiuje funkcję solve_linear_system, rozwiązuje układ równań liniowych Ax=b.
    # Konwersja na float32 - standardowy typ dla operacji ML/GPU w TensorFlow.
    A = tf.cast(A, tf.float32)  # Rzutuje macierz A na float32, standardowy typ dla operacji numerycznych GPU/TPU.
    b_cast = tf.cast(b, tf.float32)  # Rzutuje wektor b na float32, zapewnia zgodność typów z macierzą A.
    
    # Zmiana kształtu wektora b na wektor kolumnowy (n, 1), 
    # co jest wymagane przez funkcję tf.linalg.solve.
    b = tf.reshape(b_cast, (-1, 1))  # Przekształca wektor b w wektor kolumnowy, wymagane przez operację solve.
    
    # Numeryczne rozwiązywanie układu (zazwyczaj metodą eliminacji Gaussa lub rozkładu LU).
    return tf.linalg.solve(A, b)  # Rozwiązuje układ równań liniowych, zwraca wektor rozwiązań x.


@tf.function  # Dekorator tf.function dla funkcji pomocniczej CLI, optymalizacja wykonania.
def solve_linear_system_cli(A_flat, b_vec):  # Definiuje wrapper dla CLI przyjmujący płaskie listy, ułatwia obsługę danych wejściowych z konsoli.
    # Dynamiczne wyznaczenie wymiaru macierzy n na podstawie liczby elementów.
    # Zakładamy, że macierz jest kwadratowa (n*n elementów).
    n = int(math.sqrt(len(A_flat)))  # Oblicza wymiar N macierzy kwadratowej z długości listy, pozwala odtworzyć kształt macierzy.

    # Rekonstrukcja macierzy 2D (n, n) z płaskiej listy.
    A_np = np.array(A_flat, np.float32).reshape((n, n))  # Tworzy macierz NumPy z listy i nadaje jej kształt NxN, przygotowanie danych.
    A = tf.constant(A_np)  # Konwertuje macierz NumPy na tensor TensorFlow, dane wejściowe dla obliczeń TF.

    b_np = np.array(b_vec, np.float32)  # Tworzy tablicę NumPy z listy wektora b, przygotowanie danych.
    b = tf.constant(b_np)  # Konwertuje wektor na tensor TensorFlow, dane wejściowe dla obliczeń TF.

    result = solve_linear_system(A, b)  # Wywołuje główną funkcję rozwiązującą układ, wykonuje właściwe obliczenia.
    
    # Spłaszczenie wyniku z powrotem do wektora 1D dla łatwiejszego wyświetlania.
    return tf.reshape(result, (-1,))  # Spłaszcza wynikowy wektor kolumnowy do 1D, ułatwia wyświetlanie wyniku w konsoli.


# ------------------------------------------------------------------------------
# Funkcje pomocnicze dla podkomend CLI
# ------------------------------------------------------------------------------

def do_rotate(args):  # Funkcja obsługująca komendę 'rotate', przetwarza argumenty i wywołuje logikę biznesową.
    # Konwersja stopni na radiany, jeśli użytkownik użył flagi --degrees.
    angle = math.radians(args.angle) if args.degrees else args.angle  # Zamienia stopnie na radiany jeśli flaga ustawiona, normalizacja danych wejściowych.

    t_angle = tf.constant(angle, tf.float64)  # Tworzy tensor kąta, przygotowanie do przekazania do funkcji TensorFlow.
    res_tensor = rotate_point(args.x, args.y, t_angle)  # Wywołuje logikę obrotu punktu, właściwe obliczenia.
    res = res_tensor.numpy()  # Pobiera wartość z tensora do tablicy NumPy, umożliwia odczyt wyniku.

    rx, ry = float(res[0]), float(res[1])  # Rozpakowuje współrzędne wynikowe do zmiennych float, przygotowanie do wyświetlenia.

    print(f"\n🔹 Punkt ({args.x}, {args.y}) po obrocie o {args.angle}{'°' if args.degrees else ' rad'}:")  # Wypisuje nagłówek z danymi wejściowymi, informacja dla użytkownika.
    print(f"   Wynik → ({rx:.6f}, {ry:.6f})\n")  # Wypisuje sformatowany wynik obrotu, ostateczny rezultat operacji.


def do_solve(args):  # Funkcja obsługująca komendę 'solve', przetwarza argumenty macierzowe i wywołuje solver.
    A = np.array([float(x) for x in args.A], dtype=np.float32)  # Konwertuje listę stringów argumentu A na tablicę floatów, parsowanie danych wejściowych.
    b = np.array([float(x) for x in args.b], dtype=np.float32)  # Konwertuje listę stringów argumentu b na tablicę floatów, parsowanie danych wejściowych.

    n = int(math.sqrt(len(A)))  # Oblicza wymiar macierzy pierwiastkując ilość elementów, dedukcja kształtu danych.
    A = A.reshape((n, n))  # Nadaje macierzy A właściwy kształt 2D, przygotowanie struktury macierzy.
    b = b.reshape((n,))  # Nadaje wektorowi b właściwy kształt, przygotowanie wektora wyrazów wolnych.

    x = solve_linear_system(A, b)  # Rozwiązuje układ równań wywołując funkcję TensorFlow, właściwe obliczenia.

    print("\n🔹 Rozwiązanie układu A x = b:")  # Wypisuje nagłówek wyniku, sekcja wyjściowa dla użytkownika.
    print("   x =", tf.reshape(x, (-1,)).numpy(), "\n")  # Wypisuje wynikowy wektor x przekonwertowany na NumPy, prezentacja wyniku.


def do_solve_cli(args):  # Funkcja obsługująca komendę 'solve-cli', alternatywny interfejs dla solvera.
    A_flat = [float(x) for x in args.A]  # Konwertuje argument A na listę floatów, przygotowanie danych dla wrappera.
    b_vec = [float(x) for x in args.b]  # Konwertuje argument b na listę floatów, przygotowanie danych dla wrappera.

    x = solve_linear_system_cli(A_flat, b_vec)  # Wywołuje wrapper solvera CLI, uruchomienie obliczeń.

    print("\n🔹 Wynik (tryb CLI):", x.numpy(), "\n")  # Wypisuje wynik obliczeń, prezentacja rezultatu.


def main():  # Główna funkcja programu, punkt wejścia.
    parser = argparse.ArgumentParser(  # Tworzy główny parser argumentów, konfiguracja CLI.
        description="Zadania: TensorFlow – obrót punktu i układy równań"  # Ustawia opis programu w pomocy, dokumentacja CLI.
    )

    sub = parser.add_subparsers(dest="cmd", required=True)  # Dodaje pod-parsery dla komend, umożliwia strukturę 'program komenda argumenty'.

    # Podkomenda: rotate
    pr = sub.add_parser("rotate", help="Obrót punktu (x, y).")  # Dodaje definicję komendy 'rotate', konfiguracja podkomendy.
    pr.add_argument("--x", type=float, required=True, help="Współrzędna X punktu.")  # Dodaje wymagany argument --x, wejście danych.
    pr.add_argument("--y", type=float, required=True, help="Współrzędna Y punktu.")  # Dodaje wymagany argument --y, wejście danych.
    pr.add_argument("--angle", type=float, required=True, help="Kąt obrotu.")  # Dodaje wymagany argument --angle, wejście danych.
    pr.add_argument("--degrees", action="store_true", help="Interpretuj kąt w stopniach.")  # Dodaje opcjonalną flagę --degrees, sterowanie trybem kąta.
    pr.set_defaults(func=do_rotate)  # Przypisuje funkcję do_rotate do tej komendy, routing wykonania.

    # Podkomenda: solve
    ps = sub.add_parser("solve", help="Rozwiązywanie układu Ax=b.")  # Dodaje definicję komendy 'solve', konfiguracja podkomendy.
    ps.add_argument(
        "--A", nargs="+", required=True, help="Elementy macierzy A (wierszami)."  # Dodaje argument A przyjmujący listę wartości, wejście macierzy.
    )
    ps.add_argument("--b", nargs="+", required=True, help="Elementy wektora b.")  # Dodaje argument b przyjmujący listę wartości, wejście wektora.
    ps.set_defaults(func=do_solve)  # Przypisuje funkcję do_solve do tej komendy, routing wykonania.

    # Podkomenda: solve-cli
    pc = sub.add_parser("solve-cli", help="Rozwiązywanie Ax=b z parametrami CLI.")  # Dodaje definicję komendy 'solve-cli', konfiguracja podkomendy.
    pc.add_argument("--A", nargs="+", required=True)  # Dodaje argument A, wejście danych.
    pc.add_argument("--b", nargs="+", required=True)  # Dodaje argument b, wejście danych.
    pc.set_defaults(func=do_solve_cli)  # Przypisuje funkcję do_solve_cli do tej komendy, routing wykonania.

    args = parser.parse_args()  # Parsuje argumenty z linii poleceń, przetwarza wejście użytkownika.
    args.func(args)  # Wywołuje przypisaną do komendy funkcję z sparsowanymi argumentami, uruchamia właściwą logikę.


if __name__ == "__main__":  # Sprawdza czy skrypt jest uruchamiany bezpośrednio, standardowy idiom Python.
    main()  # Wywołuje funkcję main, start programu.
