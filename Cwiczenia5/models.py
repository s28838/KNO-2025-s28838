import tensorflow as tf
from tensorflow.keras import Sequential, layers


def build_model(hp):  # Konstruuje i kompiluje model Keras na podstawie przekazanych hiperparametrów.
    model = Sequential()  # Inicjalizuje pusty model sekwencyjny.
    
    # Warstwa wejściowa.
    # Obrazy Fashion MNIST mają wymiar 28x28 pikseli i 1 kanał (skala szarości).
    model.add(layers.Input(shape=(28, 28, 1)))  # Definiuje kształt danych wejściowych.
    
    # --- Sekcja Augmentacji Danych ---
    # Zwiększanie różnorodności danych przez losowe modyfikacje.
    # Zapobiega to zapamiętywaniu konkretnych przykładów przez sieć (overfitting).
    if hp.Boolean("use_augmentation", default=True):  # Sprawdza, czy augmentacja jest włączona w hiperparametrach.
        # Losowe odbicie w poziomie (sensowne dla ubrań, np. koszulka czy but).
        model.add(layers.RandomFlip("horizontal"))  # Dodaje warstwę losowego odbicia poziomego.
        # Delikatny obrót (ok. +/- 36 stopni).
        model.add(layers.RandomRotation(0.1))  # Dodaje warstwę losowego obrotu.
        # Losowe skalowanie (zoom +/- 10%).
        model.add(layers.RandomZoom(0.1))  # Dodaje warstwę losowego skalowania.

    # Wybór fundamentu architektury: "dense" (MLP) lub "cnn" (ConvNet).
    model_type = hp.Choice("model_type", ["dense", "cnn"])  # Losuje typ architektury.
    
    if model_type == "dense":  # Obsługuje przypadek architektury gęstej (MLP).
        # === Architektura Dense (MLP) ===
        # Prosta sieć oparta na spłaszczonym wektorze.
        
        # Spłaszczenie obrazu 2D do wektora 1D (28*28 = 784 wejścia).
        model.add(layers.Flatten())  # Dodaje warstwę spłaszczającą.
        
        # Dynamiczna liczba warstw ukrytych (od 1 do 3).
        for i in range(hp.Int("dense_layers", 1, 3)):  # Iteruje przez wylosowaną liczbę warstw.
            # Warstwa gęsta z aktywacją ReLU.
            # Liczba neuronów jest dobierana dynamicznie w zakresie 32-256.
            model.add(layers.Dense(  # Dodaje warstwę gęstą.
                units=hp.Int(f"dense_units_{i}", 32, 256, step=32),  # Losuje liczbę neuronów.
                activation="relu"  # Ustawia funkcję aktywacji ReLU.
            ))
            
            # Opcjonalny Dropout.
            # Losowo zeruje 20% wyjść neuronów, zmuszając sieć do korzystania z innych ścieżek.
            if hp.Boolean("use_dropout", default=False):  # Sprawdza czy włączyć Dropout.
                model.add(layers.Dropout(0.2))  # Dodaje warstwę Dropout z p=0.2.
                
    else:  # Obsługuje przypadek architektury konwolucyjnej (CNN).
        # === Architektura CNN (ConvNet) ===
        # Sieć konwolucyjna, znacznie lepiej radząca sobie z danymi obrazkowymi
        # dzięki zachowaniu struktury przestrzennej i wykrywaniu lokalnych cech.
        
        # Dynamiczna liczba bloków konwolucyjnych (Conv2D + MaxPooling).
        for i in range(hp.Int("cnn_blocks", 1, 3)):  # Iteruje przez wylosowaną liczbę bloków.
            # Warstwa splotowa (Conv2D).
            # Uczy się filtrów (cech) takich jak krawędzie, tekstury.
            # padding='same' utrzymuje rozmiar obrazu na wyjściu (np. 28x28).
            model.add(layers.Conv2D(  # Dodaje warstwę konwolucyjną.
                filters=hp.Int(f"filters_{i}", 16, 64, step=16),  # Losuje liczbę filtrów.
                kernel_size=(3, 3),  # Ustawia rozmiar jądra splotu.
                activation="relu",  # Ustawia  aktywację ReLU.
                padding="same"  # Zachowuje wymiary przestrzenne.
            ))
            
            # Warstwa pulingu (MaxPooling).
            # Redukuje wymiary przestrzenne (downsampling) o połowę, wybierając wartość maksymalną.
            # Zmniejsza ilość obliczeń i zapewnia inwariantność na małe przesunięcia.
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # Dodaje warstwę MaxPooling.
        
        # Po przejściu przez warstwy splotowe, mapy cech są spłaszczane.
        model.add(layers.Flatten())  # Spłaszcza mapy cech do wektora.
        
        # Dodatkowa warstwa gęsta przetwarzająca wyekstrahowane cechy przed klasyfikacją.
        model.add(layers.Dense(64, activation="relu"))  # Dodaje warstwę gęstą po konwolucjach.
    
    # Warstwa Wyjściowa.
    # 10 neuronów odpowiadających 10 klasom ubrań.
    # Funkcja Softmax zamienia wyniki na rozkład prawdopodobieństwa (suma = 1.0).
    model.add(layers.Dense(10, activation="softmax"))  # Dodaje warstwę wyjściową.
    
    # Dobór współczynnika uczenia (Learning Rate).
    # Skala logarytmiczna jest naturalna dla tego parametru (np. 0.01, 0.001, 0.0001).
    learning_rate = hp.Float("lr", 1e-4, 1e-2, sampling="log")  # Losuje learning rate.
    
    # Kompilacja modelu.
    # Optimizer: Adam (Adaptive Moment Estimation) - standard w DL.
    # Loss: sparse_categorical_crossentropy - bo etykiety to liczby całkowite (nie one-hot).
    model.compile(  # Kompiluje model.
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Ustawia optymalizator.
        loss="sparse_categorical_crossentropy",  # Ustawia funkcję straty.
        metrics=["accuracy"]  # Ustawia metrykę.
    )
    
    return model  # Zwraca gotowy model.
