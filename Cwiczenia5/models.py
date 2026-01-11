
import tensorflow as tf  # Import biblioteki TensorFlow (typ: module) - framework uczenia maszynowego
from tensorflow.keras import Sequential, layers  # Import klasy Sequential i modułu layers (typ: class, module)

def build_model(hp):
    """
    Buduje model Keras (Dense lub CNN) zgodny z Keras Tuner.
    Argument 'hp' to obiekt HyperParameters z library keras-tuner.
    """
    # Inicjalizacja pustego modelu sekwencyjnego (typ: keras.engine.sequential.Sequential)
    model = Sequential()
    
    # Warstwa wejściowa - explicite zdefiniowana
    # shape=(28, 28, 1) oznacza obraz 28x28 pikseli z 1 kanałem koloru (skala szarości)
    model.add(layers.Input(shape=(28, 28, 1)))  # (typ: keras.engine.base_layer.Layer)
    
    # --- BONUS: Augmentacja Danych ---
    # Sprawdzenie warunku logicznego zdefiniowanego w hyperparametrach
    # hp.Boolean zwraca wartość True lub False (typ: bool)
    if hp.Boolean("use_augmentation", default=True):
        # Dodanie warstwy losowego odbicia w poziomie (typ: keras.layers.preprocessing.image_preprocessing.RandomFlip)
        model.add(layers.RandomFlip("horizontal"))
        # Dodanie warstwy losowego obrotu o max 10% (typ: keras.layers.preprocessing.image_preprocessing.RandomRotation)
        model.add(layers.RandomRotation(0.1))
        # Dodanie warstwy losowego przybliżenia o max 10% (typ: keras.layers.preprocessing.image_preprocessing.RandomZoom)
        model.add(layers.RandomZoom(0.1))

    # Wybór architektury: Dense (MLP) vs CNN
    # hp.Choice wybiera jedną wartość z listy stringów (typ: str)
    model_type = hp.Choice("model_type", ["dense", "cnn"])
    
    # Warunek sprawdzający wybraną architekturę
    if model_type == "dense":
        # Architektura oparta o warstwy gęste (Fully Connected)
        
        # Spłaszczenie wejścia 2D (28, 28, 1) do wektora 1D (784,) (typ: keras.layers.core.flatten.Flatten)
        model.add(layers.Flatten())
        
        # Pętla for iterująca przez liczbę warstw ukrytych wybraną przez tunera
        # hp.Int zwraca liczbę całkowitą z zakresu 1-3 (typ: int)
        for i in range(hp.Int("dense_layers", 1, 3)):
            # Dodanie warstwy gęstej (Dense)
            # units: liczba neuronów (int, od 32 do 256, krok 32)
            # activation: funkcja aktywacji ("relu")
            model.add(layers.Dense(
                units=hp.Int(f"dense_units_{i}", 32, 256, step=32),  # (typ: int)
                activation="relu"  # (typ: str)
            ))  # (typ: keras.layers.core.dense.Dense)
            
            # Opcjonalne dodanie warstwy Dropout dla regularyzacji (zapobieganie overfittingowi)
            # hp.Boolean zwraca bool (typ: bool)
            if hp.Boolean("use_dropout", default=False):
                # Warstwa porzucająca 20% neuronów (typ: keras.layers.regularization.dropout.Dropout)
                model.add(layers.Dropout(0.2))
                
    else:
        # Architektura Konwolucyjna (CNN)
        
        # Pętla iterująca przez liczbę bloków konwolucyjnych (1-3) (typ: int)
        for i in range(hp.Int("cnn_blocks", 1, 3)):
            # Dodanie warstwy konwolucyjnej 2D
            model.add(layers.Conv2D(
                filters=hp.Int(f"filters_{i}", 16, 64, step=16),  # Liczba filtrów (typ: int)
                kernel_size=(3, 3),  # Rozmiar okna splotu (typ: tuple[int, int])
                activation="relu",  # Funkcja aktywacji (typ: str)
                padding="same"  # Sposób traktowania krawędzi (typ: str)
            ))  # (typ: keras.layers.convolutional.conv2d.Conv2D)
            
            # Dodanie warstwy Max Pooling (zmniejszanie wymiarowości)
            # pool_size=(2, 2) zmniejsza obraz dwukrotnie (typ: keras.layers.pooling.max_pooling2d.MaxPooling2D)
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        
        # Spłaszczenie danych po operacjach konwolucyjnych (typ: keras.layers.core.flatten.Flatten)
        model.add(layers.Flatten())
        
        # Dodanie warstwy gęstej po spłaszczeniu (typ: keras.layers.core.dense.Dense)
        model.add(layers.Dense(64, activation="relu"))
    
    # Warstwa wyjściowa - 10 klas (Fashion MNIST)
    # 10 neuronów odpowiadających prawdopodobieństwom każdej klasy
    # activation="softmax" zapewnia, że suma wyjść wynosi 1 (typ: keras.layers.core.dense.Dense)
    model.add(layers.Dense(10, activation="softmax"))
    
    # Pobranie współczynnika uczenia z tunera (logarytmicznie)
    learning_rate = hp.Float("lr", 1e-4, 1e-2, sampling="log")  # (typ: float)
    
    # Kompilacja modelu
    # Użycie optymalizatora Adam ze zmiennym learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # (typ: keras.optimizers.adam.Adam)
        loss="sparse_categorical_crossentropy",  # Funkcja straty dla etykiet typu int (typ: str)
        metrics=["accuracy"]  # Metryki do monitorowania (typ: list[str])
    )
    
    # Zwrócenie skompilowanego modelu
    return model  # (typ: keras.engine.sequential.Sequential)
