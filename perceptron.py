import random


class Perceptron:
    """
    Implementación de un Perceptrón simple.
    Atributos:
    ----------
    weights : list
        Lista de pesos inicializados aleatoriamente.
    bias : float
        Sesgo inicializado aleatoriamente.
    learning_rate : float
        Tasa de aprendizaje para el ajuste de pesos y sesgo.
    max_epochs : int
        Número máximo de épocas para el entrenamiento.
    Métodos:
    --------
    __init__(input_size, learning_rate=0.1, max_epochs=100):
        Inicializa el perceptrón con los parámetros dados.
    activation(x):
        Función de activación que devuelve 1 si x >= 0, de lo contrario 0.
    predict(inputs):
        Calcula la salida del perceptrón para una entrada dada.
    train(X, y):
        Entrena el perceptrón usando el conjunto de datos X y las etiquetas y.
    """

    def __init__(self, input_size, learning_rate=0.1, max_epochs=100):
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        self.bias = random.uniform(-1, 1)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def activation(self, x):
        """
        Función de activación para el perceptrón.
        Esta función toma un valor de entrada y devuelve 1 si el valor es mayor o igual a 0,
        de lo contrario, devuelve 0.
        Parámetros:
        x (float): El valor de entrada para la función de activación.
        Retorna:
        int: 1 si x es mayor o igual a 0, de lo contrario 0.
        """
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        """
        Predice la salida para una entrada dada utilizando el modelo de perceptrón.
        Args:
            inputs (list o array): Lista o array de valores de entrada.
        Returns:
            int: La salida predicha después de aplicar la función de activación.
        """

        summation = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self.activation(summation)

    def train(self, X, y):
        """
        Entrena el perceptrón utilizando el conjunto de datos proporcionado.
        Parámetros:
        X (list of list of float): Conjunto de características de entrada.
        y (list of int): Conjunto de etiquetas objetivo.
        Este método ajusta los pesos y el sesgo del perceptrón en función de los errores
        cometidos en las predicciones. El entrenamiento se realiza durante un número
        máximo de épocas especificado por `self.max_epochs`. Si no se cometen errores
        durante una época, el entrenamiento se detiene antes de alcanzar el número máximo
        de épocas.
        """

        for _ in range(self.max_epochs):
            error_count = 0
            for inputs, target in zip(X, y):
                prediction = self.predict(inputs)
                error = target - prediction

                if error != 0:
                    # Ajustar pesos y sesgo
                    self.weights = [
                        w + self.learning_rate * error * x
                        for w, x in zip(self.weights, inputs)
                    ]  # Regla de aprendizaje del perceptrón
                    self.bias += self.learning_rate * error
                    error_count += 1

            if error_count == 0:
                break


# Datos para entrenamiento
logic_gates = {
    "AND": {"X": [[0, 0], [0, 1], [1, 0], [1, 1]], "y": [0, 0, 0, 1]},
    "OR": {"X": [[0, 0], [0, 1], [1, 0], [1, 1]], "y": [0, 1, 1, 1]},
}


operation = input("Ingrese la operación a hacer (AND/OR): ").strip().upper()
if operation not in logic_gates:
    print("Operación no válida")
    exit()

X, y = logic_gates[operation]["X"], logic_gates[operation]["y"]

# Entrenamiento
perceptron = Perceptron(input_size=2, learning_rate=0.1, max_epochs=100)
perceptron.train(X, y)

print(f"\nPerceptrón entrenado para {operation}\n")
for inputs in X:
    print(f"Entrada: {inputs} -> Predicción: {perceptron.predict(inputs)}")
