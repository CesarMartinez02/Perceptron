import random

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, max_epochs=100):
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        self.bias = random.uniform(-1, 1)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def activation(self, x):
        """Función de activación"""
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        """Cálculo de salida"""
        summation = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self.activation(summation)

    def train(self, X, y):
        """Entrenamiento"""
        for _ in range(self.max_epochs):
            error_count = 0
            for inputs, target in zip(X, y):
                prediction = self.predict(inputs)
                error = target - prediction

                if error != 0:
                    # Ajustar pesos y sesgo
                    self.weights = [w + self.learning_rate * error * x for w, x in zip(self.weights, inputs)]
                    self.bias += self.learning_rate * error
                    error_count += 1
            
            if error_count == 0:
                break

#Datos para entrenamiento
logic_gates = {
    "AND": {
        "X": [[0, 0], [0, 1], [1, 0], [1, 1]],
        "y": [0, 0, 0, 1]
    },
    "OR": {
        "X": [[0, 0], [0, 1], [1, 0], [1, 1]],
        "y": [0, 1, 1, 1]
    }
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
