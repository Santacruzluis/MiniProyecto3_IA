import os
import numpy as np
import librosa
import soundfile as sf
from tkinter import Tk, Button, Label, filedialog, messagebox

# Función para extraer características de una canción
def extract_features(file_path):
    y, sr = sf.read(file_path)
    y = y.astype(np.float32)
    y = librosa.util.normalize(y)

    # Extraer características
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_mean = np.mean(mel_spectrogram_db, axis=1)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    energy = np.sum(y**2)

    features = np.hstack([mfccs_mean, mel_mean, tempo, zero_crossing_rate, energy])
    return features

# Normalización manual (reemplazo de StandardScaler)
def manual_scaler(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

# Codificación manual de etiquetas (reemplazo de LabelEncoder)
def manual_label_encoder(y):
    unique_labels = np.unique(y)
    label_to_code = {label: i for i, label in enumerate(unique_labels)}
    y_encoded = np.array([label_to_code[label] for label in y])
    return y_encoded, label_to_code

# División manual del dataset (reemplazo de train_test_split)
def manual_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    test_samples = int(X.shape[0] * test_size)
    
    X_train = X[indices[test_samples:]]
    X_test = X[indices[:test_samples]]
    y_train = y[indices[test_samples:]]
    y_test = y[indices[:test_samples]]
    
    return X_train, X_test, y_train, y_test

# Cargar el dataset completo
dataset_path = 'genres'  # Cambia esto a la ruta de tu dataset
genres = os.listdir(dataset_path)
X = []
y = []

for genre in genres:
    genre_path = os.path.join(dataset_path, genre)
    for file_name in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file_name)
        features = extract_features(file_path)
        X.append(features)
        y.append(genre)

# Convertir a arrays de numpy
X = np.array(X)
y = np.array(y)

# Codificar las etiquetas manualmente
y_encoded, label_to_code = manual_label_encoder(y)

# Normalizar las características manualmente
X_scaled, mean, std = manual_scaler(X)

# Dividir el dataset manualmente
X_train, X_test, y_train, y_test = manual_train_test_split(X_scaled, y_encoded, test_size=0.4, random_state=42)

# Implementación del MLP desde cero
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicialización de pesos y sesgos
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Propagación hacia adelante
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output, learning_rate):
        # Propagación hacia atrás
        error = output - y
        d_output = error * self.sigmoid_derivative(output)
        d_hidden = np.dot(d_output, self.W2.T) * self.sigmoid_derivative(self.a1)

        # Actualización de pesos y sesgos
        self.W2 -= learning_rate * np.dot(self.a1.T, d_output)
        self.b2 -= learning_rate * np.sum(d_output, axis=0)
        self.W1 -= learning_rate * np.dot(X.T, d_hidden)
        self.b1 -= learning_rate * np.sum(d_hidden, axis=0)

    def train(self, X, y, epochs, learning_rate):
        # Entrenamiento del modelo
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)

    def predict(self, X):
        # Predicción
        output = self.forward(X)
        return np.argmax(output, axis=1)

# Crear y entrenar el MLP
input_size = X_train.shape[1]
hidden_size = 10
output_size = len(np.unique(y_train))

mlp = MLP(input_size, hidden_size, output_size)
mlp.train(X_train, np.eye(output_size)[y_train], epochs=1000, learning_rate=0.01)

# Función para calcular la precisión
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Predecir en los conjuntos de entrenamiento, validación y prueba
y_pred_train = mlp.predict(X_train)
y_pred_test = mlp.predict(X_test)

# Calcular la precisión
train_accuracy = accuracy(y_train, y_pred_train)
test_accuracy = accuracy(y_test, y_pred_test)

# Mostrar las métricas de evaluación
print(f"Precisión en entrenamiento: {train_accuracy:.2f}")
print(f"Precisión en prueba: {test_accuracy:.2f}")

# Función para generar la matriz de confusión
def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)))
    for i in range(len(y_true)):
        matrix[y_true[i], y_pred[i]] += 1
    return matrix

# Generar y mostrar la matriz de confusión para el conjunto de prueba
conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Matriz de confusión (Prueba):")
print(conf_matrix)

# Función para predecir el género de una canción específica
def predict_genre(model, mean, std, label_to_code, file_path):
    # Extraer características de la canción
    features = extract_features(file_path)
    
    # Normalizar las características
    features_scaled = (features - mean) / std
    
    # Predecir el género
    prediction = model.predict([features_scaled])
    
    # Convertir la predicción numérica a la etiqueta de género
    code_to_label = {v: k for k, v in label_to_code.items()}
    predicted_genre = code_to_label[prediction[0]]
    
    return predicted_genre

# Interfaz gráfica para subir archivos
def upload_file():
    file_path = filedialog.askopenfilename(
        title="Selecciona un archivo de música",
        filetypes=[("Audio Files", "*.wav *.mp3 *.ogg *.flac *.au")]
    )
    
    if file_path:
        try:
            # Predecir el género de la canción
            predicted_genre = predict_genre(mlp, mean, std, label_to_code, file_path)
            messagebox.showinfo("Predicción", f"El género predicho es: {predicted_genre}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo procesar el archivo: {e}")

# Crear la ventana principal
root = Tk()
root.title("Clasificador de Géneros Musicales")

# Botón para subir archivo
upload_button = Button(root, text="Subir archivo de música", command=upload_file)
upload_button.pack(pady=20)

# Etiqueta de instrucciones
Label(root, text="Selecciona un archivo de música para predecir su género.").pack(pady=10)

# Iniciar la interfaz gráfica
root.mainloop()