import os
import numpy as np
import librosa
import soundfile as sf
from tkinter import Tk, Button, Label, filedialog, messagebox

# ==================== [1. Extracción de Características Optimizada] ====================
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, duration=30)  # Limitar a 30s para consistencia
    except Exception as e:
        raise RuntimeError(f"Error al cargar el archivo: {str(e)}")
    
    y = librosa.util.normalize(y)  # Normalización correcta
    
    # MFCCs (20 coeficientes)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_var = np.var(mfccs, axis=1)
    
    # Espectrograma MEL
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_mean = np.mean(mel_spectrogram_db, axis=1)
    
    # Características temporales
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
    energy = np.mean(y**2)  # Energía normalizada
    
    # Características armónicas
    harmonic = librosa.effects.harmonic(y)
    harmonic_mean = np.mean(harmonic)
    
    features = np.hstack([
        mfccs_mean, 
        mfccs_var,
        mel_mean, 
        tempo, 
        zero_crossing_rate, 
        energy,
        harmonic_mean
    ])
    
    return features

# ==================== [2. Preprocesamiento Mejorado] ====================
class AudioPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None
        self.label_encoder = None
    
    def fit(self, X, y):
        # Manejo de desviación estándar cero
        epsilon = 1e-8
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + epsilon
        
        # Codificación de etiquetas
        unique_labels = np.unique(y)
        self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
    
    def transform(self, X, y=None):
        X_scaled = (X - self.mean) / self.std
        
        if y is not None:
            y_encoded = np.array([self.label_encoder[label] for label in y])
            return X_scaled, y_encoded
        
        return X_scaled
    
    def inverse_transform_labels(self, y):
        decoder = {v: k for k, v in self.label_encoder.items()}
        return np.array([decoder[label] for label in y])

# ==================== [3. MLP Mejorado con Softmax y Entropía Cruzada] ====================
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicialización He
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros(output_size)
        self.loss_history = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self._softmax(self.z2)
        return self.a2

    def _cross_entropy_loss(self, y, probs):
        m = y.shape[0]
        log_probs = -np.log(probs[range(m), y])
        return np.sum(log_probs) / m

    def backward(self, X, y, lr):
        m = X.shape[0]
        
        # Capa de salida
        delta3 = self.a2
        delta3[range(m), y] -= 1
        delta3 /= m
        
        # Capa oculta
        delta2 = np.dot(delta3, self.W2.T) * (self.a1 * (1 - self.a1))
        
        # Gradientes
        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0)
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Actualización con momento
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, y, epochs=1000, lr=0.01, batch_size=32, val_data=None):
        for epoch in range(epochs):
            # Mini-batch
            indices = np.random.permutation(X.shape[0])
            for i in range(0, X.shape[0], batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                self.forward(X_batch)
                self.backward(X_batch, y_batch, lr)
            
            # Cálculo de pérdida
            probs = self.forward(X)
            loss = self._cross_entropy_loss(y, probs)
            self.loss_history.append(loss)
            
            # Validación
            if val_data and epoch % 10 == 0:
                X_val, y_val = val_data
                val_probs = self.forward(X_val)
                val_loss = self._cross_entropy_loss(y_val, val_probs)
                print(f"Epoch {epoch}: Train Loss={loss:.4f}, Val Loss={val_loss:.4f}")

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

# ==================== [4. Carga de Dataset y Preprocesamiento] ====================
def load_dataset(dataset_path='genres'):
    genres = [g for g in os.listdir(dataset_path) if not g.startswith('.')]  # Ignorar archivos ocultos
    X = []
    y = []
    
    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        for file in os.listdir(genre_path):
            if not file.lower().endswith(('.wav', '.mp3', '.ogg')):  # Filtrar formatos
                continue
                
            file_path = os.path.join(genre_path, file)
            try:
                features = extract_features(file_path)
                X.append(features)
                y.append(genre)
            except Exception as e:
                print(f"Error procesando {file}: {str(e)}")
                continue
    
    return np.array(X), np.array(y)

# ==================== [5. Interfaz Gráfica Mejorada] ====================
class GenreClassifierApp:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.root = Tk()
        self.root.title("Clasificador de Géneros Musicales v2.0")
        
        self.create_widgets()
        
    def create_widgets(self):
        Label(self.root, text="Seleccione un archivo de audio", font=('Arial', 12)).pack(pady=10)
        Button(self.root, text="Cargar Archivo", command=self.predict_genre, 
              bg='#4CAF50', fg='white').pack(pady=15)
        
    def predict_genre(self):
        filetypes = [
            ('Audio Files', '*.wav *.mp3 *.ogg'),
            ('Todos los archivos', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        if not file_path:
            return
            
        try:
            # Extraer y preprocesar características
            raw_features = extract_features(file_path)
            features = self.preprocessor.transform(raw_features.reshape(1, -1))
            
            # Predecir
            prediction = self.model.predict(features)
            genre = self.preprocessor.inverse_transform_labels(prediction)[0]
            
            # Mostrar resultado
            messagebox.showinfo("Resultado", 
                f"Género predicho:\n{genre.upper()}\n\nArchivo: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error de procesamiento:\n{str(e)}")

    def run(self):
        self.root.mainloop()

# ==================== [6. Ejecución Principal] ====================
if __name__ == "__main__":
    # Cargar y preprocesar datos
    X, y = load_dataset()
    preprocessor = AudioPreprocessor()
    preprocessor.fit(X, y)
    X_scaled, y_encoded = preprocessor.transform(X, y)
    
    # Dividir datos (60-20-20)
    indices = np.random.permutation(len(X))
    split_train = int(0.6 * len(X))
    split_val = int(0.8 * len(X))
    
    X_train, y_train = X_scaled[indices[:split_train]], y_encoded[indices[:split_train]]
    X_val, y_val = X_scaled[indices[split_train:split_val]], y_encoded[indices[split_train:split_val]]
    X_test, y_test = X_scaled[indices[split_val:]], y_encoded[indices[split_val:]]
    
    # Entrenar modelo
    mlp = MLP(input_size=X_train.shape[1], 
             hidden_size=64,  # Aumentar capacidad
             output_size=len(preprocessor.label_encoder))
    
    mlp.train(X_train, y_train, epochs=500, lr=0.001, batch_size=64,
             val_data=(X_val, y_val))
    
    # Evaluación final
    test_pred = mlp.predict(X_test)
    accuracy = np.mean(test_pred == y_test)
    print(f"\nPrecisión final en test: {accuracy:.2%}")
    
    # Iniciar interfaz
    app = GenreClassifierApp(mlp, preprocessor)
    app.run()