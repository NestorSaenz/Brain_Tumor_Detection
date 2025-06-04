from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from generators import DataGenerator
from Brain_Loader_Preprocessing import BrainMRIDataLoader, Brain_preprocessing
import pandas as pd

# Carga y prepara los datos igual que en tu pipeline
load_data = BrainMRIDataLoader('images', ['Healthy', 'Tumor'])
df = load_data.load_data()
df = Brain_preprocessing(df)
data = df.category_encoder() 
data['category_encoded'] = data['label'].astype('category').cat.codes

# Divide en train, val y test igual que en el pipeline
train_df, val_df, test_df = df.train_val_test_split(0.2, 42)
test_df['category_encoded'] = test_df['category_encoded'].astype(str)

# Configura el generador de datos
data_gen = DataGenerator(batch_size=32, img_size=(224, 224))
test_datagen = data_gen.get_test_datagen()
test_gen = data_gen.get_generator(test_df, test_datagen, shuffle=False)

# Carga el modelo entrenado
model = load_model('best_resnet_model.h5')

# Evaluar en el conjunto de prueba
results = model.evaluate(test_gen, verbose=1)
metric_names = model.metrics_names
print("\nMétricas de evaluación en el conjunto de prueba:")
for name, value in zip(metric_names, results):
    print(f"{name}: {value:.4f}")

# Matriz de confusión y reporte de clasificación
y_true = test_df['category_encoded'].astype(int).values
y_pred_probs = model.predict(test_gen)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

cm = confusion_matrix(y_true, y_pred)
print("Matriz de confusión:\n", cm)
print(classification_report(y_true, y_pred, target_names=['Healthy', 'Tumor']))