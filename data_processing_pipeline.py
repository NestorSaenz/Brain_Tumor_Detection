from Brain_Loader_Preprocessing import *
from generators import DataGenerator
from finetune_resnet50 import ResNet50FineTuner
from sklearn.metrics import confusion_matrix, classification_report


load_data = BrainMRIDataLoader(base_path='images', categories=['Healthy', 'Tumor'])
data_encoded = Brain_preprocessing(df=load_data.load_data())

load_data.show_summary()
df = data_encoded.category_encoder()
data_encoded.train_val_test_split(test_size=0.2, random_state=42)

train_df, val_df, test_df = data_encoded.train_val_test_split(test_size=0.2, random_state=42)
train_df = data_encoded.prepare_for_model(train_df)
val_df = data_encoded.prepare_for_model(val_df)
test_df = data_encoded.prepare_for_model(test_df)

train_df['category_encoded'] = train_df['category_encoded'].astype(str)
val_df['category_encoded'] = val_df['category_encoded'].astype(str)
test_df['category_encoded'] = test_df['category_encoded'].astype(str)


data_gen = DataGenerator(batch_size=16, img_size=(224, 224))
train_datagen = data_gen.get_train_datagen()
test_datagen = data_gen.get_test_datagen()

train_gen = data_gen.get_generator(train_df, train_datagen, shuffle=True)
val_gen = data_gen.get_generator(val_df, test_datagen, shuffle=True)
test_gen = data_gen.get_generator(test_df, test_datagen, shuffle=False)

finetuner = ResNet50FineTuner(
    input_shape=(224, 224, 3),
    dropout_rate=0.5,
    l2_reg=0.01,
    initial_lr=1e-3,
    fine_tune_lr=1e-5,
    initial_epochs=10,
    fine_tune_epochs=10,
    model_path='best_resnet_model.h5'
)

history, history_fine = finetuner.train(train_gen, val_gen)




# Obtener predicciones sobre el conjunto de prueba
y_true = test_df['category_encoded'].astype(int).values
y_pred_probs = finetuner.model.predict(test_gen)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
print("Matriz de confusión:\n", cm)

# Reporte de clasificación (incluye recall, precisión, f1-score)
print(classification_report(y_true, y_pred, target_names=['Healthy', 'Tumor']))