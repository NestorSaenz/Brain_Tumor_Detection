import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Recall, Precision


class ResNet50FineTuner:
    def __init__(
        self,
        input_shape=(224, 224, 3),
        dropout_rate=0.5,
        l2_reg=0.01,
        initial_lr=1e-3,
        fine_tune_lr=1e-5,
        initial_epochs=10,
        fine_tune_epochs=10,
        model_path='best_resnet_model.h5'
    ):
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.initial_lr = initial_lr
        self.fine_tune_lr = fine_tune_lr
        self.initial_epochs = initial_epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.model_path = model_path
        self.model = self.build_model()
        self.callbacks = self.get_callbacks()

    def build_model(self):
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )
        base_model.trainable = False
        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(
            1,
            activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )(x)
        model = Model(inputs, outputs)
        return model

    def compile_model(self, learning_rate):
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
            'accuracy',
            AUC(name='auc'),
            Recall(name='recall'),
            Precision(name='precision')
        ]
        )

    def get_callbacks(self):
        return [
            callbacks.EarlyStopping(
                monitor='val_auc',
                patience=5,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=3,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                self.model_path,
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]

    def train(self, train_gen, val_gen):
        # Fase 1: Entrenar solo las capas superiores
        print("\nEntrenando capas nuevas...")
        self.compile_model(self.initial_lr)
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.initial_epochs,
            callbacks=self.callbacks,
            verbose=1
        )

        # Fase 2: Fine-tuning de todo el modelo
        print("\nFine-tuning de todo el modelo...")
        base_model = self.model.layers[1]
        base_model.trainable = True
        self.compile_model(self.fine_tune_lr)
        total_epochs = self.initial_epochs + self.fine_tune_epochs
        history_fine = self.model.fit(
            train_gen,
            validation_data=val_gen,
            initial_epoch=history.epoch[-1] + 1,
            epochs=total_epochs,
            callbacks=self.callbacks,
            verbose=1
        )
        return history, history_fine