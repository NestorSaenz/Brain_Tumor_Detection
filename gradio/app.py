import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Carga el modelo
model = load_model("best_resnet_model.h5")

def predecir_tumor(imagen):
    img = imagen.convert("RGB").resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_batch = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_batch)[0][0]
    etiqueta = "Tumor" if pred > 0.5 else "Sano"
    probabilidad = pred if pred > 0.5 else 1 - pred
    diferencia = abs(pred - (1 - pred))
    resultado = f"🧠 Resultado: <b>{etiqueta}</b> <br>Probabilidad: <b>{probabilidad*100:.2f}%</b>"

    # Si la diferencia es menor o igual a 0.10 (10%), mostrar advertencia
    if diferencia <= 0.10:
        resultado += (
            "<br><div style='background:#fff3cd; color:#856404; border:1px solid #ffeeba; "
            "border-radius:6px; padding:10px; margin-top:10px;'>"
            "⚠️ <b>Advertencia:</b> El modelo no está seguro de la predicción. "
            "Consulta siempre a un especialista médico.</div>"
        )
    return { "Tumor": float(pred), "Sano": float(1 - pred) }, resultado

# Título y descripción en español
titulo = "🧠 Detección de Tumores Cerebrales por IA"
descripcion = """
<div style="text-align: justify;">
<b>Sube una imagen de resonancia magnética cerebral (MRI)</b> y nuestro modelo de inteligencia artificial te indicará si detecta la presencia de un tumor.<br><br>
<b>Instrucciones:</b>
<ul>
  <li>Haz clic en "Examinar archivos" o arrastra una imagen MRI en formato JPG o PNG.</li>
  <li>Presiona <b>Submit</b> para obtener el resultado.</li>
</ul>
</div>
"""

ejemplos = None  # Puedes agregar rutas a imágenes de ejemplo si lo deseas

css_personalizado = """
footer {visibility: hidden;}
.gradio-container {background: #f6f8fa;}
h1, .title {color: #4F8BF9 !important;}
"""

iface = gr.Interface(
    fn=predecir_tumor,
    inputs=gr.Image(type="pil", label="Imagen MRI"),
    outputs=[
        gr.Label(num_top_classes=2, label="Probabilidades"),
        gr.HTML(label="Resultado")
    ],
    title=titulo,
    description=descripcion,
    allow_flagging="never",
    examples=ejemplos,
    theme=gr.themes.Soft(primary_hue="blue"),
    css=css_personalizado
)

if __name__ == "__main__":
    iface.launch()