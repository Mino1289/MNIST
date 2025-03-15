import torch
import torch.nn.functional as F
import cv2
import collections
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


from models import *

models = {
    "MNISTClassifierCNN": {"path": "models/mnist_modelCNN.pth", "class": MNISTClassifierCNN},
    "MNISTClassifierMLP": {"path": "models/mnist_modelMLP.pth", "class": MNISTClassifierMLP},
    "MNISTClassifierResNet18": {"path": "models/resnet18_mnist_model.pth", "class": MNISTClassifierResNet18},
}


def load_model(model_path, model_class):
    input_size = 1 * 28 * 28
    n_classes = 10

    model = model_class(input_size, n_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=device)

    # Créer un nouveau state_dict sans le préfixe '_orig_mod.'
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('_orig_mod.', '')  # Retirer le préfixe '_orig_mod.'
        new_state_dict[name] = v

    # Charger le nouveau state_dict dans le modèle
    model.load_state_dict(new_state_dict)
    model.to(device=device)

    model.eval()

    return model

def predict(model_name, input):
    model_info = models[model_name]
    model_path = model_info["path"]
    model_class = model_info["class"]

    model = load_model(model_path, model_class)

    if isinstance(input, dict) and "composite" in input:
        image = np.array(input["composite"], dtype=np.uint8)
    elif isinstance(input, Image.Image):
        image = np.array(input, dtype=np.uint8)
    else:
        return None, None

    if image.shape[2] == 4:
        image = image[:, :, 3]
        image = np.expand_dims(image, axis=2)

    image_modif = cv2.resize(image, (28, 28), interpolation=cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_modif = image_modif.astype(np.float32) / 255

    image_tensor = torch.tensor(image_modif).unsqueeze(0).unsqueeze(0)

    output = model(image_tensor)
    
    # Appliquer la fonction softmax pour obtenir les probabilités
    probabilities = F.softmax(output, dim=1).squeeze().detach().numpy()
    # print(probabilities)
    fig, ax = plt.subplots()
    # Générer un graphique en barre avec les probabilités de chaque valeur
    ax.bar(range(10), probabilities)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Classes')
    ax.set_ylabel('Probabilités')
    ax.set_title('Probabilités des classes')
    return image, fig

with gr.Blocks() as demo:
    model_dropdown = gr.Dropdown(choices=list(models.keys()), label="Select Model")
    with gr.Row():
        input_canvas = gr.Sketchpad(label="Dessiner ici", canvas_size=(560, 560), layers=False)
        output_image = gr.Image(label="Image convertie")
        output_chart = gr.Plot(label="Probabilités des classes")

    btn = gr.Button("Prédire")
    btn.click(predict, inputs=[model_dropdown, input_canvas], outputs=[output_image, output_chart])

# Lancer l'interface Gradio
demo.launch()
