import matplotlib
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import json
from torchvision import models
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
import os

matplotlib.use('TkAgg')  # Set the backend to 'Agg'

# Load the ImageNet class names
with open('imagenet_class_index.json') as f:
    idx_to_labels = json.load(f)


def load_image(image_path):
    '''Loads an image and applies necessary transformations.'''
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def fgsm_attack(image, epsilon, data_grad):
    '''Generates the adversarial image by adding the perturbation to the original image.'''
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Keep pixel values in [0, 1]
    return perturbed_image

def ifgsm_attack(model, image, epsilon, data_grad, alpha, num_iter):
    '''Generates the adversarial image using iterative FGSM.'''
    perturbed_image = image.clone().detach()  # Detach to ensure it's a leaf tensor
    perturbed_image.requires_grad = True

    for _ in range(num_iter):
        output = model(perturbed_image)
        loss = torch.nn.CrossEntropyLoss()(output, output.max(1)[1])
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_image.grad.data

        # Apply perturbation
        sign_data_grad = data_grad.sign()
        perturbed_image = perturbed_image + alpha * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Keep pixel values in [0, 1]
        perturbed_image = perturbed_image.detach()  # Detach to ensure it's a leaf tensor for next iteration
        perturbed_image.requires_grad = True

        # Optional: Enforce the total perturbation does not exceed epsilon
        # This step is commented out as it's not always needed but can be included for stricter control
        # total_perturbation = torch.clamp(perturbed_image - image, -epsilon, epsilon)
        # perturbed_image = image + total_perturbation
        # perturbed_image = perturbed_image.detach()  # Ensure it's a leaf tensor
        # perturbed_image.requires_grad = True

    return perturbed_image




def get_predictions(model, image, topk=3):
    '''Get model predictions and return the top k predictions along with confidence scores.'''
    outputs = model(image)
    # Convert to probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    top_probs, top_idxs = torch.topk(probabilities, topk)
    top_probs = top_probs.squeeze().tolist()  # Convert to list
    top_idxs = top_idxs.squeeze().tolist()  # Convert to list

    class_names = [idx_to_labels[str(idx)][1] for idx in top_idxs]
    return class_names, top_probs


def evaluate_model(model_name, image_path, epsilon=0.01, alpha=0.001, num_iter=10, pdf=None):
    '''Evaluates the model on the original, FGSM adversarial, and iFGSM adversarial images, including parameters in the PDF.'''
    # Load the model
    model = getattr(models, model_name)(pretrained=True)
    model.eval()

    # Load and prepare the image
    image = load_image(image_path)

    # Original image predictions
    original_class_names, original_confidences = get_predictions(model, image)

    # FGSM attack
    image.requires_grad = True
    output = model(image)
    init_pred = output.max(1)[1]  # Initial prediction
    loss = torch.nn.CrossEntropyLoss()(output, init_pred)
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    fgsm_perturbed_image = fgsm_attack(image, epsilon, data_grad)

    # FGSM adversarial image predictions
    fgsm_adversarial_class_names, fgsm_adversarial_confidences = get_predictions(model, fgsm_perturbed_image)

    # iFGSM attack
    ifgsm_perturbed_image = ifgsm_attack(model, image, epsilon, data_grad, alpha, num_iter)

    # iFGSM adversarial image predictions
    ifgsm_adversarial_class_names, ifgsm_adversarial_confidences = get_predictions(model, ifgsm_perturbed_image)

    # Visualization and PDF saving logic
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    # Original image
    image_np = image.squeeze().detach().numpy().transpose(1, 2, 0)
    axs[0].imshow(image_np)
    axs[0].set_title(f"Original Image: {original_class_names[0]}")
    axs[0].set_xlabel("\n".join([f"{name}: {conf:.2f}" for name, conf in zip(original_class_names, original_confidences)]), fontsize=10)

    # FGSM adversarial image
    fgsm_perturbed_image_np = fgsm_perturbed_image.squeeze().detach().numpy().transpose(1, 2, 0)
    axs[1].imshow(fgsm_perturbed_image_np)
    axs[1].set_title(f"FGSM Adversarial Image (ε={epsilon}): {fgsm_adversarial_class_names[0]}")
    axs[1].set_xlabel("\n".join([f"{name}: {conf:.2f}" for name, conf in zip(fgsm_adversarial_class_names, fgsm_adversarial_confidences)]), fontsize=10)

    # iFGSM adversarial image
    ifgsm_perturbed_image_np = ifgsm_perturbed_image.squeeze().detach().numpy().transpose(1, 2, 0)
    axs[2].imshow(ifgsm_perturbed_image_np)
    axs[2].set_title(f"iFGSM Adversarial Image (ε={epsilon}, α={alpha}): {ifgsm_adversarial_class_names[0]}")
    axs[2].set_xlabel("\n".join([f"{name}: {conf:.2f}" for name, conf in zip(ifgsm_adversarial_class_names, ifgsm_adversarial_confidences)]), fontsize=10)

    plt.tight_layout()

    # Save the current figure to the PDF file if PdfPages object is provided
    if pdf is not None:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)





def main():
    image_path = '_images/sphaghettisquash.JPG'  # Update this path
    with PdfPages('FGSM_output/FGSM_attacks_results_sphaghettisquash.pdf') as pdf:
        for model_name in ['resnet18', 'densenet121', 'googlenet']:
            evaluate_model(model_name, image_path, epsilon=0.01, pdf=pdf)


if __name__ == "__main__":
    main()
