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
import tarfile
import pickle

import warnings

# Suppress specific UserWarning from torchvision
warnings.filterwarnings("ignore",
                        message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future.*")
warnings.filterwarnings("ignore",
                        message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.*")

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


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def ada_fgsm_attack(model, image, label, epsilon, T, suppress_term_q):
    x_prime = image.clone().detach()  # detach to ensure leaf tensor
    x_prime.requires_grad = True

    S_t = epsilon / T
    g_bar = torch.zeros_like(x_prime)  # Average of gradients (initialized to zero)
    G = torch.zeros_like(x_prime)  # Total step size (initialized to zero)
    S = 0  # Current step size

    for t in range(1, T + 1):

        # Forward pass through the model to get the logits for the current adversarial image (x'_t-1).
        output = model(x_prime)

        # Clear previous gradients before the backward pass.
        model.zero_grad()

        # Calculate the loss value (J) using the cross-entropy loss function, between the model's output and the true
        # label (y_T).
        loss = torch.nn.CrossEntropyLoss()(output, label)

        # Backward pass through the model to compute the gradient of the loss with respect to the input image (x'_t-1),
        # which is mathematically denoted as ∇xJ(x'_t-1, y_T).
        loss.backward()

        # Retrieve the gradient data from the adversarial image, which is used to calculate the perturbation.
        gradient_t = x_prime.grad.data

        # Calculate g_bar and S
        if t == 1:
            S_t = (epsilon / T) * suppress_term_q  # S_t is actually S_1 here in this if statement. The initial step
            g_bar = gradient_t
        else:
            # Element-wise operation for gradients and step sizes
            rho = sigmoid(gradient_t / g_bar + 1e-10) * 2
            S_t = ((epsilon - S) * rho) / (rho + T - t - 1)

        # Update G and g_bar for the next iteration
        G = G + gradient_t
        g_bar = G / t

        # Update S for the next iteration
        S = S + S_t

        # Update perturbed image using the normalized gradient by its p-norm
        # Assuming p-norm is required, for p=2 or p=∞ (L2 or L-infinity norm), for example:
        p = float('inf')  # or p = 2 for L-infinity norm
        x_prime = x_prime + S_t * (gradient_t / gradient_t.norm(p=p, dim=(1, 2, 3), keepdim=True))

        # Clamp the perturbed image to ensure pixel values are within [0, 1]
        x_prime = torch.clamp(x_prime, 0, 1)

        # Detach perturbed_image for the next iteration to ensure leaf tensor
        x_prime = x_prime.detach()
        x_prime.requires_grad = True

        if t == 8:
            break

    return x_prime

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


def evaluate_model(model_name, image_path, epsilon=0.01, alpha=0.003, num_iter=10, pdf=None):
    '''Evaluates the model on the original, FGSM adversarial, and iFGSM adversarial images, including parameters in the PDF.'''
    image_name = os.path.basename(image_path)  # Get the image name from the image path
    print(f"\nEvaluating Model: {model_name} on Image: {image_name}")

    # Load the model
    model = getattr(models, model_name)(pretrained=True)
    model.eval()

    # Load and prepare the image
    image = load_image(image_path)

    # Original image predictions
    original_class_names, original_confidences = get_predictions(model, image)
    print(
        f"Original Image - Model: {model_name}, Image: {image_name}, Top Prediction: {original_class_names[0]}, Confidence: {original_confidences[0]}")

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
    print(
        f"FGSM Adversarial Image - Model: {model_name}, Image: {image_name}, ε={epsilon}, Top Prediction: {fgsm_adversarial_class_names[0]}, Confidence: {fgsm_adversarial_confidences[0]}")

    # iFGSM attack
    ifgsm_perturbed_image = ifgsm_attack(model, image, epsilon, data_grad, alpha, num_iter)

    # iFGSM adversarial image predictions
    ifgsm_adversarial_class_names, ifgsm_adversarial_confidences = get_predictions(model, ifgsm_perturbed_image)
    print(
        f"iFGSM Adversarial Image - Model: {model_name}, Image: {image_name}, ε={epsilon}, α={alpha}, Top Prediction: {ifgsm_adversarial_class_names[0]}, Confidence: {ifgsm_adversarial_confidences[0]}")

    # Visualization and PDF saving logic
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    # Original image
    image_np = image.squeeze().detach().numpy().transpose(1, 2, 0)
    axs[0].imshow(image_np)
    axs[0].set_title(f"Original Image: {original_class_names[0]}")
    axs[0].set_xlabel(
        "\n".join([f"{name}: {conf:.2f}" for name, conf in zip(original_class_names, original_confidences)]),
        fontsize=10)

    # FGSM adversarial image
    fgsm_perturbed_image_np = fgsm_perturbed_image.squeeze().detach().numpy().transpose(1, 2, 0)
    axs[1].imshow(fgsm_perturbed_image_np)
    axs[1].set_title(f"FGSM Adversarial Image (ε={epsilon}): {fgsm_adversarial_class_names[0]}")
    axs[1].set_xlabel("\n".join(
        [f"{name}: {conf:.2f}" for name, conf in zip(fgsm_adversarial_class_names, fgsm_adversarial_confidences)]),
        fontsize=10)

    # iFGSM adversarial image
    ifgsm_perturbed_image_np = ifgsm_perturbed_image.squeeze().detach().numpy().transpose(1, 2, 0)
    axs[2].imshow(ifgsm_perturbed_image_np)
    axs[2].set_title(f"iFGSM Adversarial Image (ε={epsilon}, α={alpha}): {ifgsm_adversarial_class_names[0]}")
    axs[2].set_xlabel("\n".join(
        [f"{name}: {conf:.2f}" for name, conf in zip(ifgsm_adversarial_class_names, ifgsm_adversarial_confidences)]),
        fontsize=10)

    plt.tight_layout()

    # Save the current figure to the PDF file if PdfPages object is provided
    if pdf is not None:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def evaluate_model_ada_fgsm(model_name, image_path, epsilon=0.01, num_iter=10, suppress_term_q=1.0, pdf=None):
    image_name = os.path.basename(image_path)
    print(f"\nEvaluating Model with Ada-FGSM: {model_name} on Image: {image_name}")

    # Load the model
    model = getattr(models, model_name)(pretrained=True)
    model.eval()

    # Load and prepare the image
    image = load_image(image_path)

    # Original image predictions
    original_class_names, original_confidences = get_predictions(model, image)
    print(
        f"Original Image - Model: {model_name}, Image: {image_name}, Top Prediction: {original_class_names[0]}, Confidence: {original_confidences[0]}")

    # Ada-FGSM attack
    image.requires_grad = True
    output = model(image)
    init_pred = output.max(1)[1]  # Initial prediction
    ada_fgsm_perturbed_image = ada_fgsm_attack(model, image, init_pred, epsilon, num_iter, suppress_term_q)

    # Ada-FGSM adversarial image predictions
    ada_fgsm_adversarial_class_names, ada_fgsm_adversarial_confidences = get_predictions(model,
                                                                                         ada_fgsm_perturbed_image)
    print(
        f"Ada-FGSM Adversarial Image - Model: {model_name}, Image: {image_name}, ε={epsilon}, Top Prediction: {ada_fgsm_adversarial_class_names[0]}, Confidence: {ada_fgsm_adversarial_confidences[0]}")

    # Visualization and PDF saving logic
    if pdf is not None:
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        # Original image
        image_np = image.squeeze().detach().numpy().transpose(1, 2, 0)
        axs[0].imshow(image_np)
        axs[0].set_title(f"Original Image: {original_class_names[0]}")
        axs[0].set_xlabel(
            "\n".join([f"{name}: {conf:.2f}" for name, conf in zip(original_class_names, original_confidences)]),
            fontsize=10)

        # Ada-FGSM adversarial image
        ada_fgsm_perturbed_image_np = ada_fgsm_perturbed_image.squeeze().detach().numpy().transpose(1, 2, 0)
        axs[1].imshow(ada_fgsm_perturbed_image_np)
        axs[1].set_title(f"Ada-FGSM Adversarial Image (ε={epsilon}): {ada_fgsm_adversarial_class_names[0]}")
        axs[1].set_xlabel("\n".join([f"{name}: {conf:.2f}" for name, conf in
                                     zip(ada_fgsm_adversarial_class_names, ada_fgsm_adversarial_confidences)]),
                          fontsize=10)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

def run_Ada_FGSM_Attacks():
    """ADA FGSM Attacks"""
    image_path = '_images/bananas.JPG'  # Update this path
    with PdfPages('Ada_FGSM_output/Ada_FGSM_attacks_results_bananas.pdf') as pdf:
        for model_name in ['resnet18', 'densenet121', 'googlenet']:
            evaluate_model_ada_fgsm(model_name, image_path, epsilon=0.2, num_iter=10, suppress_term_q=0.2,
                                    pdf=pdf)

    image_path = '_images/bellpepper.JPG'  # Update this path
    with PdfPages('Ada_FGSM_output/Ada_FGSM_attacks_results_bellpepper.pdf') as pdf:
        for model_name in ['resnet18', 'densenet121', 'googlenet']:
            evaluate_model_ada_fgsm(model_name, image_path, epsilon=0.2, num_iter=10, suppress_term_q=0.2,
                                    pdf=pdf)

    image_path = '_images/pomegranate.jpg'  # Update this path
    with PdfPages('Ada_FGSM_output/Ada_FGSM_attacks_results_pomegranate.pdf') as pdf:
        for model_name in ['resnet18', 'densenet121', 'googlenet']:
            evaluate_model_ada_fgsm(model_name, image_path, epsilon=0.2, num_iter=10, suppress_term_q=0.2,
                                    pdf=pdf)

    image_path = '_images/sphaghettisquash.JPG'  # Update this path
    with PdfPages('Ada_FGSM_output/Ada_FGSM_attacks_results_sphaghettisquash.pdf') as pdf:
        for model_name in ['resnet18', 'densenet121', 'googlenet']:
            evaluate_model_ada_fgsm(model_name, image_path, epsilon=0.2, num_iter=10, suppress_term_q=0.2,
                                    pdf=pdf)


def run_FGSM_and_iFGSM_Attacks():

    """
    FGSM AND iFGSM Attacks
    """

    image_path = '_images/bananas.JPG'  # Update this path
    with PdfPages('FGSM_output/FGSM_attacks_results_bananas.pdf') as pdf:
        for model_name in ['resnet18', 'densenet121', 'googlenet']:
            evaluate_model(model_name, image_path, epsilon=0.01, pdf=pdf)

    image_path = '_images/bellpepper.JPG'  # Update this path
    with PdfPages('FGSM_output/FGSM_attacks_results_bellpepper.pdf') as pdf:
        for model_name in ['resnet18', 'densenet121', 'googlenet']:
            evaluate_model(model_name, image_path, epsilon=0.01, pdf=pdf)

    image_path = '_images/pomegranate.jpg'  # Update this path
    with PdfPages('FGSM_output/FGSM_attacks_results_pomegranate.pdf') as pdf:
        for model_name in ['resnet18', 'densenet121', 'googlenet']:
            evaluate_model(model_name, image_path, epsilon=0.01, pdf=pdf)

    image_path = '_images/sphaghettisquash.JPG'  # Update this path
    with PdfPages('FGSM_output/FGSM_attacks_results_sphaghettisquash.pdf') as pdf:
        for model_name in ['resnet18', 'densenet121', 'googlenet']:
            evaluate_model(model_name, image_path, epsilon=0.01, pdf=pdf)


def main():
    return


if __name__ == "__main__":
    main()
