import numpy as np
import matplotlib
import os

matplotlib.use('TkAgg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import random
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_input_resnet, \
    decode_predictions as decode_predictions_resnet
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_input_inception, \
    decode_predictions as decode_predictions_inception
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input as preprocess_input_densenet, \
    decode_predictions as decode_predictions_densenet
from tensorflow.keras.preprocessing import image
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages


def create_random_patch(patch_size):
    """Create a simple adversarial patch"""
    patch = np.uint8(np.random.rand(patch_size, patch_size, 3) * 255)  # Random noise patch
    return Image.fromarray(patch, 'RGB')


def classify_image(model, img, preprocess_input, decode_predictions, input_size):
    """Classify the image using the model and print the top 3 predictions."""
    img = img.resize(input_size)  # Resize the image to match the model's expected input
    img_array = image.img_to_array(img)  # Convert the image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image
    preds = model.predict(img_array)  # Predict the class of the image
    return decode_predictions(preds, top=3)[0]  # Return the top 3 predicted classes


def apply_patch_to_image(original_image, patch, position=(0, 0)):
    """
    Apply the adversarial patch to the original image.

    Parameters:
    - original_image: PIL.Image object.
    - patch: A numpy array or a PIL.Image object representing the adversarial patch.
    - position: A tuple (x, y) representing where to place the patch on the original image.

    Returns:
    - A new PIL.Image object with the patch applied.
    """
    if isinstance(patch, np.ndarray):
        patch = Image.fromarray(patch)
    patched_image = original_image.copy()
    patched_image.paste(patch, position)
    return patched_image


# Method for randomly choosing the position for which the patch will be applied
def get_random_position(original_image, patch_size):
    """Generate a random position for the patch within the bounds of the original image."""
    max_x = original_image.width - patch_size
    max_y = original_image.height - patch_size
    if max_x < 0 or max_y < 0:
        raise ValueError("Patch size is too large for the given image.")
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    return (x, y)


def display_images(original_image, patched_image):
    """
    Display the original and patched images side by side.

    Parameters:
    - original_image: PIL.Image object of the original image.
    - patched_image: PIL.Image object of the patched image.
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(patched_image)
    plt.title('Patched Image')
    plt.axis('off')

    plt.show()


def display_images_and_predictions(original_image, patched_image, original_preds, patched_preds, model_name, pdf):
    """
    Display the original and patched images side by side with predictions,
    and save the plot to a PDF file, now including the model name in the title.

    Parameters:
    - original_image: PIL.Image object of the original image.
    - patched_image: PIL.Image object of the patched image.
    - original_preds: Predictions for the original image.
    - patched_preds: Predictions for the patched image.
    - model_name: Name of the model used for predictions.
    - pdf: PdfPages object to save the plots.
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title(
        f'{model_name} - Original Image\n' + '\n'.join([f'{label}: {prob:.4f}' for _, label, prob in original_preds]))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(patched_image)
    plt.title(
        f'{model_name} - Patched Image\n' + '\n'.join([f'{label}: {prob:.4f}' for _, label, prob in patched_preds]))
    plt.axis('off')

    plt.tight_layout()
    pdf.savefig()
    plt.close()


def display_images_and_predictions_compact(original_image, patched_images, predictions, pdf):
    """
    Display images and predictions in a more compact layout.
    Each row will represent a model, showing the original image once and patched images next to it.

    Parameters:
    - original_image: PIL.Image object of the original image.
    - patched_images: List of tuples, each containing a PIL.Image object of the patched image and its model name.
    - predictions: Dictionary containing predictions for each model and patch size.
    - pdf: PdfPages object to save the plots.
    """
    num_models = len(patched_images)
    fig, axs = plt.subplots(num_models, len(patched_images[0][1]) + 1, figsize=(10, num_models * 2))

    for i, (model_name, patches) in enumerate(patched_images):
        axs[i, 0].imshow(original_image)
        axs[i, 0].set_title(f'{model_name} Original')
        axs[i, 0].axis('off')

        for j, (patched_image, patch_size) in enumerate(patches, start=1):
            axs[i, j].imshow(patched_image)
            title = f'Patch {patch_size}\n'
            title += '\n'.join([f'{label}: {prob:.4f}' for _, label, prob in predictions[model_name][patch_size]])
            axs[i, j].set_title(title, fontsize=8)
            axs[i, j].axis('off')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def main_not_compact():
    models = [
        (ResNet50(weights='imagenet'), preprocess_input_resnet, decode_predictions_resnet, "ResNet50", (224, 224)),
        (InceptionV3(weights='imagenet'), preprocess_input_inception, decode_predictions_inception, "InceptionV3",
         (299, 299)),
        (DenseNet201(weights='imagenet'), preprocess_input_densenet, decode_predictions_densenet, "DenseNet201",
         (224, 224))
    ]

    original_image_path = '_images/sphaghettisquash.JPG'
    original_image = Image.open(original_image_path)

    with PdfPages('patch_output/output_sphaghettisquash.pdf') as pdf:
        for model, preprocess_input, decode_predictions, model_name, input_size in models:
            # Systematic Testing with 10 Patch Sizes Incremented by 50
            for i in range(1, 11):
                patch_size = i * 50
                position = get_random_position(original_image, patch_size)
                patch = create_random_patch(patch_size)
                patched_image = apply_patch_to_image(original_image, patch, position)

                # Adjust classify_image function calls to use specific preprocesssinput and decode_predictions
                original_preds = classify_image(model, original_image, preprocess_input, decode_predictions, input_size)
                patched_preds = classify_image(model, patched_image, preprocess_input, decode_predictions, input_size)

                print(f"\nModel: {model_name}, Patch Size: {patch_size}, Random Position: {position}")
                print("Original Image Predictions:", original_preds)
                print("Patched Image Predictions:", patched_preds)

                # Save images and predictions to PDF
                display_images_and_predictions(original_image, patched_image, original_preds, patched_preds, model_name,
                                               pdf)


def main_compact():
    models = [
        (ResNet50(weights='imagenet'), preprocess_input_resnet, decode_predictions_resnet, "ResNet50", (224, 224)),
        (InceptionV3(weights='imagenet'), preprocess_input_inception, decode_predictions_inception, "InceptionV3",
         (299, 299)),
        (DenseNet201(weights='imagenet'), preprocess_input_densenet, decode_predictions_densenet, "DenseNet201",
         (224, 224))
    ]

    image_dir = '_images'
    output_dir = 'patch_output'
    patch_sizes = [300, 350, 400, 450, 500]  # Example patch sizes

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all files in the image directory
    for image_filename in os.listdir(image_dir):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Check for image files
            original_image_path = os.path.join(image_dir, image_filename)
            original_image = Image.open(original_image_path)
            patched_images_info = []
            predictions = {}

            # Create a PDF file for each image
            pdf_filename = f'output_{os.path.splitext(image_filename)[0]}_compact.pdf'
            pdf_path = os.path.join(output_dir, pdf_filename)

            with PdfPages(pdf_path) as pdf:
                for model, preprocess_input, decode_predictions, model_name, input_size in models:
                    patched_images = []
                    predictions[model_name] = {}
                    for patch_size in patch_sizes:
                        position = get_random_position(original_image, patch_size)
                        patch = create_random_patch(patch_size)
                        patched_image = apply_patch_to_image(original_image, patch, position)

                        # Classify original and patched images
                        original_preds = classify_image(model, original_image, preprocess_input, decode_predictions,
                                                        input_size)
                        patched_preds = classify_image(model, patched_image, preprocess_input, decode_predictions,
                                                       input_size)

                        patched_images.append((patched_image, patch_size))
                        predictions[model_name][patch_size] = patched_preds

                    patched_images_info.append((model_name, patched_images))

                # After collecting all patched images and predictions, display and save them compactly
                display_images_and_predictions_compact(original_image, patched_images_info, predictions, pdf)

            print(f"Results for {image_filename} saved to {pdf_filename}")


def main():
    models = [
        (ResNet50(weights='imagenet'), preprocess_input_resnet, decode_predictions_resnet, "ResNet50", (224, 224)),
        (InceptionV3(weights='imagenet'), preprocess_input_inception, decode_predictions_inception, "InceptionV3",
         (299, 299)),
        (DenseNet201(weights='imagenet'), preprocess_input_densenet, decode_predictions_densenet, "DenseNet201",
         (224, 224))
    ]

    image_dir = '_images'
    output_dir = 'patch_output'

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all files in the image directory
    for image_filename in os.listdir(image_dir):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Check for image files
            original_image_path = os.path.join(image_dir, image_filename)
            original_image = Image.open(original_image_path)

            # Create a PDF file for each image
            pdf_filename = f'output_{os.path.splitext(image_filename)[0]}.pdf'
            pdf_path = os.path.join(output_dir, pdf_filename)

            with PdfPages(pdf_path) as pdf:
                for model, preprocess_input, decode_predictions, model_name, input_size in models:
                    # Systematic Testing with 10 Patch Sizes Incremented by 50
                    for i in range(1, 11):
                        patch_size = i * 50
                        position = get_random_position(original_image, patch_size)
                        patch = create_random_patch(patch_size)
                        patched_image = apply_patch_to_image(original_image, patch, position)

                        # Classify original and patched images
                        original_preds = classify_image(model, original_image, preprocess_input, decode_predictions,
                                                        input_size)
                        patched_preds = classify_image(model, patched_image, preprocess_input, decode_predictions,
                                                       input_size)

                        # Save images and predictions to PDF
                        display_images_and_predictions(original_image, patched_image, original_preds, patched_preds,
                                                       model_name, pdf)

            print(f"Results for {image_filename} saved to {pdf_filename}")


if __name__ == "__main__":
    # main_not_compact()
    main_compact()
    main()