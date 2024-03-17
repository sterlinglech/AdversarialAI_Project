import matplotlib
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # Set the backend to 'Agg'


def mutate_image(image_array, noise_level=5):
    """
    Applies random noise to an image to create a mutation.
    Can we apply random mutations here? Like find different types of mutation?
    """
    noise = np.random.normal(0, noise_level, image_array.shape)
    mutated_image = np.clip(image_array + noise, 0, 255)
    return mutated_image


def main(img_path, noise_level=22, generations=20):
    # Load the pre-trained ResNet50 model
    model = ResNet50(weights='imagenet')

    # Load and preprocess an image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    original_prediction = model.predict(x)
    print("Original prediction:", decode_predictions(original_prediction, top=1)[0])

    for i in range(generations):  # Number of generations
        mutated_x = mutate_image(x, noise_level)
        prediction = model.predict(mutated_x)
        top_pred = decode_predictions(prediction, top=1)[0]

        print(f"Generation {i + 1}, top prediction: {top_pred}")
        if top_pred[0][1] != decode_predictions(original_prediction, top=1)[0][0][1]:
            print("Misclassification achieved.")
            plt.imshow(mutated_x[0].astype('uint8'))
            plt.title(f"Adversarial Image - Generation {i + 1}")
            plt.show()
            break


if __name__ == "__main__":
    img_path = '_images/bananas1.JPG'  # Update this to the path of your image
    main(img_path)
