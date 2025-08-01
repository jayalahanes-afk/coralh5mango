import os
import sys
import io
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter

# Ensure the standard output uses UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Load the trained model
model = load_model("mango_leaf_disease_model.h5")

# Define image preprocessing parameters
image_size = (224, 224)

# Define dummy class_labels for the CLI
class_names = [
    "Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back",
    "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"
]
class_labels = {i: name for i, name in enumerate(class_names)}

# Function to preprocess and predict the class of multiple images
def predict_images(img_paths):
    try:
        images = []
        predictions = []
        for img_path in img_paths:
            img = image.load_img(img_path, target_size=image_size)
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0  # Normalize pixel values
            images.append(img)
        
        # Create a batch from all images
        batch_images = np.vstack(images)
        predictions = model.predict(batch_images)
        
        results = []
        for i, pred in enumerate(predictions):
            predicted_class = class_labels[np.argmax(pred)]
            # Convert raw probabilities to a dictionary with rounded values
            prob_dict = {class_labels[idx]: round(float(prob), 9) for idx, prob in enumerate(pred)}
            results.append((os.path.basename(img_paths[i]), predicted_class, prob_dict))
        
        return results, predictions
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")

# Function to handle prediction and display results
def main():
    try:
        # Prompt the user to input file paths
        file_paths = input("Enter the paths of the image files separated by commas: ").split(',')
        file_paths = [path.strip() for path in file_paths if path.strip()]  # Clean up paths

        if file_paths:
            # Predict all images
            results, predictions = predict_images(file_paths)
            
            # Display the prediction results
            print("\nPrediction Results:")
            print("-" * 80)
            print(f"{'Filename':<30} {'Predicted Class':<20} {'Probabilities':<30}")
            print("-" * 80)
            
            for file_name, predicted_class, prob_dict in results:
                prob_str = ', '.join([f"{name}: {prob:.2f}" for name, prob in prob_dict.items()])
                print(f"{file_name:<30} {predicted_class:<20} {prob_str:<30}")
            
            # Determine final prediction based on all images
            all_predicted_classes = [class_labels[np.argmax(pred)] for pred in predictions]
            final_prediction = Counter(all_predicted_classes).most_common(1)[0][0]
            
            # Calculate the average probability for the final prediction class
            final_class_index = list(class_labels.values()).index(final_prediction)
            avg_probability = np.mean([pred[final_class_index] for pred in predictions]) * 100
            
            # Display final prediction and its probability
            print("\nFinal Prediction:")
            print(f"Class: {final_prediction}, Probability: {avg_probability:.2f}%")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    main()
