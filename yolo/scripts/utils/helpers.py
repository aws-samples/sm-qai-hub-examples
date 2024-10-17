import os
import glob
import sagemaker
import tarfile
from PIL import (
    Image, 
    ImageDraw, 
    ImageFont
)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import seaborn as sns


def download_tar_and_untar(model_s3_uri, local_base_path="./fine-tuned-yolo/"):
    """
    Downloads a tar file from an S3 URI, extracts its contents, and returns the path 
    to the first extracted PyTorch model (.pt) file.
    
    Args:
        model_s3_uri (str): The S3 URI of the model tar file to download.
        local_base_path (str): The local directory where the tar file will be downloaded 
                               and its contents extracted. Defaults to './fine-tuned-yolo/'.
    
    Returns:
        str: The local file path of the first extracted PyTorch model (.pt) file.
    
    Functionality:
        - Downloads a tar file from the specified S3 URI using SageMaker's S3Downloader.
        - Extracts the contents of the tar file to the specified local directory.
        - Identifies the first PyTorch model file (.pt) in the extracted contents.
        - Deletes the tar file after extraction.
    """
    
    # download the tar file
    local_model_tar_path = sagemaker.s3.S3Downloader.download(
        s3_uri=model_s3_uri,  # updated to use model_s3_uri
        local_path=local_base_path, 
    )[0]
    
    # Open the tar file
    with tarfile.open(local_model_tar_path, "r:gz") as tar:
        # Extract all contents to the specified path
        tar.extractall(path=local_base_path)
    
    # craft the model path after it's extracted
    local_model_pt_path = glob.glob(os.path.join(local_base_path, "*.pt"))[0]
    
    # Remove the tar file after extraction
    os.remove(local_model_tar_path)
    
    return local_model_pt_path


def build_image_label_pairs(data):
    """
    Builds a dictionary of image-label pairs for validation data from a YOLO dataset.

    Args:
        data: A data object containing paths to the dataset, typically extracted from
              a YAML configuration file. The object should have a `data_yaml` attribute
              that holds paths for 'path' (root directory of the dataset) and 
              'validation' (validation set directory).

    Returns:
        dict: A dictionary where the keys are the base filenames (without extension)
              and the values are dictionaries containing 'image_path' and 'label_path'
              for the corresponding image-label pair.
    """
    
    valid_images_path = glob.glob(
        os.path.join(data.data_yaml['path'], data.data_yaml['validation'], "*")
    )
    valid_images_path_dict = {os.path.basename(img_path).split('.')[0]: img_path for img_path in valid_images_path}
    valid_labels_path = glob.glob(
        os.path.join(data.data_yaml['path'], data.data_yaml['validation'].replace('images', 'labels'), "*")
    )
    valid_labels_path_dict = {os.path.basename(lbl_path).split('.')[0]: lbl_path for lbl_path in valid_labels_path}
    image_label_pairs = {}
    for _key in valid_images_path_dict:
        image_label_pairs[_key] = {"image_path": valid_images_path_dict[_key], "label_path": valid_labels_path_dict[_key]}
    return image_label_pairs


def draw_bounding_boxes(yolo_result, ground_truth, confidence_threshold=0.5):
    """
    Draws YOLO prediction and ground truth bounding boxes on an image, with optional confidence threshold filtering.

    Args:
        yolo_result: A YOLO model result object that contains the original image,
                     bounding boxes, confidence scores, and predicted class labels.
        ground_truth (list): A list of ground truth bounding boxes, where each entry
                             is a string containing class index, center_x, center_y, 
                             width, and height (normalized values).
        confidence_threshold (float): The minimum confidence score for displaying 
                                      YOLO predictions. Defaults to 0.5.

    Returns:
        PIL.Image: The image with YOLO predicted bounding boxes and ground truth
                   boxes drawn on it.
                   
    Functionality:
        - Loads the original image from the YOLO result.
        - Draws YOLO-predicted bounding boxes with their confidence scores if they
          exceed the confidence threshold.
        - Draws ground truth bounding boxes in a different color.
        - Labels the bounding boxes with class names and confidence scores.
        - Handles cases with no detections by displaying a 'No detections' message
          on the image.
    """
    
    # Load the image from the YOLO result
    image_array = yolo_result.orig_img
    image = Image.fromarray(image_array)

    # Initialize drawing context
    draw = ImageDraw.Draw(image)

    # Font settings for labeling (increase font size by 40%)
    try:
        font = ImageFont.truetype("arial.ttf", int(15 * 1.4))  # 40% larger
    except:
        font = ImageFont.load_default(size=18)

    # Get boxes and confidences from YOLO result
    boxes = yolo_result.boxes.xywh  # Format: (center_x, center_y, width, height)
    confidences = yolo_result.boxes.conf  # Confidence scores for the predictions
    labels = yolo_result.boxes.cls  # Predicted class labels (index-based)

    # List of class names from YOLO (update this with your specific names if different)
    class_names = yolo_result.names

    # Check if there are no detections
    if len(boxes) == 0:
        # Display a message at the top: "No detections"
        text = "No detections"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw a box and add the text at the top of the image
        text_position = (image.width // 2 - text_width // 2, 10)  # Centered horizontally
        draw.rectangle([text_position, (text_position[0] + text_width, text_position[1] + text_height)], fill="red")
        draw.text(text_position, text, fill="white", font=font)
    else:
        # Draw YOLO prediction boxes
        for i, (box, confidence, label_idx) in enumerate(zip(boxes, confidences, labels)):
            if confidence < confidence_threshold:
                continue  # Skip low-confidence predictions

            label = class_names[int(label_idx)]
            confidence_text = f"{label} {confidence:.2f}"

            # Convert from center_x, center_y, width, height to top-left, bottom-right
            x_center, y_center, width, height = box
            top_left = (x_center - width / 2, y_center - height / 2)
            bottom_right = (x_center + width / 2, y_center + height / 2)

            # Draw the prediction box
            draw.rectangle([top_left, bottom_right], outline="red", width=2)

            # Draw label text for prediction in the format: `label confidence` above the box
            text_bbox = draw.textbbox((0, 0), confidence_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_position = (top_left[0], top_left[1] - text_height)  # Positioning text above the box
            draw.rectangle([text_position, (top_left[0] + text_width, top_left[1])], fill="red")  # Background for text
            draw.text(text_position, confidence_text, fill="black", font=font)

    # Draw Ground Truth boxes
    for truth in ground_truth:
        truth_data = truth.split()
        class_idx, x_center, y_center, width, height = map(float, truth_data)

        label = class_names[int(class_idx)]
        label_text = f"{label}"

        # Convert normalized coordinates to pixel values
        x_center *= image.width
        y_center *= image.height
        width *= image.width
        height *= image.height

        # Convert from center_x, center_y, width, height to top-left, bottom-right
        top_left = (x_center - width / 2, y_center - height / 2)
        bottom_right = (x_center + width / 2, y_center + height / 2)

        # Draw the ground truth box
        draw.rectangle([top_left, bottom_right], outline="cyan", width=2)

        # Draw label text for ground truth below the box
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Positioning text below the box
        text_position = (top_left[0], bottom_right[1])
        draw.rectangle([text_position, (top_left[0] + text_width, bottom_right[1] + text_height)], fill="cyan")  
        draw.text(text_position, label_text, fill="black", font=font)

    # Return the image
    return image


def create_image_mosaic(image_paths, grid_size=(2, 2), figsize=(10, 10)):
    """
    Creates a mosaic of images using the given image paths.
    
    Parameters:
        image_paths (list of str): List of paths to image files.
        grid_size (tuple): Number of rows and columns in the grid (rows, cols).
        figsize (tuple): Size of the figure for the plot.
    """
    # Create a figure
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
    
    # Flatten the axes array for easier iteration
    axes = axes.ravel()

    # Load and display each image in the respective subplot
    for idx, img in enumerate(image_paths):
        # Display image on the axis
        axes[idx].imshow(img)
        # Remove x and y ticks
        axes[idx].axis('off')

    # Hide any unused subplots if image_paths < grid_size
    for i in range(len(image_paths), len(axes)):
        axes[i].axis('off')

    # Show the plot
    plt.tight_layout()
    plt.show()

    
def create_metrics_plots(confusion_matrix, curves_results, class_names):
    """
    Creates a mosaic of four plots:
    1. Confusion Matrix Heatmap
    2. Precision-Recall Curve for each class
    3. F1 Score vs. Confidence Threshold for each class
    4. Precision vs. Confidence Threshold for each class

    Parameters:
    - confusion_matrix: 2D NumPy array representing the confusion matrix.
    - curves_results: List containing curve data arrays.
        curves_results[0]: Precision-Recall data (Recall, Precision per class)
        curves_results[1]: F1-Confidence data (Confidence, F1 Score per class)
        curves_results[2]: Precision-Confidence data (Confidence, Precision per class)
    - class_names: List of class names corresponding to the classes in the confusion matrix.
    """

    # Exclude the background class if present
    if len(curves_results[0][1]) < len(class_names):
        class_names = class_names[1:]
        confusion_matrix = confusion_matrix[1:, 1:]

    num_classes = len(class_names)

    # Create figure and axes for the 2x2 mosaic
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # First plot: Confusion Matrix
    ax = axes[0, 0]
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_title('Confusion Matrix')

    # Second plot: Precision-Recall Curve
    pr_curve_data = curves_results[0]
    x_values = pr_curve_data[0]  # Recall values
    y_values = pr_curve_data[1]  # Precision values for each class

    ax = axes[0, 1]
    for i in range(y_values.shape[0]):
        ax.plot(x_values, y_values[i], label=class_names[i])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()

    # Third plot: F1 Score vs Confidence Threshold
    f1_curve_data = curves_results[1]
    x_values = f1_curve_data[0]  # Confidence values
    y_values = f1_curve_data[1]  # F1 scores for each class

    ax = axes[1, 0]
    for i in range(y_values.shape[0]):
        ax.plot(x_values, y_values[i], label=class_names[i])
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs. Confidence Threshold')
    ax.legend()

    # Fourth plot: Precision vs Confidence Threshold
    precision_conf_curve_data = curves_results[2]
    x_values = precision_conf_curve_data[0]  # Confidence values
    y_values = precision_conf_curve_data[1]  # Precision values for each class

    ax = axes[1, 1]
    for i in range(y_values.shape[0]):
        ax.plot(x_values, y_values[i], label=class_names[i])
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs. Confidence Threshold')
    ax.legend()

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

