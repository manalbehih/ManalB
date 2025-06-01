import cv2
import numpy as np
import os


def process_image(input_path, output_path):
    # Load the image from the given path
    image = cv2.imread(input_path)
    if image is None:
        print("Error: Could not load image from", input_path)
        return

    # Define a structuring element (kernel) for the opening operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Apply the opening operation (erosion followed by dilation)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # # Apply a Gaussian smoothing filter to the opened image
    # smoothed = cv2.GaussianBlur(opened, (5, 5), 0)

    # Convert the smoothed image to grayscale before thresholding
    gray = cv2.cvtColor(opened, cv2.COLOR_BGR2GRAY)

    # # Convert the grayscale image to a binary image using Otsu's thresholding method
    # # The function returns a tuple (threshold value, binary image), so we take the binary image.
    # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure that the output directory exists; if not, create it.
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the binary image to the specified output path
    cv2.imwrite(output_path, gray)
    print(f"Processed binary image saved to {output_path}")


if __name__ == "__main__":
    # Specify the path to your input image and output image file
    input_image_path = "ImageFolder/Layer1.jpg"  # Replace with your actual image file path
    output_image_path = "processed_image/processed_image1.jpg"  # Replace with your desired output file path

    process_image(input_image_path, output_image_path)
