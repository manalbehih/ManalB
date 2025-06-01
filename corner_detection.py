import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt


def generate_heatmap(histogram, heatmap_output_path):
    """
    Generates a heatmap from the corner density histogram, displays it,
    and saves it to the specified path.
    """
    # Ensure that the output directory for the heatmap exists
    output_dir = os.path.dirname(heatmap_output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the heatmap using matplotlib
    plt.figure()
    plt.imshow(histogram, cmap='hot', interpolation='nearest')
    plt.title("Corner Density Heatmap")
    plt.xlabel("Bin (X)")
    plt.ylabel("Bin (Y)")
    plt.colorbar()
    plt.savefig(heatmap_output_path)
    print(f"Heatmap saved to {heatmap_output_path}")
    plt.show()
    plt.close()


def detect_corners_with_density(image_path, output_path, density_threshold=3):
    """
    Detects corners in an image, marks them with green circles,
    draws a red square (17x17 pixels) around regions with a high density
    of corners (>= density_threshold), displays and saves the final image,
    and returns the density histogram.
    """
    # Load the image from the specified path
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return None

    # Convert the image to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Parameters for the corner detection
    max_corners = 1000      # Maximum number of corners to detect
    quality_level = 0.05    # Minimal accepted quality of image corners
    min_distance = 1        # Minimum possible Euclidean distance between corners

    # Detect corners using goodFeaturesToTrack
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)

    # Store the coordinates of the detected corners
    corner_coords = []
    if corners is not None:
        corners = np.int0(corners)  # Convert to integer for drawing
        for corner in corners:
            x, y = corner.ravel()   # Get x, y coordinates
            corner_coords.append((x, y))
            # Draw a small green circle for each detected corner
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

    # Define the bin (square) size for density estimation
    bin_size = 17
    height, width = gray.shape[:2]
    # Determine the number of bins along the x and y dimensions
    bins_x = math.ceil(width / bin_size)
    bins_y = math.ceil(height / bin_size)

    # Create a histogram (2D array) to count corners per bin
    histogram = np.zeros((bins_y, bins_x), dtype=np.int32)
    for (x, y) in corner_coords:
        bin_x = x // bin_size
        bin_y = y // bin_size
        histogram[bin_y, bin_x] += 1

    # For bins with high density of corners, draw a red square around the bin
    for i in range(bins_y):
        for j in range(bins_x):
            if histogram[i, j] >= density_threshold:
                top_left = (j * bin_size, i * bin_size)
                bottom_right = (min((j + 1) * bin_size - 1, width - 1),
                                min((i + 1) * bin_size - 1, height - 1))
                cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 1)

    # Ensure that the output directory exists; if not, create it.
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the final image with detected corners and red squares
    cv2.imwrite(output_path, image)
    print(f"Final output saved to {output_path}")

    # Display the final image
    cv2.imshow("Corners with Density", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return histogram, bin_size


def filter_dark_squares(image_path, histogram, bin_size, density_threshold, intensity_threshold, output_filtered_path):
    """
    Filters the high-density bins using an intensity threshold.
    For each bin with density >= density_threshold, the average intensity is computed
    from the original image's grayscale version. If the average intensity is below intensity_threshold,
    a blue square is drawn on a copy of the original image.
    The function then displays and saves the filtered image.
    """
    # Re-read the original image and convert to grayscale for intensity measurement
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        print("Error: Could not load original image for filtering.")
        return
    gray_orig = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    height, width = gray_orig.shape[:2]

    # Create a copy of the original image to draw blue squares on
    filtered_image = orig_image.copy()

    bins_y, bins_x = histogram.shape

    # Iterate over each bin that had high density
    for i in range(bins_y):
        for j in range(bins_x):
            if histogram[i, j] >= density_threshold:
                # Determine bin boundaries
                x_start = j * bin_size
                x_end = min((j + 1) * bin_size, width)
                y_start = i * bin_size
                y_end = min((i + 1) * bin_size, height)
                # Calculate average intensity in the bin using the original grayscale image
                region = gray_orig[y_start:y_end, x_start:x_end]
                avg_intensity = np.mean(region)
                # If the region is darker than the intensity threshold, mark with a blue square
                if avg_intensity < intensity_threshold:
                    top_left = (x_start, y_start)
                    bottom_right = (x_end - 1, y_end - 1)
                    cv2.rectangle(filtered_image, top_left, bottom_right, (255, 0, 0), 1)

    # Ensure that the output directory exists; if not, create it.
    output_dir = os.path.dirname(output_filtered_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the filtered image with blue squares
    cv2.imwrite(output_filtered_path, filtered_image)
    print(f"Filtered squares image saved to {output_filtered_path}")

    # Display the filtered image
    cv2.imshow("Filtered Dark Squares", filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Specify the paths for your input image, final output image, heatmap, and filtered squares image
    image_path = "ImageFolder/Layer1.jpg"           # Replace with your image file path
    output_path = "density_pic/corners_with_density.jpg"            # Final image with red squares
    heatmap_output_path = "density_pic/corner_density_heatmap.png"    # Output path for the heatmap image
    filtered_output_path = "density_pic/filtered_dark_squares.jpg"    # Output path for the filtered (blue squares) image

    density_threshold = 3    # Adjust the threshold as needed for high-density detection
    intensity_threshold = 100  # Threshold for average intensity (0-255); lower values mean darker regions

    # Detect corners, mark high-density areas, and obtain the histogram and bin size
    histogram, bin_size = detect_corners_with_density(image_path, output_path, density_threshold)

    # Generate and save the heatmap of corner density if histogram exists
    if histogram is not None:
        generate_heatmap(histogram, heatmap_output_path)

        # Filter the high-density squares using the intensity threshold and mark them with blue squares
        filter_dark_squares(image_path, histogram, bin_size, density_threshold, intensity_threshold, filtered_output_path)
