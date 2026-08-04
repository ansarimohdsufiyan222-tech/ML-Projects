import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the image
image = cv2.imread("image.png")  # Replace with your image path

# Check if image is loaded
if image is None:
    print("Error: Image not found!")
    exit()

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Store original shape
original_shape = image.shape

# Reshape image into 2D array of pixels
pixels = image.reshape((-1, 3))

# Apply K-Means
k = 64  # Number of colors
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixels)

# Replace each pixel with its cluster center
compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]

# Convert to unsigned integers
compressed_pixels = np.uint8(compressed_pixels)

# Reshape back to original image
compressed_image = compressed_pixels.reshape(original_shape)

# Display Original and Compressed Images
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(compressed_image)
plt.title(f"Compressed Image ({k} Colors)")
plt.axis("off")

plt.show()

# Save compressed image
compressed_bgr = cv2.cvtColor(compressed_image, cv2.COLOR_RGB2BGR)
cv2.imwrite("compressed_image.jpg", compressed_bgr)

print("Image compression completed successfully!")
print("Compressed image saved as compressed_image.jpg")