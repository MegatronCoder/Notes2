import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import filters
from scipy import ndimage

def load_image(path):
    """Load and preprocess image"""
    img = cv2.imread(path, 0)  # Read as grayscale
    return img

def simple_thresholding(img, threshold=127):
    """Apply simple thresholding"""
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary

def adaptive_thresholding(img):
    """Apply adaptive thresholding"""
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    return binary

def otsu_thresholding(img):
    """Apply Otsu's thresholding"""
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def region_growing(img, seed_point=None):
    """Simple region growing implementation"""
    if seed_point is None:
        seed_point = (img.shape[0]//2, img.shape[1]//2)
    
    # Create mask
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[seed_point] = 255
    
    # Region growing parameters
    threshold = 10
    
    # Get seed point intensity
    seed_intensity = img[seed_point]
    
    # Create binary mask where similar pixels are marked
    similar_pixels = np.abs(img.astype(np.int32) - seed_intensity) < threshold
    return (similar_pixels * 255).astype(np.uint8)

def watershed_segmentation(img):
    """Apply watershed segmentation"""
    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), markers)
    return (markers > 1).astype(np.uint8) * 255

def plot_results(original, segmented, title):
    """Plot original and segmented images side by side"""
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(segmented, cmap='gray')
    plt.title(f'Segmented Image ({title})')
    plt.axis('off')
    plt.show()

def main():
    # Get image path from user
    image_path = input("Enter the path to your image: ")
    img = load_image(image_path)
    
    if img is None:
        print("Error loading image!")
        return
    
    while True:
        print("\nImage Segmentation Menu:")
        print("1. Simple Thresholding")
        print("2. Adaptive Thresholding")
        print("3. Otsu Thresholding")
        print("4. Region Growing")
        print("5. Watershed Segmentation")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == '1':
            threshold = int(input("Enter threshold value (0-255): "))
            result = simple_thresholding(img, threshold)
            plot_results(img, result, "Simple Thresholding")
            
        elif choice == '2':
            result = adaptive_thresholding(img)
            plot_results(img, result, "Adaptive Thresholding")
            
        elif choice == '3':
            result = otsu_thresholding(img)
            plot_results(img, result, "Otsu Thresholding")
            
        elif choice == '4':
            result = region_growing(img)
            plot_results(img, result, "Region Growing")
            
        elif choice == '5':
            result = watershed_segmentation(img)
            plot_results(img, result, "Watershed")
            
        elif choice == '6':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()
