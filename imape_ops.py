import numpy as np
import cv2
import matplotlib.pyplot as plt

def display_image(image, title="Image"):
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()

def convolve2d(image, kernel):
    m, n = kernel.shape
    y, x = image.shape
    y = y - m + 1
    x = x - n + 1
    new_image = np.zeros((y, x))
    for i in range(y):
        for j in range(x):
            new_image[i][j] = np.sum(image[i:i+m, j:j+n] * kernel)
    return new_image

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def sobel_filter(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    grad_x = convolve2d(image, kernel_x)
    grad_y = convolve2d(image, kernel_y)
    
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude)) * 255
    
    return gradient_magnitude.astype(np.uint8)

def median_filter(image, kernel_size=3):
    pad = kernel_size // 2
    padded_img = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')
    filtered_img = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_img[i:i+kernel_size, j:j+kernel_size]
            filtered_img[i, j] = np.median(window)
    
    return filtered_img

def gaussian_kernel(size, sigma=1):
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def gaussian_filter(image, kernel_size=5, sigma=1):
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve2d(image, kernel)

def process_image(image_path):
    image = cv2.imread(image_path)
    gray_image = to_grayscale(image)
    
    while True:
        print("\nMenu:")
        print("1. Display Original Image")
        print("2. Display Grayscale Image")
        print("3. Apply and Display Sobel Filter")
        print("4. Apply and Display Median Filter")
        print("5. Apply and Display Gaussian Filter")
        print("6. Compare All Filters")
        print("7. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            display_image(to_rgb(image), "Original Image")
        elif choice == '2':
            display_image(gray_image, "Grayscale Image")
        elif choice == '3':
            sobel_image = sobel_filter(gray_image)
            display_image(sobel_image, "Sobel Filter")
        elif choice == '4':
            median_image = median_filter(gray_image)
            display_image(median_image, "Median Filter")
        elif choice == '5':
            gaussian_image = gaussian_filter(gray_image)
            display_image(gaussian_image, "Gaussian Filter")
        elif choice == '6':
            sobel_image = sobel_filter(gray_image)
            median_image = median_filter(gray_image)
            gaussian_image = gaussian_filter(gray_image)

            fig, axs = plt.subplots(2, 3, figsize=(20, 20))

            colored_image = to_rgb(image)
            axs[0, 0].imshow(colored_image)
            axs[0, 0].set_title("Original Image")
            axs[0, 0].axis('off')

            axs[0, 1].imshow(gray_image, cmap='gray')
            axs[0, 1].set_title("Grayscale Image")
            axs[0, 1].axis('off')

            axs[0, 2].imshow(gaussian_image, cmap='gray')
            axs[0, 2].set_title("Gaussian Image")
            axs[0, 2].axis('off')

            axs[1, 0].imshow(sobel_image, cmap='gray')
            axs[1, 0].set_title("Sobel Filter")
            axs[1, 0].axis('off')

            axs[1, 1].imshow(median_image, cmap='gray')
            axs[1, 1].set_title("Median Filter")
            axs[1, 1].axis('off')

            axs[1, 2].imshow(colored_image)
            axs[1, 2].set_title("Original Image")
            axs[1, 2].axis('off')

            plt.tight_layout()
            plt.show()
        elif choice == '7':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    image_path = r"C:\Dataset\bdd100k\bdd100k\images\100k\train\91a4614d-82bce903.jpg"
    process_image(image_path)
