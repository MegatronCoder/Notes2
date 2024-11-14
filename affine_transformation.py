import cv2
import numpy as np
import matplotlib.pyplot as plt


clicked_points_1 = [] 
clicked_points_2 = []
current_image = 1
points_needed = 3

def translate_image(image, tx, ty):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (cols, rows))

def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))

def scale_image(image, sx, sy):
    return cv2.resize(image, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)

def reflect_image(image, axis):
    if axis == 'x':
        return cv2.flip(image, 0)
    elif axis == 'y':
        return cv2.flip(image, 1)
    else:
        return image

def shear_image(image, sx, sy):
    rows, cols = image.shape[:2]
    M = np.float32([[1, sx, 0], [sy, 1, 0]])
    return cv2.warpAffine(image, M, (cols, rows))

def main():
    image_path = input("Enter the path to your image: ")
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not read the image.")
        return

    while True:
        print("\nAffine Transformation Menu:")
        print("1. Translation")
        print("2. Rotation")
        print("3. Scaling")
        print("4. Reflection")
        print("5. Shearing")
        print("6. Affine Transformation")
        print("7. Exit")

        choice = input("Enter your choice (1-6): ")

        if choice == '1':
            tx = int(input("Enter translation in x direction: "))
            ty = int(input("Enter translation in y direction: "))
            result = translate_image(image, tx, ty)
        elif choice == '2':
            angle = float(input("Enter rotation angle in degrees: "))
            result = rotate_image(image, angle)
        elif choice == '3':
            sx = float(input("Enter scaling factor for x direction: "))
            sy = float(input("Enter scaling factor for y direction: "))
            result = scale_image(image, sx, sy)
        elif choice == '4':
            axis = input("Enter axis for reflection (x/y): ").lower()
            result = reflect_image(image, axis)
        elif choice == '5':
            sx = float(input("Enter shearing factor for x direction: "))
            sy = float(input("Enter shearing factor for y direction: "))
            result = shear_image(image, sx, sy)

        elif choice == '6':

            def get_coordinates(event, x, y, flags, param):
                global current_image
                if event == cv2.EVENT_LBUTTONDOWN:
                    if current_image == 1 and len(clicked_points_1) < points_needed:
                        clicked_points_1.append((x, y))
                        print(f"Clicked at: ({x}, {y}) on Image 1")
                        cv2.circle(resized_image1, (x, y), 5, (0, 0, 255), -1)
                        cv2.imshow("Image", resized_image1)
                        if len(clicked_points_1) == points_needed:
                            print("3 points selected on Image 1. Switching to Image 2.")
                            current_image = 2
                            cv2.imshow("Image", resized_image2)
                    elif current_image == 2 and len(clicked_points_2) < points_needed:
                        clicked_points_2.append((x, y))
                        print(f"Clicked at: ({x}, {y}) on Image 2")
                        cv2.circle(resized_image2, (x, y), 5, (0, 0, 255), -1)
                        cv2.imshow("Image", resized_image2)
                        if len(clicked_points_2) == points_needed:
                            print("3 points selected on Image 2.")

            def resize_image(image, target_size=(1024, 1024)):
                return cv2.resize(image, target_size)

            # Load and resize images
            image1 = cv2.imread(r"C:\Users\Manthan\Desktop\temp33\Screenshot 2024-09-30 225715.png")
            image2 = cv2.imread(r"C:\Users\Manthan\Desktop\temp33\Screenshot 2024-09-30 225701.png")

            resized_image1 = resize_image(image1)
            resized_image2 = resize_image(image2)

            # Create a single window for both images
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Image", get_coordinates)
            cv2.imshow("Image", resized_image1)

            # Main loop for key control
            while len(clicked_points_1) < points_needed or len(clicked_points_2) < points_needed:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            cv2.destroyAllWindows()

            # Affine transformation
            if len(clicked_points_1) == points_needed and len(clicked_points_2) == points_needed:
                src_pts = np.float32(clicked_points_1[:3])
                dst_pts = np.float32(clicked_points_2[:3])
                
                matrix = cv2.getAffineTransform(src_pts, dst_pts)
                result = cv2.warpAffine(resized_image1, matrix, (1024, 1024))

                # Plot results
                plt.figure(figsize=(15, 5))
                plt.subplot(131), plt.imshow(cv2.cvtColor(resized_image1, cv2.COLOR_BGR2RGB))
                plt.title('Image 1'), plt.axis('off')
                plt.subplot(132), plt.imshow(cv2.cvtColor(resized_image2, cv2.COLOR_BGR2RGB))
                plt.title('Image 2'), plt.axis('off')
                plt.subplot(133), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                plt.title('Transformed Image 1'), plt.axis('off')
                plt.show()
            else:
                print("Not enough points selected for affine transformation.")

        elif choice == '7':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
            continue

        cv2.imshow('Original Image', image)
        cv2.imshow('Transformed Image', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()