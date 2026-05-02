from ultralytics import YOLO
import cv2

def main():
    # Load trained YOLOv11 model
    # Replace 'best.pt' with the path to your trained model file
    model = YOLO("best.pt")

    # Load input image
    image_path = "test.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image '{image_path}'")
        return

    # Perform detection
    results = model(image)

    # Get annotated image
    annotated_image = results[0].plot()

    # Display output
    cv2.imshow("Forest Fire and Smoke Detection using YOLOv11", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save output image
    output_path = "output_detected.jpg"
    cv2.imwrite(output_path, annotated_image)
    print(f"Detection completed successfully. Output saved as '{output_path}'")

if __name__ == "__main__":
    main()
