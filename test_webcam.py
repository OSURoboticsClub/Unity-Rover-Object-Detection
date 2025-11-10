from ultralytics import YOLO
import cv2

# Load a pretrained YOLO model (e.g., YOLO11n)
model = YOLO("runs/detect/train11/weights/best.pt")

# Open webcam (0 = default camera)
results = model(source=0, stream=True, conf=0.75)

# Loop over the generator and display results
for result in results:
    # result.plot() returns an annotated frame (NumPy array)
    frame = result.plot()

    # Display the frame
    cv2.imshow("YOLO Live", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
