from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="hammer-urc-data/data.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("mallet1.jpg")
results[0].show()
# results.save("output/")

try:
    annotated_img = results[0].plot()
    cv2.imwrite("output/annotated_mallet1.jpg", annotated_img)
except:
    print("Could not write")
    pass

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
