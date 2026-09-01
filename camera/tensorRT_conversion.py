from ultralytics import YOLO

# Load a YOLO26 model
model = YOLO("yolo26n.pt")

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolo26n.engine'
