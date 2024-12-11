import cv2
import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Không thể mở hình ảnh.")
        return

    results = model(image)
    boxes = results.xyxy[0].numpy()

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        label = f"{model.names[int(cls)]}: {conf:.2f}"

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Detected Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def upload_image():
    image_path = filedialog.askopenfilename()
    if image_path:
        detect_image(image_path)

root = tk.Tk()
root.title("YOLOv5 Image Detection")
root.geometry("300x150")

upload_btn = tk.Button(root, text="Chọn Hình Ảnh", command=upload_image)
upload_btn.pack(expand=True)

root.mainloop()
