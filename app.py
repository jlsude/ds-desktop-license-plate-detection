from ultralytics import YOLO
import easyocr
import cv2


# Load model
model = YOLO('LicensePlateDetector.pt')

# load reader
reader = easyocr.Reader(['en'], gpu=True)

def perform_ocr_on_image(img, coordinates):
    x, y, w, h = map(int, coordinates)
    cropped_img = img[y:h, x:w]

    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray_img)

    text = ""
    for res in results:
        if len(results) == 1 or (len(res[1]) > 6 and res[2] > 0.2):
            text = res[1]

    return str(text)

# Use YOLO model to detect objects in the frame
results = model(source="0", show=True) #source is set to "0" to use the webcam

for result in results:
    coordinates = result.boxes.xyxy

    # Get the license plate number
    licensePlate = perform_ocr_on_image(result.img, coordinates)

    #print the license plate number
    print(licensePlate)








