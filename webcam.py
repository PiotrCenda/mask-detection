import os
import cv2
from mask_detect_model import frame_classification, model_load, frame_to_tensor

def satisfy_reqs():
    os.system('pip install -r requirements.txt')


if __name__ == "__main__":
    satisfy_reqs()

    classes = ["mask", "no_mask"]

    model = model_load()

    cam = cv2.VideoCapture(0)

    print("\nPress \"q\" to close program")

    while True:
        success, frame = cam.read()

        if not success:
            print("Error: Failed to grab frame.")
            break

        result = frame_classification(frame_to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)), model)
        
        color = ((0, 0, 255) if result else (0, 255, 0))
        top, bottom, left, right = [5]*4

        frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        cv2.putText(img=frame, text=classes[result].upper(), org=(15, 40), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2, 
                    color=color, thickness=3)
        
        cv2.imshow("webcam", frame)

        # close program if "q" key is pressed
        if cv2.waitKey(1) == ord('q'):
            print("Closing window...")
            break

    cam.release()
    cv2.destroyAllWindows()