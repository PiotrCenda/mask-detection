import os
import cv2
from mask_detect_model import frame_classification, model_load, frame_to_tensor

def satisfy_reqs():
    os.system('pip install -r requirements.txt')


if __name__ == "__main__":
    satisfy_reqs()

    model = model_load()

    cam = cv2.VideoCapture(0)

    print("\nPress \"q\" to close program")

    while True:
        success, frame = cam.read()

        if not success:
            print("Error: Failed to grab frame.")
            break

        cv2.imshow("webcam", frame)
        frame_classification(frame_to_tensor(frame), model)

        # close program if "q" key is pressed
        if cv2.waitKey(1) == ord('q'):
            print("Closing window...")
            break

    cam.release()
    cv2.destroyAllWindows()