import datetime

import cv2
import os

import numpy as np
from yolov5 import YOLOv5

from sort import Sort
from telegram_client import alarm

path = 'video'
video_name = 'depositphotos_324815918-stock-video-people-crosswalk-busy-street-zebra.mp4'

cap = cv2.VideoCapture(os.path.join(path, video_name))
mot_tracker = Sort()

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# set model params
model_path = "weights/yolov5s.pt"  # it automatically downloads yolov5s model to given path
device = "cpu"  # or "c"

model = YOLOv5(model_path, device)

num_crossed = 0
is_tracked = set()
crossed_the_line = set()
# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        line_0 = int(width * .65)
        line_1 = int(width * .7)
        # perform inference
        results = model.predict(frame)
        detections = results.pred[0]
        detections_person = detections[np.where(detections[:, 5] == 0)]
        tracked_objects = mot_tracker.update(detections_person)

        for tr in tracked_objects:
            x1, y1, x2, y2 = int(tr[0]), int(tr[1]), int(tr[2]), int(tr[3])
            idx = int(tr[4])
            if line_0 < x2 < line_1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
                cv2.putText(frame, str(idx), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                if idx in is_tracked and idx not in crossed_the_line:
                    is_tracked.discard(idx)
                    crossed_the_line.add(idx)
                    print(f'Пересек черту - {idx}')

                    # save crossed the line
                    time_stamp = datetime.datetime.now().time()
                    file_name = f'id_{idx}_timestamp_{time_stamp.strftime("%H-%M-%S")}'
                    file_path = os.path.join('crossed_the_line', file_name + '.png')
                    this_frame = frame[y1:y2, x1:x2]
                    # print(file_path)
                    cv2.imwrite(file_path, this_frame)
                    message = f'Пешеход {idx} пересек черту в {time_stamp}'
                    # message to telegram
                    alarm(message, file_path)
                    num_crossed = num_crossed + 1
                else:
                    is_tracked.add(idx)

        # Display the resulting frame
        cv2.line(frame, (line_0, 0), (line_0, height), (0, 255, 255), thickness=2)
        cv2.line(frame, (line_1, 0), (line_1, height), (0, 255, 0), thickness=2)
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break

    # When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()

alarm(f'Всего зафиксировано {num_crossed} переходов.')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass
