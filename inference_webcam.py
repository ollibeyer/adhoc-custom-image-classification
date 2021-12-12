import numpy as np
import cv2
import json
import tensorflow as tf
import time

# Load config
CONFIG_FILE = "config.json"
f = open(CONFIG_FILE)
config = json.load(f)
f.close()

IMG_SIZE    = (224, 224)
MODEL_FILE  = "model.h5"
LABEL_FILE  = "labels.txt"
THRESHOLD   = config['threshold']

# Load model
model = tf.keras.models.load_model("model.h5")

# Load labels
label_file = open(LABEL_FILE, 'r')
labels = label_file.read().splitlines()

cv2.namedWindow("Inference")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

prev_frame_time = 0
new_frame_time = 0

while rval:
    cv2.imshow("Inference", frame)
    rval, frame = vc.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = np.expand_dims(img,axis=0)
    img = img[...,::-1].astype(np.float32)
    
    x_pos = 10
    y_pos = 25
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, "fps: " + str(round(fps)), (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0, 127), 1)

    # Predict
    preds = model.predict(img)[0]
    l_pred = []
    for i in range(len(preds)):
        if preds[i] >= THRESHOLD:
            l_pred.append((labels[i], round(preds[i]*100, 1)))
    l_pred.sort(key=lambda y: y[1], reverse=True)
    for i in range(len(l_pred)):
        pred_text = "'" + l_pred[i][0] + "' " + str(l_pred[i][1]) + " %"
        y = y_pos + (i+1)*20
        cv2.putText(frame, "class: " + pred_text, (x_pos,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255, 127), 1)
    key = cv2.waitKey(20)
    if key == 27 or key == ord('q'): # exit on ESC or q
        break

cv2.destroyWindow("Inference")
vc.release()