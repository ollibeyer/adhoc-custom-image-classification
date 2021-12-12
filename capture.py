import cv2
import os

# Create/read directory
PATH_ROOT="dataset"
os.makedirs(PATH_ROOT, exist_ok=True)

# Function to show number of existing images
def num_images():
    cats = os.listdir(PATH_ROOT)
    cats.sort()
    print("Number of training images:")
    for cat in cats:
        num = len(os.listdir(os.path.join(PATH_ROOT, cat)))
        print("'" + cat + "': " + str(num))

# Function for adding training images of category (0-9)
def addImage (frame, category):
    cat_path = os.path.join(PATH_ROOT, category)
    if os.path.isdir(cat_path):
        idx = len(os.listdir(cat_path))
    else:
        os.makedirs(cat_path)
        idx = 0
    filename = os.path.join(cat_path, "img_" + str(idx) +".jpg")
    cv2.imwrite(filename, frame)
    num_images()

# List of valid categories
category_list = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]

cv2.namedWindow("Capture")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("Capture", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key in category_list:
        addImage(frame, chr(key))
    if key == 27 or key == ord('q'): # exit on ESC or q
        break

cv2.destroyWindow("Capture")
vc.release()