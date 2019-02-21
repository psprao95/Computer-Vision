import cv2

videocapture = cv2.VideoCapture(0)

if not videocapture.isOpened():
    print("Can't open default video camera")

cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE);

showframe = True
while(showframe):
    ret, frame = videocapture.read()

    if not ret:
        print("Can't capture frame")
        break

    cv2.imshow("video", frame)
    if cv2.waitKey(30) >= 0:
       showframe = False

videocapture.release()
cv2.destroyAllWindows()
