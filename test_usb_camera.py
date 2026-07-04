import cv2

index = 1
cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

print("opened:", cap.isOpened())
print("width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps:", cap.get(cv2.CAP_PROP_FPS))

if not cap.isOpened():
    raise SystemExit("Camera did not open")

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("failed to read frame")
        break

    cv2.imshow("USB Camera Test - press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
