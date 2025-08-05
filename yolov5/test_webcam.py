import cv2

# Try VideoCapture(0), if not working, change to 1
cap = cv2.VideoCapture(1)  # if this fails, try 2 or 3


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
