import cv2

cap = cv2.VideoCapture(0)

def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while True:
    rect, frame = cap.read()
    frame75 = rescale_frame(frame, percent=75)
    cv2.imshow('frame75', frame75)
   

cap.release()
cv2.destroyAllWindows()
