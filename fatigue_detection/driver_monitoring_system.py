import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)

cap = cv2.VideoCapture(0)

closed_count = 0

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        1.1,
        5
    )

    fatigue = False

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        roi_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            1.1,
            3
        )

        if len(eyes) < 1:
            closed_count += 1
        else:
            closed_count = 0

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                frame,
                (x+ex, y+ey),
                (x+ex+ew, y+ey+eh),
                (0, 255, 0),
                2
            )

    if closed_count > 15:
        fatigue = True

    if fatigue:
        cv2.putText(
            frame,
            "FATIGUE WARNING!",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("Fatigue Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()