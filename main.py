import cv2 as cv
import numpy as np
import dlib
import time
import math

def distance(x1, y1, x2, y2):
    #print("x1 :" + str(x1) + " y1 :" + str(y1) + " x2 :" + str(x2) + " y2 :" + str(y2) )
    x = (x1-x2) ** 2
    y = (y1-y2) ** 2
    d = (x + y) ** .5
    return d

def ear(f,pos):
    if(pos == "left"):
        i = 0
    else:
        i = 6
    s1x = f.part(36 + i).x
    s1y = f.part(36 + i).y
    u1x = f.part(37 + i).x
    u1y = f.part(37 + i).y
    u2x = f.part(38 + i).x
    u2y = f.part(38 + i).y
    s2x = f.part(39 + i).x
    s2y = f.part(39 + i).y
    b2x = f.part(40 + i).x
    b2y = f.part(40 + i).y
    b1x = f.part(41 + i).x
    b1y = f.part(41 + i).y
    # cv.line(frame, (s1x, s1y), (u1x, u1y), (0, 255, 0), 2)
    # cv.line(frame, (u1x, u1y), (u2x, u2y), (0, 255, 0), 2)
    # cv.line(frame, (u2x, u2y), (s2x, s2y), (0, 255, 0), 2)
    # cv.line(frame, (s2x, s2y), (b2x, b2y), (0, 255, 0), 2)
    # cv.line(frame, (b2x, b2y), (b1x, b1y), (0, 255, 0), 2)
    # cv.line(frame, (b1x, b1y), (s1x, s1y), (0, 255, 0), 2)
    vd1 = distance(u1x, u1y, b1x, b1y)
    vd2 = distance(u2x, u2y, b2x, b2y)
    hd = distance(s1x, s1y, s2x, s2y)
    ear = (((vd1 + vd2) / 2) / hd)
    #print(ear)
    return ear

def eye(frame, lm, pos):
    eyes = np.zeros(frame.shape, dtype="uint8")
    if pos == "left":
        k = 0
    else:
        k = 6
    eyelm = np.array([(lm.part(36 + k).x, lm.part(36 + k).y), (lm.part(37 + k).x, lm.part(37 + k).y), (lm.part(38 + k).x, lm.part(38 + k).y),
                      (lm.part(39 + k).x, lm.part(39 + k).y), (lm.part(40 + k).x, lm.part(40 + k).y), (lm.part(41 + k).x, lm.part(41 + k).y)])
    cv.fillConvexPoly(eyes, eyelm, (255, 255, 255))
    eyes = cv.cvtColor(eyes, cv.COLOR_BGR2GRAY)
    _, eyes = cv.threshold(eyes, 250, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(eyes, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # cv.drawContours(frame, contours, -1, (0, 255, 0), 2, lineType=cv.LINE_AA)
    # cv.imshow(pos + "Eyes", eyes)
    return cv.boundingRect(contours[0])

def meta(control, blinkno, elapsed, blinkrate, diff, xold, yold, xnew, ynew, t1, t2):
    print("Control Array:" + str(control))
    print("Number of blinks: " + str(blinkno))
    print("Seconds elapsed: " + str(elapsed))
    print("Blinkrate:" + str(blinkrate))
    print("Difference:" + str(diff))
    print("xold: " + str(xold) + " yold: " + str(yold))
    print("xnew: " + str(xnew) + " ynew: " + str(ynew))
    print("t1: " + str(t1) + " t2: " + str(t2))
    print("\n")


cap = cv.VideoCapture(0)
face_det = dlib.get_frontal_face_detector()
landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
window = cv.namedWindow("Video", cv.WINDOW_AUTOSIZE)

blinkno = 0
blinkrate = 0
start = time.perf_counter()
control = np.zeros(3, dtype="i")
xold = yold = 0
diff = 0
decision = 0
while True:
    t1 = math.floor(time.perf_counter() - start)
    ret, frame = cap.read()
    if ret == False:
        print("Camera not found")
        continue
    faces = face_det(frame)
    if(len(faces) == 0):
        cv.putText(frame, "NO FACE", (0,100), cv.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 3)
        cv.putText(frame, "DETECTED", (0, 225), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
    for f in faces:
        xnew = ynew = 0
        lm = landmarks(frame, f)
        xnew = xnew + lm.part(27).x
        ynew = ynew + lm.part(27).y
        # for i in range(68):
            # cv.putText(frame, str(i), (lm.part(i).x, lm.part(i).y), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
            # cv.circle(frame, (lm.part(i).x, lm.part(i).y), 1, (0, 255, 0), 1)
        # cv.circle(frame, (xnew, ynew), 1, (0, 255, 0), 1)
        x1 = f.left()
        y1 = f.top()
        x2 = f.right()
        y2 = f.bottom()
        lear = ear(lm, "left")
        rear = ear(lm, "right")
        eyear = (lear+rear)/2
        if eyear < 0.26:
            control[0] = 1
        else:
            control[0] = 0
        if eyear < .18:
            blinkno = blinkno + 1
        elapsed = time.perf_counter() - start
        if elapsed >= 10:
            blinkrate = blinkno / elapsed
            if (blinkrate <= 1):
                control[1] = 1
            else:
                control[1] = 0
        # lex1, ley1, lw, lh = eye(frame, lm, "left")
        # cv.rectangle(frame, (lex1, ley1), (lex1 + lw, ley1 + lh), (0, 255, 255), 1)
        # rex1, rey1, rw, rh = eye(frame, lm, "right")
        # cv.rectangle(frame, (rex1, rey1), (rex1 + rw, rey1 + rh), (0, 255, 255), 1)
        t2 = round(time.perf_counter() - start)
        if t2 - t1 >= 1:
            diff = distance(xnew, ynew, xold, yold)
            xold = xnew
            yold = ynew
            if(diff >= 5):
                 control[2] = 1
                 cv.putText(frame, "MOTION DETECTED", (50, 50), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
            else:
                control[2] = 0
            decision = np.sum(control)
        # meta(control, blinkno, elapsed, blinkrate, diff, xold, yold, xnew, ynew, t1, t2
        if decision > 1.5:
            cv.putText(frame, "DRUNK", (x1, y2+50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            frame = cv.rectangle(frame, (x1 - 20, y1 - 20), (x2 + 20, y2 + 20), (0, 0, 255), 3)
        else:
            frame = cv.rectangle(frame, (x1 - 20, y1 - 20), (x2 + 20, y2 + 20), (0, 255, 0), 3)
            cv.putText(frame, "NORMAL", (x1, y2+50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv.imshow("Video", frame)
    if cv.waitKey(1) == 27:
        break
cap.release()
cv.destroyAllWindows()