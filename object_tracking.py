import numpy as np
import cv2

cap = cv2.VideoCapture(0)

blank = np.zeros((480, 640, 3), dtype=np.uint8)
last_x = 0
last_y = 0

# To save video
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
save_vid = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    # Our operations on the frame come here
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_bound = np.array([30, 80, 40])
    upper_bound = np.array([102, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # filter the noise
    kernelOpen = np.ones((5, 5))
    kernelClose = np.ones((20, 20))

    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
    final_mask = maskClose

    # find contours and draw rectangle around the source
    # ret, thresh = cv2.threshold(final_mask, 127,255,0)
    _, conts, hierarchy = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, conts, -1, (255, 0, 0), 3)

    for i in range(len(conts)):
        area = cv2.contourArea(conts[i])
        if area > 700:
            x, y, w, h = cv2.boundingRect(conts[i])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            M = cv2.moments(conts[i])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)  # add circle to main frame
            if last_x != 0 and last_y != 0:
                cv2.line(blank, (last_x, last_y), (cX, cY), (255, 255, 255), 5)  # add line to blank frame
            last_x = cX
            last_y = cY

    out = cv2.add(blank, frame)  # add blank frame to main frame

    #save the vido
    save_vid.write(out)

    # Display the resulting frame
    cv2.imshow("object tracking", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
save_vid.release()
cv2.destroyAllWindows()
