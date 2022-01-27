import cv2

cap=cv2.VideoCapture("./video/highway_traffic.mp4")

detector = cv2.createBackgroundSubtractorMOG2(history=400,varThreshold=50)
def checkInsiders(t):
    print(t)
    k = []
    for i, val in enumerate(t):
        x, y, w, h = val
        for j, val2 in enumerate(t):
            if j > i:
                xp, yp, wp, hp = val2
                if (xp < x < xp + wp and yp < y < yp + hp) or (x < xp < x + w and y < yp < y + h):
                    if wp * hp > w * h:
                        k.append(i)
                    else:
                        k.append(j)
    k=list(set(k))
    k.sort(reverse=True)
    for l in k:
        t.pop(l)
    print(t)
    print("-----------------------------------------")
    return t

while True:
    succ,img=cap.read()
    # try 1.0
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("imgGray", imgGray)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    # cv2.imshow("imgBlur", imgBlur)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 16)
    # cv2.imshow("imgThreshold", imgThreshold)
    imgMedian = cv2.medianBlur(imgThreshold, 3)
    # cv2.imshow("imgMedian", imgMedian)

    mask=detector.apply(img)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    conts,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rects=[]
    for cnt in conts:
        area=cv2.contourArea(cnt)
        if area > 400:
            x,y,w,h=cv2.boundingRect(cnt)
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            rects.append((x,y,w,h))
            # cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
    rects=checkInsiders(rects)
    for x,y,w,h in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (250, 0, 250), 2)

    #cv2.imshow("mask", mask)
    cv2.imshow("origin", img)



    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()