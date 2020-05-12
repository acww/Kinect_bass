import freenect
import cv2
import frame_convert2
import imutils
import numpy as np
import math
from array import array
from time import sleep
import simpleaudio as sa


cv2.namedWindow('Hand0 frame')
cv2.namedWindow('Hand frame')
cv2.namedWindow('Binary')
cv2.namedWindow('Hand binary')
cv2.namedWindow('Hand0 binary')
cv2.namedWindow('RGB')
keep_running = True

def play_note(string, loop):
    c = sa.WaveObject.from_wave_file("/home/pi/notes/c.wav")
    c = c.play
    cS = sa.WaveObject.from_wave_file("/home/pi/notes/c#.wav")
    cS = cS.play
    d = sa.WaveObject.from_wave_file("/home/pi/notes/d.wav")
    d = d.play
    dS = sa.WaveObject.from_wave_file("/home/pi/notes/d#.wav")
    dS = dS.play
    e = sa.WaveObject.from_wave_file("/home/pi/notes/e.wav")
    e = e.play
    f = sa.WaveObject.from_wave_file("/home/pi/notes/f.wav")
    f = f.play
    fS = sa.WaveObject.from_wave_file("/home/pi/notes/f#.wav")
    fS = fS.play
    g = sa.WaveObject.from_wave_file("/home/pi/notes/g.wav")
    g = g.play
    gS = sa.WaveObject.from_wave_file("/home/pi/notes/g#.wav")
    gS = gS.play
    a = sa.WaveObject.from_wave_file("/home/pi/notes/a.wav")
    a = a.play
    aS = sa.WaveObject.from_wave_file("/home/pi/notes/a#.wav")
    aS = aS.play
    b = sa.WaveObject.from_wave_file("/home/pi/notes/b.wav")
    b = b.play
    C = sa.WaveObject.from_wave_file("/home/pi/notes/oC.wav")
    C = C.play
    print(string)
    print(loop)
    if string == 1:
        if loop == 1:
            c()
        elif loop == 2:
            cS()
        else:
            d()
    elif string == 2:
        if loop == 1:
            dS()
        elif loop == 2:
            e()
        else:
            f()
    elif string == 3:
        if loop == 1:
            fS()
        elif loop == 2:
            g()
        else:
            gS()
    elif string == 4:
        if loop == 1:
            a()
        elif loop == 2:
            aS()
        else:
            b()
    else:
        C()

def set_image_up(data):
    img = frame_convert2.pretty_depth_cv(data)
    img = cv2.medianBlur(img,9)
    thresh, binary = cv2.threshold(img,50,255,cv2.THRESH_BINARY_INV)
    contour, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return img, binary, contour

def draw(data):
    dots = []
    img, binary, contour = set_image_up(data)
    cv2.imshow('Binary', binary)
    height, width = img.shape
    frame = img[0:height, 0:int(width/2)] #this line crops
    frame0 = img[0:height, int(width/2):width]
    height, width = binary.shape
    binary0 = binary[0:height, 0:int(width/2)] #this line crops
    binary1 = binary[0:height, int(width/2):width]
    height, width = binary0.shape
    height0, width0 = binary1.shape
    strum = hand(binary0, frame, height, width)
    if strum == 1:
        hand_0(binary1, frame0, width, height)

direction = 0
cy = 0
def hand(binary, frame, height, width):
    global cy, direction
    pre_cy = cy
    cv2.imshow('Hand binary', binary)
    contours,hierarchy= cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)<1:
        strum = 0
    else:
        cnt = contours[0]
        M = cv2.moments(cnt)
        cy = int(M['m01']/M['m00'])

        #find contour of max area(hand)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))

        #approx the contour a little
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)


        #make convex hull around hand
        hull = cv2.convexHull(cnt)

        #define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)

        #find the percentage of area not covered by hand in convex hull
        arearatio=((areahull-areacnt)/areacnt)*100

        #find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)

        # l = no. of defects
        l=0

        #code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt= (100,180)


            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))

            #distance between point and convex hull
            d=(2*ar)/a

            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57


            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d>10:
                l += 1
                cv2.circle(frame, far, 3, [255,0,0], -1)

            #draw lines around hand
            cv2.line(frame,start, end, [0,255,0], 2)


        l+=1

        #display corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l==1:
            if areacnt<2000:
                cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            else:
                if arearatio<12:
                    cv2.putText(frame,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

                else:
                    cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

        elif l==2:
            cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

        elif l==3:

              if arearatio<27:
                    cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
              else:
                    cv2.putText(frame,'ok',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

        elif l==4:
            cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

        elif l==5:
            cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

        elif l==6:
            cv2.putText(frame,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

        else :
            cv2.putText(frame,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

        cv2.imshow('Hand frame',frame)
        cv2.imshow('Hand frame',frame)
        pre_direction = direction
        if cy < pre_cy:
            direction = 0
        else:
            direction = 1
        strum = 0
        if direction != pre_direction:
            if abs(cy-pre_cy) > 10:
                strum = 1
    return strum


def hand_0(binary0, frame0, width, height):
    cv2.imshow('Hand0 binary', binary0)
    contours,hierarchy = cv2.findContours(binary0,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    cnt = contours[0]
    M = cv2.moments(cnt)

    cx = int(M['m10']/M['m00'])
    print('hi')
    #approx the contour a little
    epsilon = 0.0005*cv2.arcLength(cnt,True)
    approx= cv2.approxPolyDP(cnt,epsilon,True)


    #make convex hull around hand
    hull = cv2.convexHull(cnt)

    #define area of hull and area of hand
    areahull = cv2.contourArea(hull)
    areacnt = cv2.contourArea(cnt)

    #find the percentage of area not covered by hand in convex hull
    arearatio=((areahull-areacnt)/areacnt)*100

    #find the defects in convex hull with respect to hand
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)

    # l = no. of defects
    l=0

    #code for finding no. of defects due to fingers
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(approx[s][0])
        end = tuple(approx[e][0])
        far = tuple(approx[f][0])
        pt= (100,180)


        # find length of all sides of triangle
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        s = (a+b+c)/2
        ar = math.sqrt(s*(s-a)*(s-b)*(s-c))

        #distance between point and convex hull
        d=(2*ar)/a

        # apply cosine rule here
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57


        # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
        if angle <= 90 and d>10:
            l += 1
            cv2.circle(frame0, far, 3, [255,0,0], -1)

        #draw lines around hand
        cv2.line(frame0,start, end, [0,255,0], 2)


    l+=1

    #display corresponding gestures which are in their ranges
    font = cv2.FONT_HERSHEY_SIMPLEX
    if l==1:
        if areacnt<2000:
            cv2.putText(frame0,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        else:
            if arearatio<12:
                cv2.putText(frame0,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

            else:
                cv2.putText(frame0,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

    elif l==2:
        cv2.putText(frame0,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

    elif l==3:

          if arearatio<27:
                cv2.putText(frame0,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
          else:
                cv2.putText(frame0,'ok',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

    elif l==4:
        cv2.putText(frame0,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

    elif l==5:
        cv2.putText(frame0,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

    elif l==6:
        cv2.putText(frame0,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

    else :
        cv2.putText(frame0,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    segments = 4
    loop = 1
    string = l
    while loop < segments:
        seg = loop*(width/segments)
        if cx < seg:
            play_note(string, loop)
            loop = segments
        loop = loop + 1
    cv2.imshow('Hand0 frame', frame0)


def main(dev, data, timestamp):
    global keep_running, last
    img, binary, contour = set_image_up(data)
    draw(data)
    if cv2.waitKey(10) == 27:
        keep_running = False


def display_rgb(dev, data, timestamp):
    global keep_running
    cv2.imshow('RGB', frame_convert2.video_cv(data))
    if cv2.waitKey(10) == 27:
        keep_running = False


def body(*args):
    if not keep_running:
        raise freenect.Kill


print('Press ESC in window to stop')
freenect.runloop(depth=main,
                 video=display_rgb,
                 body=body)