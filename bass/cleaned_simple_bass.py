import freenect             # Library for the kinect
import cv2                  # Opencv library
import frame_convert2       # Easy manipulation of lists
import math                 # Math library
import numpy as np          # Used for creating arrays
import simpleaudio as sa    # Most basic audio manipulator i could find
                            # https://simpleaudio.readthedocs.io/en/latest/index.html

cv2.namedWindow('Hand0 frame')
cv2.namedWindow('Binary')
cv2.namedWindow('Hand binary')
cv2.namedWindow('Hand0 binary')
cv2.namedWindow('RGB')
cv2.namedWindow('note')
keep_running = True

def play_note(string, loop):
    blank = np.zeros((500, 500, 3), dtype = "uint8")
    # Gives a path to a wav file and asigns it to a function
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
    """I know there are neater ways of doing this however
    its alot easier to visulize, and debug, when its written
    out in a truth table. The idea is that there are twelve
    notes in an octave so we can devide this by 4 (the number
    of strings on a bass guitar) and then have three segments
    on the left part of the image. This while limiting us to
    an octave in range allows the user interface to be easier."""
    if string == 1:
        if loop == 1:
            c()
            note = 'c'
        elif loop == 2:
            cS()
            note = 'c#'
        else:
            d()
            note = 'd'
    elif string == 2:
        if loop == 1:
            dS()
            note = 'd#'
        elif loop == 2:
            e()
            note = 'e'
        else:
            f()
            note = 'f'
    elif string == 3:
        if loop == 1:
            fS()
            note = 'f#'
        elif loop == 2:
            g()
            note = 'g'
        else:
            gS()
            note = 'g#'
    elif string == 4:
        if loop == 1:
            a()
            note = 'a'
        elif loop == 2:
            aS()
            note = 'a#'
        else:
            b()
            note = 'b'
    else:
        C()
        note = 'C'
    cv2.putText(blank, note, (200,250), cv2.FONT_ITALIC, 5, (255,255,255), 3, cv2.LINE_AA)
    cv2.imshow('note', blank)

def set_image_up(data):     # Gives us a depth, binary and contoured image
    img = frame_convert2.pretty_depth_cv(data)
    img = cv2.medianBlur(img, 9)
    thresh, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
    contour, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return img, binary, contour

def draw(data):
    img, binary, contour = set_image_up(data)
    cv2.imshow('Binary', binary)
    height, width = img.shape
    frame = img[0:height, 0:int(width/2)]       # Splits the screen into two so we can distinguish between the two hands
    frame0 = img[0:height, int(width/2):width]
    height, width = binary.shape
    binary0 = binary[0:height, 0:int(width/2)]
    binary1 = binary[0:height, int(width/2):width]
    height, width = binary0.shape
    height0, width0 = binary1.shape
    strum = hand(binary0, frame, height, width)
    if strum == 1:      # If we are strumming play a note
        hand_0(binary1, frame0, width, height)

direction = 0
cy = 0
def hand(binary, frame, height, width):
    global cy, direction
    pre_cy = cy
    cv2.imshow('Hand binary', binary)
    contours, hierarchy= cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 1:       # If we have nothing to find dont try finding it
        strum = 0
    else:
        cnt = contours[0]
        M = cv2.moments(cnt)
        cy = int(M['m01']/M['m00'])
        pre_direction = direction
        if cy < pre_cy:     # This is so we can check to make sure we are strumming ina new direction
            direction = 0
        else:
            direction = 1
        strum = 0
        if direction != pre_direction:
            if abs(cy-pre_cy) > 10:     # Checks if we are strumming
                strum = 1
    return strum


def hand_0(binary0, frame0, width, height):     # Left hand
    cv2.imshow('Hand0 binary', binary0)

    """This finds the amount of fingers you are holding up
    based on the amount of defects in the the hull of a shape.
    This works really well as long as you have nothing but your
    hand in the image and you can clearly see the gaps between fingers.
    The code was originally done by DarshNaik and can be found here:
    https://github.com/DarshNaik/Hand-Detection-Finger-Counting
    it has been adapted to by me to work for this project"""
    contours, hierarchy = cv2.findContours(binary0, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    cnt = contours[0]
    M = cv2.moments(cnt)

    cx = int(M['m10']/M['m00'])

    epsilon = 0.0005*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    hull = cv2.convexHull(cnt)

    areahull = cv2.contourArea(hull)
    areacnt = cv2.contourArea(cnt)

    arearatio = ((areahull-areacnt)/areacnt)*100

    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)

    l = 0

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(approx[s][0])
        end = tuple(approx[e][0])
        far = tuple(approx[f][0])

        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        s = (a+b+c)/2
        ar = math.sqrt(s*(s-a)*(s-b)*(s-c))

        d=(2*ar)/a

        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        if angle <= 90 and d >10:
            l += 1
            cv2.circle(frame0, far, 3, [255,0,0], -1)

        cv2.line(frame0, start, end, [0,255,0], 2)

    l += 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    if l == 1:
        if areacnt < 2000:
            cv2.putText(frame0, 'Put hand in the box', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        else:
            if arearatio<  12:
                cv2.putText(frame0, '0', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

            else:
                cv2.putText(frame0,'1', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

    elif l == 2:
        cv2.putText(frame0, '2', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

    elif l == 3:

        if arearatio < 27:
            cv2.putText(frame0, '3', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame0, 'ok', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

    elif l == 4:
        cv2.putText(frame0, '4', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

    elif l == 5:
        cv2.putText(frame0, '5', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

    elif l == 6:
        cv2.putText(frame0, 'reposition', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

    else:
        cv2.putText(frame0, 'reposition', (10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

    segments = 4
    loop = 1
    string = l      # Replicates the strings on a bass guitar based on the amount of fingers held up
    while loop < segments:      # Finds the center of mass on the left side of the screen
        seg = loop*(width/segments)
        if cx < seg:
            play_note(string, loop)     # Plays a note based on wich string you are playing and wich part of the screen you hand is in
            loop = segments     # Makes sure we don't carry on repeating this once we have gotten to the target
        loop = loop + 1
    cv2.imshow('Hand0 frame', frame0)


def main(dev, data, timestamp):                 # I have not managed to get the code to work without this
    global keep_running
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