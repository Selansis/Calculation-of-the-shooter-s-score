import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import pymysql
import random


# connnection = pymysql.connect(host="localhost", user="root", 
# passwd="", database="test")
# cursor = connnection.cursor()
    

# funckja zmniejszająca obraz / duża rodzielczosc zdjęcia z telefonu 

def resizing(scale,img,str): # percent of original size
    height = int(img.shape[0] * scale / 100)
    width = int(img.shape[1] * scale / 100)
    dim = (width, height) #wymiary
    resized = cv2.resize(img, dim) # funkcja resizująca
    cv2.imshow( str, resized) #pokazanie zresizowanego zdjęcia 
    cv2.imwrite(str+".png", img)
#funkcja znajdująca środek konturu

def centroid(contour):
    M = cv2.moments(contour)
    cx = int(round(M['m10']/M['m00']))
    cy = int(round(M['m01']/M['m00']))
    center = (cx, cy)
    return center




# transformacja perspektywy

# tutaj zczytywanie zdjęcia, oraz pobieranie jego parametrów czyli wysokosci i szerokosci
img = cv2.imread("img/test.jpg")
crapped = img.copy()
height = img.shape[0]
width = img.shape[1]
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 255, 255])
mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
mask_yellow = cv2.GaussianBlur(mask_yellow, (7,7), 0)

for i in range(0,height):
    for j in range(0,width):
        if mask_yellow[i,j] == 255:
            mask_yellow[i,j] = 0
        else:
            mask_yellow[i,j] = 1 

contours, _ = cv2.findContours(mask_yellow, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
biggest_cntr = None
biggest_area = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 1000:
        approx = cv2.approxPolyDP(contour, .03 * cv2.arcLength(contour, True), True)
        if area > biggest_area and len(approx) == 4:
            biggest_area = area
            biggest_cntr = approx
            

cv2.drawContours(crapped, [biggest_cntr], -1, (0, 255, 0), 3)

points = biggest_cntr.reshape(4, 2)
obj_points = np.zeros((4, 2), dtype="float32")
points_sum = points.sum(axis=1)
obj_points[0] = points[np.argmin(points_sum)]
obj_points[3] = points[np.argmax(points_sum)]
points_diff = np.diff(points, axis=1)
obj_points[1] = points[np.argmin(points_diff)]
obj_points[2] = points[np.argmax(points_diff)]
desired_points = np.float32([[0, 0], [3000, 0], [0, 3000], [3000, 3000]])

matrix = cv2.getPerspectiveTransform(obj_points, desired_points)
crapped = cv2.warpPerspective(crapped, matrix, (3000, 3000))
cont = crapped.copy()




img = crapped;
height = img.shape[0]
width = img.shape[1]
# zamiana obrazu rgb na hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)
v_mask = cv2.inRange(v, 0, 195)
v_mask = cv2.GaussianBlur(v_mask, (7,7), 0)
# znajdowanie największego okręgu  cv2.RETR_LIST zwraca liste konturów 
contours, _ = cv2.findContours(v_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
biggest_cntr = None
biggest_area = 0
wyn = img.copy()
for contour in contours:
    approx = cv2.approxPolyDP(contour, .03 * cv2.arcLength(contour, True), True)
    area = cv2.contourArea(contour)
    if len(approx)==8 and area > biggest_area:
        biggest_area = area
        biggest_cntr = contour
if biggest_cntr is not None:        
    cv2.drawContours(wyn, [biggest_cntr], -1, (0, 255, 0), 6)
    center = centroid(biggest_cntr)
    wyn = cv2.circle(wyn, center, 2, (155,155,0), -1)
    #obliczanie największego promienia
    biggest_radius = math.sqrt(biggest_area / math.pi)


# rysowanie dziur
hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
for i in range(0, 10): 
    radius = 50
    color =  (20, 100, 100)
    color1 = (0,255,255)
    x = random.randint(200, 2800)
    y = random.randint(200, 2800)
    cv2.circle(hsv1, (int(x), int(y)), radius, color, -1)
    cv2.circle(wyn, (int(x), int(y)), radius, color1, -1)


# oznaczanie dziur po kulach 
#h_mask = cv2.medianBlur(h_mask, 19);
# h_mask = cv2.inRange(h, 0, 30)
# h_mask = cv2.inRange(s, 0, 200)


kernel = np.ones((9, 9), np.uint8)

yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 255, 255])
dimg = cv2.inRange(hsv1, yellow_lower, yellow_upper)
for i in range(0,height):
    for j in range(0,width):
        if dimg[i,j] == 255:
            dimg[i,j] = 0
        else:
            dimg[i,j] = 1

dimg = cv2.dilate(dimg, kernel, iterations = 8)
dimg = cv2.erode(dimg, kernel, iterations = 2)
holes = []
contours, _= cv2.findContours(dimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    area1 = cv2.contourArea(contour)
    if area1 > 0 and area1 < 30000:
     c = centroid(contour)
     holes.append(c)
     wyn = cv2.circle(wyn, c, 50, (0,0,155), 8)

# resizing(30, wyn, "wyn2")


little_radius = 0.138461538*biggest_radius
remaining_radius = biggest_radius - little_radius
ring = remaining_radius / 8;
scores = []
#obliczanie odległości
for hole in holes:
    dx = hole[0] - center[0]
    dy = hole[1] - center[1]
    dist = math.sqrt(dx*dx + dy*dy) + 0.065*biggest_radius
    dist -= little_radius
    if dist < 0:
        scores.append(10)
    else:
        if 9 - int(dist / ring)>=0:
         scores.append(9 - int(dist / ring))
        else:
         scores.append(0)

font = cv2.FONT_HERSHEY_DUPLEX  
for a in range(len(holes)):
    con = (holes[a][0]+10, holes[a][1]-30)
    wyn = cv2.putText(wyn, str(scores[a]), con, font,  3, (0,0,255), 6, cv2.LINE_AA)
counth = len(scores)    
suma = sum(x for x in scores if x > 0)    
wyn = cv2.putText(wyn, str(suma), (60, 80), font,  3, (0, 0, 0), 5, cv2.LINE_AA)
#notatnik
# file = open("wyniki.txt", "a")
# file.write("Wynik = " + str(suma) + ", Liczba strzałów: "+ str(counth) + ', Średni wynik na strzał ' + str(suma/counth)+'\n')
# file.close()
# srednia = suma/counth
# # zapisywanie do bazy danych
# cursor.execute("INSERT INTO wyniki(wynik, liczba_strzalow, wynik_sredni) VALUES(%s, %s, %s);", (suma, counth, srednia))
# connnection.commit()
# connnection.close()
resizing(30, wyn, "oznaczone")
# plt.figure(1)
# plt.imshow(dimg); plt.show()
cv2.waitKey(0)