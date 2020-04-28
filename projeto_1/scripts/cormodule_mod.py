#! /usr/bin/env python
# -*- coding:utf-8 -*-


import rospy
import numpy as np
import tf
import math
import cv2
import time
from geometry_msgs.msg import Twist, Vector3, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import smach
import smach_ros



def convert_to_tuple(html_color):
    colors = html_color.split("#")[1]
    r = int(colors[0:2],16)
    g = int(colors[2:4],16)
    b = int(colors[4:],16)
    return (r,g,b)

def to_1px(tpl):
    img = np.zeros((1,1,3), dtype=np.uint8)
    img[0,0,0] = tpl[0]
    img[0,0,1] = tpl[1]
    img[0,0,2] = tpl[2]
    return img

def to_hsv(html_color):
    tupla = convert_to_tuple(html_color)
    hsv = cv2.cvtColor(to_1px(tupla), cv2.COLOR_RGB2HSV)
    return hsv[0][0]

def ranges(value):
    hsv = to_hsv(value)
    hsv2 = np.copy(hsv)
    hsv[0] = max(0, hsv[0]-10)
    hsv2[0] = min(180, hsv[0]+ 10)
    hsv[1:] = 50
    hsv2[1:] = 255
    return hsv, hsv2 

def escapePoint(p1,p2,q1,q2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    
    x3 = q1[0]
    y3 = q1[1]
    x4 = q2[0]
    y4 = q2[1]
    
    delta_x0 = x2 - x1
    delta_y0 = y2 - y1
    
    delta_x1 = x4 - x3
    delta_y1 = y4 - y3
    
    m0 = delta_y0/delta_x0
    h0 = y1 - m0*x1
    
    m1 = delta_y1/delta_x1
    h1 = y3 - m1*x3
    
    xi = int((h1-h0)/(m0-m1))
    yi = int(m0*xi +h0)
    
    ps = [xi,yi]
    
    return ps
    
def dist(x,y):
    d = math.sqrt(x**2+y**2)
    return d

def coefAang(a,b):
    x0 = a[0]
    y0 = a[1]
    x1 = b[0]
    y1 = b[1]
    delta_x = x1 - x0
    delta_y = y1 - y0
    if delta_x == 0:
        delta_x = 0.1
    return delta_y/delta_x
    
def mediaPontos(l):
    soma_x = 0
    soma_y = 0
    if len(l) < 1:
        return [0,0]

    else:
        for i in l:
            soma_x += i[0]
            soma_y += i[1]
        media_x = int(soma_x/len(l))
        media_y = int(soma_y/len(l))
        ponto_medio = [media_x,media_y]
        return ponto_medio

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def direction(frame):

    half_height = int(frame.shape[0]/2)
    frame_util = frame[half_height:][:][:]

    frame_hsv = cv2.cvtColor(frame_util, cv2.COLOR_BGR2HSV)

    hsv0, hsv1 = ranges("#FFFFFF")

    segmentado = cv2.inRange(frame_hsv, hsv0, hsv1)

    blur = cv2.GaussianBlur(segmentado,(5,5),0)
    bordas = auto_canny(blur)
    
    hough_img = bordas.copy()

    lines = cv2.HoughLinesP(hough_img, 20, math.pi/180.0, 100, np.array([]), 10, 5)

    a,b,c = lines.shape

    hough_img_rgb = cv2.cvtColor(hough_img, cv2.COLOR_GRAY2RGB)

    linhas = [0,0,0,0]

    fuga_points = []
    for i in range(a):
        xp = lines[i][0][0]
        yp = lines[i][0][1]
        xs = lines[i][0][2]
        ys = lines[i][0][3]

        p = [xp,yp,dist(xp,yp)]
        s = [xs,ys,dist(xp,yp)]
        
        if -3 < coefAang(p,s) < -0.5:
            if p[2] > linhas[2]:
                linhas[2] = p[2]
                linhas[0] = [p,s]
        elif 0.5 < coefAang(p,s) < 3:
            if p[2] > linhas[3]:
                linhas[3] = p[2]
                linhas[1] = [p,s]

        if linhas[0] == 0 or linhas[1] == 0:
            print("sem pf")
            cv2.imshow('video', frame_hsv)
            cv2.waitKey(1)

            return [0,0], frame
            
        else:
            pf = escapePoint(linhas[0][0],linhas[0][1],linhas[1][0],linhas[1][1])
            cv2.circle(hough_img_rgb,(pf[0],pf[1]),2,(0,255,0),2)

            print("era pra ir")

            return pf, hough_img_rgb

        





def identifica_cor(frame,cor):
    '''
    Segmenta o maior objeto cuja cor é parecida com cor_h (HUE da cor, no espaço HSV).
    '''

    # No OpenCV, o canal H vai de 0 até 179, logo cores similares ao 
    # vermelho puro (H=0) estão entre H=-8 e H=8. 
    # Precisamos dividir o inRange em duas partes para fazer a detecção 
    # do vermelho:
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsv0, hsv1 = ranges(cor)


    segmentado_cor = cv2.inRange(frame_hsv, hsv0, hsv1)
 
    # Note que a notacão do numpy encara as imagens como matriz, portanto o enderecamento é
    # linha, coluna ou (y,x)
    # Por isso na hora de montar a tupla com o centro precisamos inverter, porque 
    centro = (frame.shape[1]//2, frame.shape[0]//2)


    def cross(img_rgb, point, color, width,length):
        cv2.line(img_rgb, (point[0] - length/2, point[1]),  (point[0] + length/2, point[1]), color ,width, length)
        cv2.line(img_rgb, (point[0], point[1] - length/2), (point[0], point[1] + length/2),color ,width, length) 



    # A operação MORPH_CLOSE fecha todos os buracos na máscara menores 
    # que um quadrado 7x7. É muito útil para juntar vários 
    # pequenos contornos muito próximos em um só.
    segmentado_cor = cv2.morphologyEx(segmentado_cor,cv2.MORPH_CLOSE,np.ones((7, 7)))

    # Encontramos os contornos na máscara e selecionamos o de maior área
    #contornos, arvore = cv2.findContours(segmentado_cor.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)	
    contornos, arvore = cv2.findContours(segmentado_cor.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    maior_contorno = None
    maior_contorno_area = 0

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area > maior_contorno_area:
            maior_contorno = cnt
            maior_contorno_area = area

    # Encontramos o centro do contorno fazendo a média de todos seus pontos.
    if not maior_contorno is None :
        cv2.drawContours(frame, [maior_contorno], -1, [0, 0, 255], 5)
        maior_contorno = np.reshape(maior_contorno, (maior_contorno.shape[0], 2))
        media = maior_contorno.mean(axis=0)
        media = media.astype(np.int32)
        cv2.circle(frame, (media[0], media[1]), 5, [0, 255, 0])
        cross(frame, centro, [255,0,0], 1, 17)
    else:
        media = (0, 0)

    # Representa a area e o centro do maior contorno no frame
    # font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    # cv2.putText(frame,"{:d} {:d}".format(*media),(20,100), 1, 4,(255,255,255),2,cv2.LINE_AA)
    # cv2.putText(frame,"{:0.1f}".format(maior_contorno_area),(20,50), 1, 4,(255,255,255),2,cv2.LINE_AA)

   # cv2.imshow('video', frame)
    #cv2.imshow('seg', segmentado_cor)
    cv2.waitKey(1)

    return media, centro, maior_contorno_area
