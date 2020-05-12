#! /usr/bin/env python 
# -*- coding:utf-8 -*-

from __future__ import print_function, division
import rospy
import numpy as np
import numpy
import tf
import math
import cv2
import time
from sensor_msgs.msg import Image, CompressedImage, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from numpy import linalg
from tf import transformations
from tf import TransformerROS
import tf2_ros
from geometry_msgs.msg import Twist, Vector3, Pose, Vector3Stamped
from ar_track_alvar_msgs.msg import AlvarMarker, AlvarMarkers
from nav_msgs.msg import Odometry 
from std_msgs.msg import Header
import cormodule_mod
import visao_module


bridge = CvBridge()


cv_image = None
media = []
centro = []
media2 = []
atraso = 1.5E9 # 1 segundo e meio. Em nanossegundos
dist = []
x = None
y = None
margem = 0.03
contador = 0
pula = 50
angulos = 0

h = True
f = False
g = False
x_inicial = 0
y_inicial = 0
x_final = 0
y_final = 0


area = 0.0 # Variavel com a area do maior contorno
area2 = 0.0

# Só usar se os relógios ROS da Raspberry e do Linux desktop estiverem sincronizados. 
# Descarta imagens que chegam atrasadas demais
check_delay = False 

resultados = [] # Criacao de uma variavel global para guardar os resultados vistos

x = 0
y = 0
z = 0 
id = 0

lines = [0,0]
pf = [0,0]
pfs = []
times = [0,0]

roxo = "#4c015b"
verde = "#006507"
azul = "#218dff"
amarelo = "#ffed36"

frame_final = None
frame = "camera_link"
# frame = "head_camera"  # DESCOMENTE para usar com webcam USB via roslaunch tag_tracking usbcam

tfl = 0

tf_buffer = tf2_ros.Buffer()

def recebe(msg):
	global x # O global impede a recriacao de uma variavel local, para podermos usar o x global ja'  declarado
	global y
	global z
	global id
	for marker in msg.markers:
		id = marker.id
		marcador = "ar_marker_" + str(id)

		print(tf_buffer.can_transform(frame, marcador, rospy.Time(0)))
		header = Header(frame_id=marcador)
		# Procura a transformacao em sistema de coordenadas entre a base do robo e o marcador numero 100
		# Note que para seu projeto 1 voce nao vai precisar de nada que tem abaixo, a 
		# Nao ser que queira levar angulos em conta
		trans = tf_buffer.lookup_transform(frame, marcador, rospy.Time(0))
		
		# Separa as translacoes das rotacoes
		x = trans.transform.translation.x
		y = trans.transform.translation.y
		z = trans.transform.translation.z
		# ATENCAO: tudo o que vem a seguir e'  so para calcular um angulo
		# Para medirmos o angulo entre marcador e robo vamos projetar o eixo Z do marcador (perpendicular) 
		# no eixo X do robo (que e'  a direcao para a frente)
		t = transformations.translation_matrix([x, y, z])
		# Encontra as rotacoes e cria uma matriz de rotacao a partir dos quaternions
		r = transformations.quaternion_matrix([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
		m = numpy.dot(r,t) # Criamos a matriz composta por translacoes e rotacoes
		z_marker = [0,0,1,0] # Sao 4 coordenadas porque e'  um vetor em coordenadas homogeneas
		v2 = numpy.dot(m, z_marker)
		v2_n = v2[0:-1] # Descartamos a ultima posicao
		n2 = v2_n/linalg.norm(v2_n) # Normalizamos o vetor
		x_robo = [1,0,0]
		cosa = numpy.dot(n2, x_robo) # Projecao do vetor normal ao marcador no x do robo
		angulo_marcador_robo = math.degrees(math.acos(cosa))

		# Terminamos
		print("id: {} x {} y {} z {} angulo {} ".format(id, x,y,z, angulo_marcador_robo))


def scaneou(dado):
    global dist
    # print("Faixa valida: ", dado.range_min , " - ", dado.range_max )
    # print("Leituras:")
    dists = []
    indices = [-5,-4,-3,-2,0,1,2,3,4,5]
    for e in indices:
        dists.append((np.array(dado.ranges).round(decimals=2))[e])
    
    dist = np.amin(dists)

def recebe_odometria(data):
    global x
    global y
    global contador
    global angulos

    x = data.pose.pose.position.x
    y = data.pose.pose.position.y

    quat = data.pose.pose.orientation
    lista = [quat.x, quat.y, quat.z, quat.w]
    angulos = np.degrees(transformations.euler_from_quaternion(lista))    

    if contador % pula == 0:
        print("Posicao (x,y)  ({:.2f} , {:.2f}) + angulo {:.2f}".format(x, y,angulos[2]))
    contador = contador + 1

# A função a seguir é chamada sempre que chega um novo frame
def roda_todo_frame(imagem):
    print("frame")
    global cv_image
    global media
    global centro
    global frame_final
    global pf
    global resultados
    global area
    global media2
    global centro2
    global area2

    now = rospy.get_rostime()
    imgtime = imagem.header.stamp
    lag = now-imgtime # calcula o lag
    delay = lag.nsecs
    # print("delay ", "{:.3f}".format(delay/1.0E9))
    if delay > atraso and check_delay==True:
        print("Descartando por causa do delay do frame:", delay)
        return 
    try:
        antes = time.clock()
        cv_image = bridge.compressed_imgmsg_to_cv2(imagem, "bgr8")
        pf, frame_pf = cormodule_mod.direction(cv_image,lines, pfs, times)
        # Note que os resultados já são guardados automaticamente na variável
        # chamada resultados
        centro, imagem, resultados =  visao_module.processa(cv_image)   
        media, centro, area =  cormodule_mod.identifica_cor(cv_image,azul)
        media2, centro2, area2 =  cormodule_mod.identifica_cor(cv_image,amarelo)   
        depois = time.clock() 
        
       # for r in resultados:
           # print(r) 
                        
           # pass

        depois = time.clock()
        frame_final = frame_pf
        # Desnecessário - Hough e MobileNet já abrem janelas
        #cv2.imshow("Camera", cv_image)
    except CvBridgeError as e:
        print('ex', e)

def anda_na_pista(pf, pfs, centro, erro):
    vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
    time = 5
    if pf != [0,0]:
        if len(pfs) > 1:
            pf = cormodule_mod.mediaPontos(pfs)
        t_lf = int(times[0].secs - times[1].secs)
        t_rl = int(times[1].secs - times[0].secs)
        print("Dif left/right",t_lf," Dif right/left",t_rl)
        if t_lf < time and t_rl < time:
            if centro[0] < pf[0]-erro:
                vel = Twist(Vector3(0.1,0,0), Vector3(0,0,-0.1))
                print("direita")
            elif centro[0] > pf[0]+erro:
                vel = Twist(Vector3(0.1,0,0), Vector3(0,0,0.1))
                print("esquerda")
            elif centro[0] > pf[0]-erro and centro[0] < pf[0]+erro:
                vel = Twist(Vector3(0.2,0,0), Vector3(0,0,0))
                print("reto")
        elif t_lf > time:
            print("corrigindo p/ direita")
            vel = Twist(Vector3(0.1,0,0), Vector3(0,0,-0.1))
        elif t_rl > time:
            print("corrigindo p/ esquerda")
            vel = Twist(Vector3(0.1,0,0), Vector3(0,0,0.1))
            
        
        velocidade_saida.publish(vel)
        rospy.sleep(0.5)

def segue_creeper(media, centro, dist):
    
    global f
    global g

    if len(media) != 0 and len(centro) != 0:
        # Calcula a diferença entre o centro do creeper e o centro da tela
        dif = int(media[0]) - int(centro[0])
        vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
        if dist > 0.8 or media[0] == 0:
            print(dist)
            if -30 < dif and dif < 30:
                vel = Twist(Vector3(0.1,0,0), Vector3(0,0,0))
                
            elif dif < -30:
                vel = Twist(Vector3(0,0,0), Vector3(0,0,0.1))

            elif dif > 30:
                vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.1))

        
        elif dist > 0.3 and dist <= 0.8:
            print("usando laser")
            vel = Twist(Vector3(0.05,0,0), Vector3(0,0,0))

        else:
            vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
            print("parou")
            f = False
            g = True
        

        velocidade_saida.publish(vel)
        rospy.sleep(0.5)

def volta_pra_pista(media2, centro2):
    print("Procurando Amarelo")
    vel = Twist(Vector3(0,0,0), Vector3(0,0,0.1))
    if len(media2) != 0 and len(centro2) != 0:
        # Calcula a diferença entre o centro do creeper e o centro da tela
        dif = int(media2[0]) - int(centro2[0])
        
        if area2 > 7000:
            print("Achei")
            if -30 < dif and dif < 30:
                vel = Twist(Vector3(0.2,0,0), Vector3(0,0,0))
                
            elif dif < -30:
                vel = Twist(Vector3(0,0,0), Vector3(0,0,0.1))

            elif dif > 30:
                vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.1))

    velocidade_saida.publish(vel)
    rospy.sleep(0.5)

        
        


if __name__=="__main__":
    rospy.init_node("cor")

    topico_imagem = "/camera/rgb/image_raw/compressed"

    recebedor = rospy.Subscriber(topico_imagem, CompressedImage, roda_todo_frame, queue_size=4, buff_size = 2**24)
    #recebedor = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, recebe) # Para recebermos notificacoes de que marcadores foram vistos
    recebe_scan = rospy.Subscriber("/scan", LaserScan, scaneou)

    print("Usando ", topico_imagem)

    velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)


    tolerancia = 25
    erro = 45

    # Exemplo de categoria de resultados
    # [('chair', 86.965459585189819, (90, 141), (177, 265))]

    try:
        # Inicializando - por default gira no sentido anti-horário
        
        
        while not rospy.is_shutdown():
            if h:
                if pf != [0,0]:
                    print("seguindo caminho")
                    anda_na_pista(pf, pfs, centro, erro)
                    if area > 10400:
                        h = False
                        f = True
                    

            elif f:
                print("Area: ",area)
                print("seguindo creeper")
                segue_creeper(media,centro,dist)

            elif g:
                print("Area 2: ", area2)
                volta_pra_pista(media2, centro)


            if frame_final   is not None:
                cv2.imshow("420",frame_final)
                cv2.waitKey(1)

            #velocidade_saida.publish(vel)
            #rospy.sleep(0.5)

    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")


