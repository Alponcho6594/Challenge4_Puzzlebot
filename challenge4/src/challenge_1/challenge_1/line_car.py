import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from time import sleep
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class LineFollower_Car(Node):
    def __init__(self):
        super().__init__('line_follower')

        self.valid_img = False
        self.bridge = CvBridge()
        self.speed_msg = Twist()
        self.color = 'GREEN'
        self.color_actual = String()

        self.hsvVals = [0, 0, 0, 146, 255, 88]
        self.sensors = 3
        self.threshold = 0.2
        self.width = 480
        self.height = 360

        self.p = 0.01
        self.d = 0.5
        self.prev_error = 0

        self.desired_pos = self.width // 2

        self.compr = CompressedImage()

        #self.image_sub = self.create_subscription(Image, 'video_source/raw', self.camera_callback, 10)
        self.image_sub = self.create_subscription(CompressedImage, 'image', self.camera_callback, 10)
        self.sub = self.create_subscription(String, 'color', self.listener_callback_color, 10)


        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        #self.pub = self.create_publisher(Image, '/img_processing/normal', 10)
        self.image_pub = self.create_publisher(Image, '/img_processing/treshold', 10)

        self.img = None
        self.imgThres = None
        dt = 0.2
        self.light = 1.0
        self.timer = self.create_timer(dt, self.timer_callback)
        self.get_logger().info('CV Node started')

    def timer_callback(self):
        try:
            if self.valid_img:
                self.speed_msg.linear.x = 0.0
                self.speed_msg.linear.y = 0.0
                self.speed_msg.linear.z = 0.0

                self.speed_msg.angular.x = 0.0
                self.speed_msg.angular.y = 0.0
                self.speed_msg.angular.z = 0.0
                if self.color == 'RED':
                    self.light = 0.0
                elif self.color == 'YELLOW':
                    self.light = 0.5
                else:
                    self.light = 1.0
                self.img = cv2.resize(self.cv_image, (self.width, self.height))
                self.img = self.img[190:360, 0:480]
                hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
                
                mask = hsv[:,:,2]

                satured = self.controller(mask, 290, 200)
                _, black_mask = cv2.threshold(satured, 80, 255, cv2.THRESH_BINARY_INV)

                # Asignar la máscara negra a `self.imgThres` para el procesamiento posterior
                self.imgThres = black_mask

                self.cx = self.getContours(self.imgThres)


                self.senOut = self.getSensorOutput(self.imgThres, self.sensors)
           
                self.sendCommands(self.senOut)

                if self.img is not None:
                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.img, encoding='bgr8'))
                #if self.imgThres is not None:
                #    self.pub.publish(self.bridge.cv2_to_imgmsg(self.imgThres))
                self.valid_img = False
        except:
                self.get_logger().info('Failed to process image')
   
    def controller(self, img, brightness=255, contrast=127): 
        brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255)) 

        contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127)) 

        if brightness != 0: 

            if brightness > 0: 

                shadow = brightness 

                max = 255

            else: 

                shadow = 0
                max = 255 + brightness 

            al_pha = (max - shadow) / 255
            ga_mma = shadow 

            cal = cv2.addWeighted(img, al_pha, 
                                img, 0, ga_mma) 

        else: 
            cal = img 

        if contrast != 0: 
            Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast)) 
            Gamma = 127 * (1 - Alpha) 

            cal = cv2.addWeighted(cal, Alpha, 
                                cal, 0, Gamma) 

        return cal 

    

    def getContours(self, imgThres):
        cx = 0
        contours, hierachy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) != 0:
            biggest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(biggest)
            cx = x + w // 2
            cy = y + h // 2

            cv2.drawContours(self.img, contours, -1, (255, 0, 255), 7)
            cv2.circle(self.img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        return cx

    def getSensorOutput(self, imgThres, sensors):
        imgs = np.hsplit(imgThres, sensors)
        senOut = []
        totalPixels = (self.img.shape[1] // sensors) * self.img.shape[0]
        for x, im in enumerate(imgs):
            pixelCount = cv2.countNonZero(im)
            if pixelCount > self.threshold * totalPixels:
                senOut.append(1)
            else:
                senOut.append(0)
        print(senOut)
        return senOut

    def sendCommands(self, senOut):
        error=self.desired_pos-self.cx
        pid=self.p*error+self.d*(error-self.prev_error)
        self.prev_error=error
        speed=float(np.clip(pid,-0.1,0.1))
        print("speed: ",speed)
        print("error: ", error)

  # rotation
        if senOut == [1, 0, 0]:
            print("Turn left")
            if self.color == 'RED':
                self.speed_msg.linear.x = 0.0 
                self.speed_msg.angular.z = 0.0
            elif self.color == 'YELLOW':
                self.speed_msg.linear.x = 0.0 
                self.speed_msg.angular.z = speed * 0.5
            else:
                self.speed_msg.linear.x = 0.0 
                self.speed_msg.angular.z = speed
        elif senOut == [1, 1, 0]:
            print("Turn left")
            if self.color == 'RED':
                self.speed_msg.linear.x = 0.0 
                self.speed_msg.angular.z = 0.0
            elif self.color == 'YELLOW':
                self.speed_msg.linear.x = 0.0 
                self.speed_msg.angular.z = speed * 0.5
            else:
                self.speed_msg.linear.x = 0.0 
                self.speed_msg.angular.z = speed
        elif senOut == [0, 1, 0]:
            if self.color == 'RED':
                self.speed_msg.linear.x = 0.0 
                self.speed_msg.angular.z = 0.0
            elif self.color == 'YELLOW':
                self.speed_msg.linear.x = 0.03 * 0.5
                self.speed_msg.angular.z = 0.0
            else:
                self.speed_msg.linear.x = 0.03
                self.speed_msg.angular.z = 0.0
            print("NO detecta centro")
        elif senOut == [0, 1, 1]:
            print("Turn right")
            if self.color == 'RED':
                self.speed_msg.linear.x = 0.0 
                self.speed_msg.angular.z = 0.0
            elif self.color == 'YELLOW':
                self.speed_msg.linear.x = 0.0 
                self.speed_msg.angular.z = speed * 0.5
            else:
                self.speed_msg.linear.x = 0.0 
                self.speed_msg.angular.z = speed
        elif senOut == [0, 0, 1]:
            print("Turn right")
            if self.color == 'RED':
                self.speed_msg.linear.x = 0.0 
                self.speed_msg.angular.z = 0.0
            elif self.color == 'YELLOW':
                self.speed_msg.linear.x = 0.0 
                self.speed_msg.angular.z = speed * 0.5
            else:
                self.speed_msg.linear.x = 0.0 
                self.speed_msg.angular.z = speed
        elif senOut == [0, 0, 0]:
            if self.color == 'RED':
                self.speed_msg.linear.x = 0.0 
                self.speed_msg.angular.z = 0.0
            elif self.color == 'YELLOW':
                self.speed_msg.linear.x = 0.03 * 0.5
                self.speed_msg.angular.z = 0.0
            else:
                self.speed_msg.linear.x = 0.03
                self.speed_msg.angular.z = 0.0 
        elif senOut == [1, 1, 1]:
            print("Stop, No se detecto ninguna linea")
        elif senOut == [1, 0, 1]:
            if self.color == 'RED':
                self.speed_msg.linear.x = 0.0 
                self.speed_msg.angular.z = 0.0
            elif self.color == 'YELLOW':
                self.speed_msg.linear.x = 0.03 * 0.5
                self.speed_msg.angular.z = 0.0
            else:
                self.speed_msg.linear.x = 0.03
                self.speed_msg.angular.z = 0.0
            print("Solo se detecto linea central")
        self.publisher.publish(self.speed_msg)
        sleep(0.1)

    def move(self, x, y, z, wx, wy, wz):
        self.speed_msg.linear.x = x
        self.speed_msg.linear.y = y
        self.speed_msg.linear.z = z

        self.speed_msg.angular.x = wx
        self.speed_msg.angular.y = wy
        self.speed_msg.angular.z = wz

        self.publisher.publish(self.speed_msg)

    def camera_callback(self, msg):
        try:
            self.compr.data = msg.data
            imgCompr = np.asarray(bytearray(self.compr.data))
            self.cv_image = cv2.imdecode(imgCompr, cv2.IMREAD_COLOR)
            self.cv_image=cv2.flip(self.cv_image,0)

            self.cv_image=cv2.flip(self.cv_image,1)
            self.valid_img = True
        except:
            self.get_logger().info('Failed to get an image')
    
    def listener_callback_color(self, msg):
        self.color = msg.data

def main(args=None):
    rclpy.init(args=args)
    l_f = LineFollower_Car()
    rclpy.spin(l_f)
    l_f.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
