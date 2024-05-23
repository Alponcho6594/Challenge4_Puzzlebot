import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

class CVExample(Node):
    def __init__(self):
        super().__init__('cv_node')

        self.red_count = 0
        self.yellow_count = 0
        self.green_count = 0
        
        self.img = np.ndarray((720, 1280, 3))
        self.lower_red = np.array([136, 87, 111])
        self.upper_red = np.array([180, 255, 255])

        self.lower_yellow = np.array([100, 100, 20])
        self.upper_yellow = np.array([125, 255, 255])

        self.lower_green = np.array([40, 50, 50])
        self.upper_green = np.array([100, 255, 255])

        """""
        self.lower_red = np.array([136, 87, 111])
        self.upper_red = np.array([180, 255, 255])

        self.lower_yellow = np.array([100, 100, 20])
        self.upper_yellow = np.array([125, 255, 255])

        self.lower_green = np.array([12, 25, 25])
        self.upper_green = np.array([86, 255, 255])
        """""

        self.valid_img = False
        self.bridge = CvBridge()
        self.compr = CompressedImage()

        self.sub = self.create_subscription(CompressedImage, 'image', self.camera_callback, 10)
        #self.pub = self.create_publisher(Image, '/img_processing/color', 10)
        self.publisher = self.create_publisher(String, '/color', 10)
    
        self.msg = String()
        dt = 0.01
        self.timer = self.create_timer(dt, self.timer_callback)
        self.get_logger().info('CV Node started')

    def camera_callback(self, msg):
        try:
            self.compr.data = msg.data
            imgCompr = np.asarray(bytearray(self.compr.data))
            self.img = cv2.imdecode(imgCompr, cv2.IMREAD_COLOR)
            self.img=cv2.flip(self.img,0)

            self.img=cv2.flip(self.img,1)
            self.valid_img = True
        except:
            self.get_logger().info('Failed to get an image')

    def timer_callback(self):
        try:
            if self.valid_img:
                hsvFrame = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

                mask_red = cv2.inRange(hsvFrame, self.lower_red, self.upper_red)

                mask_yellow = cv2.inRange(hsvFrame, self.lower_yellow, self.upper_yellow)

                mask_green = cv2.inRange(hsvFrame, self.lower_green, self.upper_green)

                # Aplicar la máscara a la imagen original para cada color
                red_filtered = cv2.bitwise_and(self.img, self.img, mask=mask_red)
                yellow_filtered = cv2.bitwise_and(self.img, self.img, mask=mask_yellow)
                green_filtered = cv2.bitwise_and(self.img, self.img, mask=mask_green)

                # Convertir las imágenes filtradas a escala de grises
                red_gray = cv2.cvtColor(red_filtered, cv2.COLOR_BGR2GRAY)
                yellow_gray = cv2.cvtColor(yellow_filtered, cv2.COLOR_BGR2GRAY)
                green_gray = cv2.cvtColor(green_filtered, cv2.COLOR_BGR2GRAY)

                # Aplicar un desenfoque para reducir el ruido para cada color
                red_blur = cv2.medianBlur(red_gray, 5)
                yellow_blur = cv2.medianBlur(yellow_gray, 5)
                green_blur = cv2.medianBlur(green_gray, 5)

                # Detección de bordes utilizando Canny para cada color
                red_canny = cv2.Canny(red_blur, 50, 150)
                yellow_canny = cv2.Canny(yellow_blur, 50, 150)
                green_canny = cv2.Canny(green_blur, 50, 150)

                red_countours , _= cv2.findContours(red_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                yellow_countours, _ = cv2.findContours(yellow_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                green_countours, _ = cv2.findContours(green_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                for red_countour in red_countours:
                    red_area = cv2.contourArea(red_countour)
                    if red_area > 5: #distancia
                        cv2.drawContours(self.img, red_countour, -1, (0, 0, 255), cv2.LINE_AA) 
                        self.red_count += 1
                        self.yellow_count = 0
                        self.green_count = 0
                
                for yellow_countour in yellow_countours:
                    yellow_area = cv2.contourArea(yellow_countour)
                    if yellow_area > 5:
                        cv2.drawContours(self.img, yellow_countour, -1, (0, 255, 255), cv2.LINE_AA)
                        self.red_count = 0
                        self.yellow_count += 1
                        self.green_count = 0

                for green_countour in green_countours:
                    green_area = cv2.contourArea(green_countour)
                    if green_area > 5:
                        cv2.drawContours(self.img, green_countour, -1, (0, 255, 0), cv2.LINE_AA)
                        self.red_count = 0
                        self.yellow_count = 0
                        self.green_count += 1
                    
                if self.red_count >= 100:
                    self.msg.data = 'RED'  
                elif self.yellow_count >= 100:
                    self.msg.data = 'YELLOW'
                elif self.green_count >= 100:
                    self.msg.data = 'GREEN'
            
                self.publisher.publish(self.msg)

                #self.pub.publish(self.bridge.cv2_to_imgmsg(self.img, encoding='bgr8'))
                self.valid_img = False
        except:
            self.get_logger().info('Failed to process image')
        


def main(args=None):
    rclpy.init(args=args)
    cv_e = CVExample()
    rclpy.spin(cv_e)
    cv_e.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()