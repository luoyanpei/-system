import sys
import cv2
import random
import torch
from torchvision import transforms
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout, QLabel, QGroupBox, QDialog
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QFont
from PyQt5.QtCore import Qt, QTimer
from PIL import Image
import torch.nn.functional as F
from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtWidgets import QVBoxLayout
import mss
import os
import threading
import time
import torch
import numpy as np
from win32gui import FindWindow, GetWindowRect
from PyQt5.QtWidgets import QFileDialog
import MagSDK
from MagDevice import MagDevice
import condition


# # 通过 torch.hub.load 函数从指定路径加载Yolov5模型，
# 使用的是自定义模型（'custom'），模型文件为'yolov5s.pt'，设备为GPU设备编号为0，源为本地。
# 加载完成后，将模型赋值给变量 yolov5 。
yolov5 = torch.hub.load('D:\\JSU\\yolov5-master', 'custom', path='yolov5s.pt', device='0', source='local')
# 设置了 yolov5 的置信度阈值为0.3（ yolov5.conf = 0.3 ）和IoU阈值为0.4（ yolov5.iou = 0.4 ），用于筛选检测结果。
yolov5.conf = 0.3
yolov5.iou = 0.4
# 定义了一个颜色列表 COLORS ，其中包含了一些颜色的BGR值，用于在图像上绘制不同类别的目标框。
# COLORS = [
# (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255),
# (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0),
# (128, 128, 0), (0, 128, 0)]

# (128, 0, 128), (0, 128, 128), (0, 0, 128)
# 定义了一个标签列表 LABELS ，这里的类别还没有修改成对应yolov5s.pt对应的类别
LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant','stop sign']
# 创建了一个大小为(1280, 720, 3)的空图像 img_src ，用于显示检测结果。
img_src = np.zeros((1280, 720, 3), np.uint8)

# 总的来说，这段代码的作用是加载Yolov5模型，并设置模型的一些参数。同时定义了颜色列表和标签列表，以及一个空的图像用于显示检测结果。

def getLargestBox(bboxes, type):
    # 定义了一个变量 largest 并初始化为-1，用于记录当前最大的面积值。
    largest = -1
    # 定义了一个空的NumPy数组 bbox_largest ，用于存储最大的边界框。
    bbox_largest = np.array([])
    for bbox in bboxes:
        # 通过一个循环遍历每个边界框。对于每个边界框，首先判断其对应的类别是否在给定的 type 类型列表中。
        # 这里通过 LABELS[int(bbox[5])] 来获取边界框的类别标签，并判断其是否在 type 列表中。
        # 如果在，则继续执行下面的操作；如果不在，则跳过该边界框。
        if LABELS[int(bbox[5])] in type:
            # 获取当前边界框的左上角和右下角坐标。
            x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            # 计算其面积，使用的是边界框的宽度乘以高度。
            area = (x1 - x0) * (y1 - y0)
            # 将当前边界框的面积与 largest 进行比较。
            # 如果当前边界框的面积大于 largest ，则更新 largest 为当前面积值，
            # 并将当前边界框赋值给 bbox_largest 。
            if area > largest:
                largest = area
                bbox_largest = bbox
    return bbox_largest


def drawBBox(image, bboxes,color=None):#这段函数的目的是使用yolo网络检测目标生成bbox参数
    # 通过一个循环遍历每个边界框。
    for bbox in bboxes:
        # 对于每个边界框，首先获取其置信度 conf 和类别ID classID 。
        conf = bbox[4]
        tl = round(0.002 * max(image.shape[0:2])) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        classID = int(bbox[5])
        # 如果置信度大于 yolov5.conf （即设定的置信度阈值），则执行下面的操作。
        if conf > yolov5.conf:
            # 获取当前边界框的左上角和右下角坐标，并将其转换为整数类型
            x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            # 这里根据需求设置条件，演示只取10个颜色
            if classID >= 10:
                classID = 10
            #color = [int(c) for c in COLORS[classID]]
            tf = max(tl - 1, 1)  # font thickness
            # 使用 CV2.rectangle 函数在图像上绘制边界框，传入边界框的左上角坐标和右下角坐标，颜色值以及线宽（这里设定为3）。
            cv2.rectangle(image, (x0, y0), (x1, y1), color, tl, cv2.LINE_AA)
            text = "{}: {:.2f}".format(LABELS[classID], conf)
            # CV2.putText 函数在图像上绘制标签文本，传入标签文本内容、文本位置、字体、字体大小、颜色值以及文本厚度
            #cv2.rectangle(image, (max(0, x0), max(0, y0 - 5)), color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, text, (max(0, x0), max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            # 打印出边界框的坐标和标签文本
            print([x0, y0, x1, y1], text)
    return image


def getDetection(img):# 使用yolov5模型对图像进行目标检测# 将图像转换为RGB格式，并调整大小为1280
    bboxes = np.array(yolov5(img[:, :, ::-1], size=1280).xyxy[0].cpu())
    return bboxes




##这段代码主要作用是为了调用红外相机画面##
new_infrared_cb = None
new_visible_cb = None

new_frame_cond = condition.Condition()
# Called when a new visible frame arrived
def new_visible_frame(channel, data, width, height, pixel_format, user_data):
    # In this callback function, you can't do something which need a long time.
    # You should return as soon as possible.
    device = user_data
    # TBD
    # ...


# Called when a new infrared frame arrived
def new_infrared_frame(channel, camera_temperature, ffc_counter_down, camera_state, stream_type, user_data):
    # In this callback function, you can't do something which need a long time.
    # You should return as soon as possible.
    device = user_data
    new_frame_cond.notify()


def fix_temperature(device, temperature: int, pos: int):
    (fix_para, fix_opt) = device.GetFixPara()
    if fix_opt == MagSDK.FixSelection.FixSelectionDisabled:
        return temperature
    else:
        (x, y) = device.ConvertPos2XY(pos)
        return device.FixTemperature(temperature, fix_para.fEmissivity, x, y)


def login_camera(ip_addr, timeout):
    device = MagDevice()

    if not device.LinkCamera(ip_addr, timeout):
        print("Link camera fail")
        return None

    return device


def logout_camera(device):
    if device.IsLinked():
        device.DisLinkCamera()


def start_infrared(device):
    global new_infrared_cb
    new_infrared_cb = MagSDK.MAG_FRAMECALLBACK(new_infrared_frame)

    if not device.StartProcessImage_v2(new_infrared_cb, MagSDK.STREAM_TEMPERATURE, device):
        print("Start camera fail")
        return False

    # Set display style
    device.SetColorPalette(MagSDK.ColorPalette.IronBow)

    # # begin recording mgs
    # if not device.LocalStorageMgsRecord("test.mgs", 1):
    #     print("LocalStorageMgsRecord fail")
    return True


def stop_infrared(device):
    # # stop recording mgs
    # if device.IsLocalMgsRecording():
    #     device.LocalStorageMgsStop()

    if device.IsProcessingImage():
        device.StopProcessImage()


def start_visible(device, ip_addr, timeout):
    if not device.VisIsSupported():
        print("Visible is not supported")
        return True

    global new_visible_cb
    new_visible_cb = MagSDK.MAG_VISFRAMECALLBACK(new_visible_frame)

    if not device.VisPlay_v2("rtsp://%s/camera1" % ip_addr, MagSDK.enumVideoPixFormat.pixFmtRGB24,
                          new_visible_cb, device, 1, timeout):
        print("VisPlay fail")
        return False
    else:
        return True


def stop_visible(device):
    if device.VisIsPlaying():
        device.VisStop()


def save_something(device, timeout):
    if not device.SaveBMP(0, "test.bmp"):
        print("SaveBMP fail")

    if not device.SaveJpg("test.jpg", MagSDK.enumJpgExt.JpgExtMagnityDDT, timeout):
        print("SaveJpg fail")

    if not device.SaveDDT("test.ddt"):
        print("SaveBMP fail")

    if not device.VisSaveBMP_v2("test_vis.bmp"):
        print("VisSaveBMP_v2 fail")


def handle_data(device):
    # lock is needed if sdk function called out of new frame callback function
    device.Lock()

    # frame statistics
    frame_statistics = device.GetFrameStatisticalData()
    if frame_statistics:
        max_temperature = fix_temperature(device, frame_statistics.intMaxTemperature, frame_statistics.intPosMax)
        min_temperature = fix_temperature(device, frame_statistics.intMinTemperature, frame_statistics.intPosMin)
        print("[live-opencv] max: %d mC, min: %d mC" % (max_temperature, min_temperature))

    # Temperature at a specified position
    point_temp = device.GetTemperatureProbe(60, 60, 1)
    print("[live-opencv] point: %d mC" % point_temp)

    # Max & Min temperature for a rectangle region
    rect_info = device.GetRectTemperatureInfo(0, 0, 100, 100)
    if rect_info:
        rt_min_temperature = fix_temperature(device, rect_info[0], rect_info[3])
        rt_max_temperature = fix_temperature(device, rect_info[1], rect_info[4])
        print("[live-opencv] rect info: min: %d mC, max: %d mC" % (rt_min_temperature, rt_max_temperature))

    # Get rgb data
    ir_rgb_data = device.GetOutputVideoDataRGB24_v3(True)
    if not ir_rgb_data:
        ir_rgb_data = device.GetOutputBMPDataRGB24_v3(True)
        if not ir_rgb_data:
            print("GetOutputBMPDataRGB24_v2 fail")

    # Get temperature matrix
    # temp_matrix = device.GetTemperatureData(True)  # Slowly but higher precision
    # temp_matrix = device.GetTemperatureData_Raw(True)  # Quickly but lower precision
    # if not temp_matrix:
    #     print("GetTemperatureData fail")

    device.Unlock()

    # convert ir_data to opencv readable format, a new array generated
    if ir_rgb_data:
        came_info = device.GetCamInfo()
        ir_cv_data = cv2.flip(
            np.asarray(ir_rgb_data).reshape((came_info.intVideoHeight, came_info.intVideoWidth, 3)).astype(np.uint8), 0)
        #cv2.imshow("Infrared - press 'q' to quit", ir_cv_data)
        return ir_cv_data

    if device.VisIsPlaying():
        vis_rgb_data = device.VisGetData_v2()
        if vis_rgb_data:
            width = device.VisGetWidth()
            height = device.VisGetHeight()
            vis_cv_data = np.asarray(vis_rgb_data).reshape((height, width, 3)).astype(np.uint8)
            resized_visible_cv_data = cv2.resize(vis_cv_data, (640, 480), interpolation=cv2.INTER_AREA)
            #cv2.imshow("Visible - press 'q' to quit", resized_visible_cv_data)
            return resized_visible_cv_data
        else:
            print("VisGetData_v2 fail")

    # # convert temperature matrix to opencv matrix
    # if temp_matrix:
    #     camera_info = device.GetCamInfoEx()
    #     temp_array = np.asarray(temp_matrix).reshape(
    #         (camera_info.BaseInfo.intFPAHeight, camera_info.BaseInfo.intFPAWidth)).astype(np.int32)
    #     print("\ninfrared temperature matrix: ")
    #     print(temp_array)
##这段代码主要作用是为了调用红外相机画面##
class CameraThread(QThread):#这段函数的目的是调用系统中的摄像头单独用一个线程
    change_pixmap_signal = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_image = cv2.flip(rgb_image, 1)#对采集到的画面进行镜像翻转
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(convert_to_Qt_format)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

class InfraredCameraThread(QThread):#这段代码重新创建出了一个线程方便实时显示红外画面
    change_pixmap_signal = pyqtSignal(object)

    def run(self):
        device = login_camera("192.168.1.161", 30000)
        if not device:
            return
        start_infrared(device)
        start_visible(device, "192.168.1.161", 30000)
        while True:
            img = handle_data(device)
            height, width, channel = img.shape
            bytesPerLine = 3 * width
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.change_pixmap_signal.emit(pixmap)
            time.sleep(0.1)  # 控制更新频率，避免过于频繁


class MyWindow(QWidget):#窗口类函数
    def __init__(self):
        super().__init__()
        self.initUI()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_environment)
        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.infrared_camera_thread = InfraredCameraThread()  # 新增红外相机线程
        self.infrared_camera_thread.change_pixmap_signal.connect(self.update_infrared_image)
        #self.infrared_camera_thread.start()  # 启动红外相机线程

    def initUI(self):#UI界面定义
        grid = QGridLayout()
        self.setLayout(grid)
        self.setStyleSheet("background-color: #f0f0f0;")  # 设置窗口背景颜色
        self.camera_label = QLabel()#设置摄像头捕捉画面的位置
        grid.addWidget(self.camera_label, 3, 0, 4, 2)

        self.capture_label = QLabel()#设摄像头抓取画面的位置
        grid.addWidget(self.capture_label, 8, 0, 4, 2)

        self.yolo_label = QLabel()#设置yolo检测画面的位置
        grid.addWidget(self.yolo_label, 3, 5, 4, 2)

        self.infrared_cam_label = QLabel()#设置红外相机捕捉画面的位置
        grid.addWidget(self.infrared_cam_label, 8, 5, 4, 2)

        self.jsu_label = QLabel()  # 新增的 QLabel，用于显示图片
        pixmap = QPixmap('view.png')  # 加载图片
        scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio)  # 缩放图片
        self.jsu_label.setPixmap(scaled_pixmap)  # 设置图片到 QLabel
        grid.addWidget(self.jsu_label, 0, 2)  # 将 QLabel 放置在右上方



        buttons = ['打开摄像头', '打开红外相机', 'YOLO识别', '骨架识别', '图像抓取', '关闭']
        for i, button_name in enumerate(buttons):
            button = QPushButton(button_name)
            button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border: none;"
                                 "border-radius: 4px; padding: 10px 20px;}"
                                 "QPushButton:hover { background-color: #45a049; }"
                                 "QPushButton:pressed { background-color: #3e8e41; }")
            grid.addWidget(button, i // 2, i % 2)
            if button_name == '打开摄像头':
                button.clicked.connect(self.start_camera)
            elif button_name == '关闭':
                button.clicked.connect(self.close)
            elif button_name == '图像抓取':
                button.clicked.connect(self.catch_image)
            elif button_name == 'YOLO识别':
                button.clicked.connect(self.yolo_detec)
            elif button_name == '打开红外相机':
                button.clicked.connect(self.start_infrared_camera)   

        self.group_box = QGroupBox('生理状态参数')
        grid.addWidget(self.group_box, 1, 2, 8, 1)
        self.group_layout = QGridLayout()
        self.group_box.setLayout(self.group_layout)

        self.environment_temperature_label = QLabel()
        self.group_layout.addWidget(self.environment_temperature_label, 0, 0)

        self.environment_humidity_label = QLabel()
        self.group_layout.addWidget(self.environment_humidity_label, 1, 0)

        self.setWindowTitle('个人人体热舒适预测系统')
        self.setGeometry(100, 100, 800, 800)
        self.show()

    def start_camera(self):#此函数开启摄像头线程
        self.camera_thread.start()
        self.timer.start(2000)

    def update_image(self, image):#此函数用来更新摄像头参数
        self.camera_label.setPixmap(QPixmap.fromImage(image).scaled(400, 300, Qt.KeepAspectRatio))
    def start_infrared_camera(self):
            self.infrared_camera_thread.start()

    def update_infrared_image(self, pixmap):
        self.infrared_cam_label.setPixmap(pixmap)

    def update_environment(self):#此函数用来显示个人人体热舒适参数，未来有可能做参数融合用来预测人体的热舒适
        temperature = random.randint(20, 30)
        humidity = random.randint(40, 70)
        self.environment_temperature_label.setText(f'环境温度: {temperature}℃')
        self.environment_humidity_label.setText(f'环境湿度: {humidity}%')

    def catch_image(self):#此函数用来抓取图像
        self.capture_label.setPixmap(self.camera_label.pixmap())
        pixmap = self.camera_label.pixmap()
        if pixmap is not None:
        # 指定保存路径为当前目录下的 ui_save.png
            file_path = 'ui_save.png'
        # 保存图片
        pixmap.save(file_path)
    def yolo_detec(self):#此函数用来调用YOLO来进行识别
            img = cv2.imread('ui_save.png')
            # 调用 getDetection 函数对 img 进行目标检测，
            # 返回检测到的边界框信息存储在变量 bboxes 中。
            bboxes = getDetection(img)
            # 调用 drawBBox 函数在 img 上绘制检测到的边界框
            img = drawBBox(img, bboxes)
            height, width, channel = img.shape
            bytesPerLine = 3 * width
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qImg = QImage(img.data, width, height, bytesPerLine,QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.yolo_label.setPixmap(pixmap)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())