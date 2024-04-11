# yolov5屏幕实时检测

import mss
import cv2 as CV2
import os
import threading
import time
import torch
import numpy as np
from win32gui import FindWindow, GetWindowRect
# # 通过 torch.hub.load 函数从指定路径加载Yolov5模型，
# 使用的是自定义模型（'custom'），模型文件为'yolov5s.pt'，设备为GPU设备编号为0，源为本地。
# 加载完成后，将模型赋值给变量 yolov5 。
yolov5 = torch.hub.load('D:\\JSU\\yolov5-master', 'custom', path='yolov5s.pt', device='0', source='local')
# 设置了 yolov5 的置信度阈值为0.3（ yolov5.conf = 0.3 ）和IoU阈值为0.4（ yolov5.iou = 0.4 ），用于筛选检测结果。
yolov5.conf = 0.3
yolov5.iou = 0.4
# 定义了一个颜色列表 COLORS ，其中包含了一些颜色的BGR值，用于在图像上绘制不同类别的目标框。
COLORS = [
(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255),
(255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0),
(128, 128, 0), (0, 128, 0)]

# (128, 0, 128), (0, 128, 128), (0, 0, 128)
# 定义了一个标签列表 LABELS ，这里的类别还没有修改成对应yolov5s.pt对应的类别
LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant','stop sign']
# 创建了一个大小为(1280, 720, 3)的空图像 img_src ，用于显示检测结果。
img_src = np.zeros((1280, 720, 3), np.uint8)

# 总的来说，这段代码的作用是加载Yolov5模型，并设置模型的一些参数。同时定义了颜色列表和标签列表，以及一个空的图像用于显示检测结果。


def getScreenshot():
    # 这段代码用于获取屏幕截图。首先，通过注释掉的代码可以看出，它尝试根据窗口标题找到窗口的句柄，
    # 然后获取窗口的位置信息（左上角坐标和右下角坐标）。
    # 但是这部分代码被注释掉了，所以直接给定了一个固定的窗口位置信息（左上角坐标为(0, 0)，右下角坐标为(961, 1035)）。
    # id = FindWindow(None, "Windows.UI.Core.CoreWindow")
    # x0, y0, x1, y1 = GetWindowRect(id)
    x0, y0, x1, y1 = 0, 0, 961, 1035
    #  定义了两个变量 mtop 和 mbot ，分别表示从截图中去除的顶部和底部的像素行数。
    mtop, mbot = 30, 50
    # 创建了一个字典 monitor ，包含了屏幕截图的位置和大小信息，其中左上角坐标为 x0 和 y0 ，宽度为 x1 - x0 ，高度为 y1 - y0 。
    monitor = {"left": x0, "top": y0, "width": x1 - x0, "height": y1 - y0}
    # 使用 mss.mss().grab(monitor) 函数获取屏幕截图，并将结果存储在变量 img_src 中。
    # mss 是一个用于屏幕截图的库， grab 函数用于捕获指定位置和大小的屏幕区域。
    img_src = np.array(mss.mss().grab(monitor))
    # 通过 time.sleep(0.1) 函数等待一段时间，以确保截图操作完成。
    time.sleep(0.1)
    # 对截图进行处理。将 img_src 的通道数限制为3，即去除可能存在的第四个通道（alpha通道）。
    img_src = img_src[:, :, :3]
    # 根据设定的 mtop 和 mbot 值，从截图中去除对应的顶部和底部像素行
    img_src = img_src[mtop:-mbot]
    # 返回处理后的截图 img_src 以及窗口位置信息的列表。
    return img_src, [x0, y0, x1, y1, mtop, mbot]


def getMonitor():
    global img_src
    counter=0
    start_time = time.time()
    while True:
        # 通过 getScreenshot() 函数获取屏幕截图，并将结果赋值给 img_src 变量。
        # 函数返回的    第二个值   用下划线，表示不需要使用该值。
        img_src, _ = getScreenshot()
        counter += 1  # 计算帧数
        if (time.time() - start_time) != 0:  # 实时显示帧数
            print("原视频FPS: ", counter / (time.time() - start_time))
            counter = 0
            start_time = time.time()
            if CV2.waitKey(1) & 0xFF == ord("q"):
                break   

def yolov5Detect():
    # 通过 CV2.namedWindow 函数创建一个名为"Window Name"的窗口，
    # 并指定窗口的属性为 CV2.WINDOW_NORMAL ，即可调整窗口的大小。
    CV2.namedWindow("Window Name", CV2.WINDOW_NORMAL)
    # 使用 CV2.resizeWindow 函数将窗口大小设置为960x540像素
    CV2.resizeWindow("Window Name", 960, 540)
    # 使用 CV2.moveWindow 函数将窗口移动到屏幕上的指定位置（1560, 0）
    CV2.moveWindow("Window Name", 1560, 0)
    #设置画笔颜色，写在视频中
    start_time = time.time()
    counter = 0
    global img_src
    fourcc = CV2.VideoWriter_fourcc(*'mp4v') #保存格式为mp4格式
    out = CV2.VideoWriter('out.mp4',fourcc, 5, (1920,1080),True) 
    while True:
        # 通过 while True 创建一个无限循环。
        # 在每次循环中，首先将全局变量 img_src 的值复制给变量 img ，
        # 以确保获取到最新的屏幕截图。
        img = img_src.copy()
        # 调用 getDetection 函数对 img 进行目标检测，
        # 返回检测到的边界框信息存储在变量 bboxes 中。
        bboxes = getDetection(img)
        # 调用 drawBBox 函数在 img 上绘制检测到的边界框
        img = drawBBox(img, bboxes)
        # 通过 CV2.imshow 函数在窗口中显示绘制好边界框的图像
        counter += 1  # 计算帧数
        if (time.time() - start_time) != 0:  # 实时显示帧数
            CV2.putText(img, "FPS {0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (30, 50),
                        CV2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
            print("FPS: ", counter / (time.time() - start_time))
            counter = 0
            start_time = time.time()
            out.write(img)#保存视频
            CV2.imshow("Window Name", img)
        # 如果用户按下键盘上的"q"键，通过 CV2.waitKey 函数检测到按键事件，
        if CV2.waitKey(1) & 0xFF == ord("q"):
            # 就会销毁窗口并退出循环，否则继续下一次循环。
            CV2.destroyAllWindows()
            break


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


def drawBBox(image, bboxes):
    # 通过一个循环遍历每个边界框。
    for bbox in bboxes:
        # 对于每个边界框，首先获取其置信度 conf 和类别ID classID 。
        conf = bbox[4]
        classID = int(bbox[5])
        # 如果置信度大于 yolov5.conf （即设定的置信度阈值），则执行下面的操作。
        if conf > yolov5.conf:
            # 获取当前边界框的左上角和右下角坐标，并将其转换为整数类型
            x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            # 这里根据需求设置条件，演示只取10个颜色
            if classID >= 10:
                classID = 10
            color = [int(c) for c in COLORS[classID]]
            # 使用 CV2.rectangle 函数在图像上绘制边界框，传入边界框的左上角坐标和右下角坐标，颜色值以及线宽（这里设定为3）。
            CV2.rectangle(image, (x0, y0), (x1, y1), color, 3)     
            text = "{}: {:.2f}".format(LABELS[classID], conf)
            # CV2.putText 函数在图像上绘制标签文本，传入标签文本内容、文本位置、字体、字体大小、颜色值以及文本厚度
            CV2.putText(image, text, (max(0, x0), max(0, y0 - 5)), CV2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # 打印出边界框的坐标和标签文本
            print([x0, y0, x1, y1], text)
    return image


def getDetection(img):
    # 使用yolov5模型对图像进行目标检测
    # 将图像转换为RGB格式，并调整大小为1280
    bboxes = np.array(yolov5(img[:, :, ::-1], size=1280).xyxy[0].cpu())
    return bboxes


if __name__ == '__main__':
    # 创建了两个线程，分别执行 getMonitor 和 yolov5Detect 函数。
    t1 = threading.Thread(target=getMonitor, args=())
    t1.start()
    t2 = threading.Thread(target=yolov5Detect, args=())
    t2.start() 