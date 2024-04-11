import MagSDK
from MagDevice import MagDevice
import condition
import cv2
import numpy as np
import time

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
        cv2.imshow("Infrared - press 'q' to quit", ir_cv_data)

    if device.VisIsPlaying():
        vis_rgb_data = device.VisGetData_v2()
        if vis_rgb_data:
            width = device.VisGetWidth()
            height = device.VisGetHeight()
            vis_cv_data = np.asarray(vis_rgb_data).reshape((height, width, 3)).astype(np.uint8)
            resized_visible_cv_data = cv2.resize(vis_cv_data, (640, 480), interpolation=cv2.INTER_AREA)
            cv2.imshow("Visible - press 'q' to quit", resized_visible_cv_data)
        else:
            print("VisGetData_v2 fail")

    # # convert temperature matrix to opencv matrix
    # if temp_matrix:
    #     camera_info = device.GetCamInfoEx()
    #     temp_array = np.asarray(temp_matrix).reshape(
    #         (camera_info.BaseInfo.intFPAHeight, camera_info.BaseInfo.intFPAWidth)).astype(np.int32)
    #     print("\ninfrared temperature matrix: ")
    #     print(temp_array)


def test_live_camera_opencv(ip_addr, timeout):
    # 1. login camera
    device = login_camera(ip_addr, timeout)
    if not device:
        return False

    # 2. start sensors in camera
    start_infrared(device)
    start_visible(device, ip_addr, timeout)

    t = time.time()

    # run a while
    while time.time() - t < 5:
        if new_frame_cond.wait(100):
            handle_data(device)

        keycode = cv2.waitKey(5)
        if (keycode & 0xFF) == ord('q'):
            break

    # 3. save something, not a must
    save_something(device, timeout)

    # 4. stop sensors in camera
    stop_visible(device)
    stop_infrared(device)

    # 5. logout camera
    logout_camera(device)
    return True
