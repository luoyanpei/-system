import MagSDK
from MagDevice import MagDevice
import condition
import cv2
import numpy as np

new_infrared_cb = None
new_frame_cond = condition.Condition()


# Called when a new infrared frame arrived
def new_infrared_frame(channel, camera_temperature, ffc_counter_down, camera_state, stream_type, user_data):
    # In this callback function, you can't do something which need a long time.
    # You should return as soon as possible.
    print("[mgs-opencv] Infrared new frame")
    device = user_data
    new_frame_cond.notify()


def fix_temperature(device, temperature: int, pos: int):
    (fix_para, fix_opt) = device.GetFixPara()
    if fix_opt == MagSDK.FixSelection.FixSelectionDisabled:
        return temperature
    else:
        (x, y) = device.ConvertPos2XY(pos)
        return device.FixTemperature(temperature, fix_para.fEmissivity, x, y)


def handle_data(device):
    # lock & unlock is needed if out of new frame callback function
    device.Lock()

    # frame statistics
    frame_statistics = device.GetFrameStatisticalData()
    if frame_statistics:
        max_temperature = fix_temperature(device, frame_statistics.intMaxTemperature, frame_statistics.intPosMax)
        min_temperature = fix_temperature(device, frame_statistics.intMinTemperature, frame_statistics.intPosMin)
        print("[mgs-opencv] max: %d mC, min: %d mC" % (max_temperature, min_temperature))

    # Temperature at a specified position
    point_temp = device.GetTemperatureProbe(60, 60, 1)
    print("[mgs-opencv] point: %d mC" % point_temp)

    # Max & Min temperature for a rectangle region
    rect_info = device.GetRectTemperatureInfo(20, 20, 60, 80)
    if rect_info:
        rt_min_temperature = fix_temperature(device, rect_info[0], rect_info[3])
        rt_max_temperature = fix_temperature(device, rect_info[1], rect_info[4])
        print("[mgs-opencv] rect info: min: %d mC, max: %d mC" % (rt_min_temperature, rt_max_temperature))

    # Get rgb data
    rgb_data = device.GetOutputVideoDataRGB24_v3(True)
    if not rgb_data:
        rgb_data = device.GetOutputBMPDataRGB24_v3(True)
        if not rgb_data:
            print("GetOutputBMPDataRGB24_v2 fail")

    device.Unlock()

    # convert ir_data to opencv readable format, a new array generated
    if rgb_data:
        came_info: MagSDK.CamInfo = device.GetCamInfo()
        ir_cv_data = cv2.flip(
            np.asarray(rgb_data).reshape((came_info.intVideoHeight, came_info.intVideoWidth, 3)).astype(np.uint8), 0)

        cv2.imshow("Infrared - press 'q' to quit", ir_cv_data)


def test_mgs_file_opencv(file_name):
    device = MagDevice()

    global new_infrared_cb
    new_infrared_cb = MagSDK.MAG_FRAMECALLBACK(new_infrared_frame)

    # start
    if not device.LocalStorageMgsPlay(file_name, new_infrared_cb, device):
        print("LocalStorageMgsPlay fail")
        return False

    fps = device.GetCamInfo().intCurrentFPS

    while True:
        # pop a frame from mgs
        if not device.LocalStorageMgsPopFrame():
            print("End of stream")
            break

        # wait for frame ready
        if new_frame_cond.wait(5):
            handle_data(device)

        keycode = cv2.waitKey(1000//fps)
        if (keycode & 0xFF) == ord('q'):
            break

    # stop
    device.LocalStorageMgsStop()
