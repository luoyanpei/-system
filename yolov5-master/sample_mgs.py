import MagSDK
from MagDevice import MagDevice
import time

new_infrared_cb = None


# Called when a new infrared frame arrived
def new_infrared_frame(channel, camera_temperature, ffc_counter_down, camera_state, stream_type, user_data):
    print("[mgs] Infrared new frame")
    device = user_data

    # In this callback function, you can't do something which need a long time.
    # You should return as soon as possible.
    device = user_data
    handle_data(device)


def handle_data(device):
    # # lock & unlock is needed if sdk function called out of new frame callback function
    # device.Lock()

    # frame statistics
    frame_statistics = device.GetFrameStatisticalData()
    if frame_statistics:
        max_temperature = fix_temperature(device, frame_statistics.intMaxTemperature, frame_statistics.intPosMax)
        min_temperature = fix_temperature(device, frame_statistics.intMinTemperature, frame_statistics.intPosMin)
        print("[mgs] max: %d mC, min: %d mC" % (max_temperature, min_temperature))

    # Temperature at a specified position
    point_temp = device.GetTemperatureProbe(60, 60, 1)
    print("[mgs] point: %d mC" % point_temp)

    # Max & Min temperature for a rectangle region
    rect_info = device.GetRectTemperatureInfo(20, 20, 60, 80)
    if rect_info:
        rt_min_temperature = fix_temperature(device, rect_info[0], rect_info[3])
        rt_max_temperature = fix_temperature(device, rect_info[1], rect_info[4])
        print("[mgs] rect info: min: %d mC, max: %d mC" % (rt_min_temperature, rt_max_temperature))

    # Get rgb data
    rgb_data = device.GetOutputVideoDataRGB24_v3(True)
    if not rgb_data:
        rgb_data = device.GetOutputBMPDataRGB24_v3(True)
        if not rgb_data:
            print("GetOutputBMPDataRGB24_v2 fail")

    # # Get temperature matrix
    # temp_matrix = device.GetTemperatureData(True) # Slowly but higher precision
    # temp_matrix = device.GetTemperatureData_Raw(True) # Quickly but lower precision
    # if not temp_matrix:
    #     print("GetTemperatureData fail")

    # device.Unlock()


def fix_temperature(device, temperature: int, pos: int):
    (fix_para, fix_opt) = device.GetFixPara()
    if fix_opt == MagSDK.FixSelection.FixSelectionDisabled:
        return temperature
    else:
        (x, y) = device.ConvertPos2XY(pos)
        return device.FixTemperature(temperature, fix_para.fEmissivity, x, y)


def test_mgs_file(file_name):
    device = MagDevice()

    global new_infrared_cb
    new_infrared_cb = MagSDK.MAG_FRAMECALLBACK(new_infrared_frame)

    # start
    if not device.LocalStorageMgsPlay(file_name, new_infrared_cb, device):
        print("LocalStorageMgsPlay fail")
        return

    fps = device.GetCamInfo().intCurrentFPS

    while True:
        if not device.LocalStorageMgsPopFrame():
            print("End of stream")
            break

        time.sleep(1/fps)

    # stop
    device.LocalStorageMgsStop()
