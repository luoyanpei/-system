import MagSDK
from MagDevice import MagDevice


def handle_data(device):
    # frame statistics
    frame_statistics = device.GetFrameStatisticalData()
    if frame_statistics:
        max_temperature = fix_temperature(device, frame_statistics.intMaxTemperature, frame_statistics.intPosMax)
        min_temperature = fix_temperature(device, frame_statistics.intMinTemperature, frame_statistics.intPosMin)
        print("[ddt] max: %d mC, min: %d mC" % (max_temperature, min_temperature))

    # Temperature at a specified position
    point_temp = device.GetTemperatureProbe(60, 60, 1)
    print("[ddt] point: %d mC" % point_temp)

    # Max & Min temperature for a rectangle region
    rect_info = device.GetRectTemperatureInfo(20, 20, 60, 80)
    if rect_info:
        rt_min_temperature = fix_temperature(device, rect_info[0], rect_info[3])
        rt_max_temperature = fix_temperature(device, rect_info[1], rect_info[4])
        print("[ddt] rect info: min: %d mC, max: %d mC" % (rt_min_temperature, rt_max_temperature))

    # Get rgb data
    rgb_data = device.GetOutputVideoDataRGB24_v3(True)
    if not rgb_data:
        rgb_data = device.GetOutputBMPDataRGB24_v3(True)
        if not rgb_data:
            print("GetOutputBMPDataRGB24_v2 fail")

    # Get temperature matrix
    temp_matrix = device.GetTemperatureData(True) # Slowly but higher precision
    # temp_matrix = device.GetTemperatureData_Raw(True) # Quickly but lower precision
    if not temp_matrix:
        print("GetTemperatureData fail")


def fix_temperature(device, temperature: int, pos: int):
    (fix_para, fix_opt) = device.GetFixPara()
    if fix_opt == MagSDK.FixSelection.FixSelectionDisabled:
        return temperature
    else:
        (x, y) = device.ConvertPos2XY(pos)
        return device.FixTemperature(temperature, fix_para.fEmissivity, x, y)


def test_ddt_file(file_name):
    device = MagDevice()

    if not device.LoadDDT_v3(file_name):
        print("LoadDDT_v2 fail")
        return

    # print camera information
    camera_info = device.GetCamInfoEx()
    print("[ddt] infrared: "
          "serial_num:", camera_info.intCameraSN,
          ", name:", camera_info.BaseInfo.charName.decode("gbk"),
          ", type:", camera_info.BaseInfo.charType.decode("gbk"),
          ", resolution:", camera_info.BaseInfo.intFPAWidth, "x", camera_info.BaseInfo.intFPAHeight)

    # handle data
    handle_data(device)

    # Don't forget to destroy it
    device.Deinitialize()




