import MagSDK
from MagDevice import MagDevice
from MagService import MagService
import time
import socket
import struct

new_infrared_cb = None
infrared_frame_counter = 0

new_visible_cb = None
visible_frame_counter = 0


# Called when a new visible frame arrived
def new_visible_frame(channel, data, width, height, pixel_format, user_data):
    global visible_frame_counter
    visible_frame_counter += 1
    print("[live] Visible new frame #%d" % visible_frame_counter)

    # In this callback function, you can't do something which need a long time.
    # You should return as soon as possible.
    device = user_data
    # TBD
    # ...


# Called when a new infrared frame arrived
def new_infrared_frame(channel, camera_temperature, ffc_counter_down, camera_state, stream_type, user_data):
    global infrared_frame_counter
    infrared_frame_counter += 1
    print("[live] Infrared new frame #%d" % ++infrared_frame_counter)

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
        print("[live] max: %d mC, min: %d mC" % (max_temperature, min_temperature))

    # Temperature at a specified position
    point_temp = device.GetTemperatureProbe(60, 60, 1)
    print("[live] point: %d mC" % point_temp)

    # Max & Min temperature for a rectangle region
    rect_info = device.GetRectTemperatureInfo(20, 20, 60, 80)
    if rect_info:
        rt_min_temperature = fix_temperature(device, rect_info[0], rect_info[3])
        rt_max_temperature = fix_temperature(device, rect_info[1], rect_info[4])
        print("[live] rect info: min: %d mC, max: %d mC" % (rt_min_temperature, rt_max_temperature))

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


def login_camera(ip_addr, timeout):
    device = MagDevice()

    if not device.LinkCamera(ip_addr, timeout):
        print("Link camera fail")
        return None

    return device


def logout_camera(device):
    if device.IsLinked():
        device.DisLinkCamera()


def config_camera(device, timeout):
    # Fixed information, readonly
    camera_info = device.GetCamInfoEx()
    print("infrared: "
          "serial_num:", camera_info.intCameraSN,
          ", name:", camera_info.BaseInfo.charName.decode("gbk"),
          ", type:", camera_info.BaseInfo.charType.decode("gbk"),
          ", resolution:", camera_info.BaseInfo.intFPAWidth, "x", camera_info.BaseInfo.intFPAHeight)

    # camera temperature, readonly
    camera_temperature = device.GetCameraTemperature(timeout)
    if not camera_temperature:
        print("GetCameraTemperature fail")
    else:
        print("camera temperature: " + str(camera_temperature))

    # PTZ information, readonly
    ptz_val = device.QueryPTZState(MagSDK.PTZQuery.PTZQueryZoomPosition, timeout)
    if ptz_val is None:
        print("QueryPTZState fail")
    else:
        print("ptz_query: " + str(ptz_val))

    # Output message from camera serial port
    if not device.SetSerialCmd("1111-1111-1111-1111"):
        print("SetSerialCmd fail")

    # Set Fix parameters
    (fix_para, fix_opt) = device.GetFixPara()
    # fix_para.fEmissivity = 0.98
    # fix_para.fTemp = 25.0
    # ... and so on
    device.SetFixPara(fix_para, fix_opt)

    # Get & Set parameters to camera
    # a and b is supported by all cameras
    # c is a newly added function, and is recommended now
    # a
    reg1 = device.ReadCameraRegContent(timeout)
    if not reg1:
        print("ReadCameraRegContent fail")
    print(reg1.charName.decode('gbk'))
    if not device.SetCameraRegContent(reg1):
        print("SeadCameraRegContent fail")

    # b
    reg2 = device.ReadCameraRegContent2(timeout)
    if not reg2:
        print("ReadCameraRegContent2 fail")
    print(reg2.charName.decode('gbk'))
    if not device.SetCameraRegContent2(reg2):
        print("SetCameraRegContent2 fail")

    # c
    reg3 = device.ReadCamRegs(timeout)
    if not reg3:
        print("ReadCamRegs fail")
    print(reg3.charNameUTF8.decode('utf-8'))
    if not device.SetCamRegs(reg3):
        print("ReadCamRegs fail")

    # Upload rois to camera, a will be covered by b
    # a. irregular rois
    rois = []
    roi = MagSDK.IrregularROI()
    roi.charROIName = b'123'
    roi.intRoiType = MagSDK.enumRoiType.RoiRect.value
    roi.x0 = 20
    roi.y0 = 20
    roi.x1 = 60
    roi.y1 = 80
    roi.intEmissivity = 100
    roi.intAlarmTemp = 3000000  # 3000 C
    roi.intTextPos = 1
    roi.intPtNumber = 2
    roi.Points[0].x = 20
    roi.Points[0].y = 20
    roi.Points[1].x = 60
    roi.Points[1].y = 80
    roi.intSamplePeriod = 1
    roi.dwReserved = 0
    roi.bNonvolatile = 0  # don't save when camera power off
    rois.append(roi)
    if not device.SetIrregularROIs(rois):
        print("SetIrregularROIs fail")

    # b. rectangle rois
    rois2 = []
    roi2 = MagSDK.RectROI()
    roi2.charROIName = b'123'
    roi2.x0 = 20
    roi2.y0 = 20
    roi2.x1 = 60
    roi2.y1 = 80
    roi2.intEmissivity = 100
    roi2.intAlarmTemp = 3000000
    roi2.intSamplePeriod = 1
    for i in range(8):
        roi2.dwReserved[i] = 0
    rois2.append(roi2)
    if not device.SetUserROIsEx(rois2):
        print("SetUserROIsEx fail")


def start_infrared(device):
    global new_infrared_cb
    new_infrared_cb = MagSDK.MAG_FRAMECALLBACK(new_infrared_frame)

    if not device.StartProcessImage_v2(new_infrared_cb, MagSDK.STREAM_TEMPERATURE, device):
        print("Start camera fail")
        return False

    # Set display style
    device.SetColorPalette(MagSDK.ColorPalette.IronBow)

    # begin recording mgs, not a must
    if not device.LocalStorageMgsRecord("test.mgs", 1):
        print("LocalStorageMgsRecord fail")

    return True


def stop_infrared(device):
    if device.IsLocalMgsRecording():
        device.LocalStorageMgsStop()

    if device.IsProcessingImage():
        device.StopProcessImage()


def start_visible(device, ip_addr, timeout):
    if not device.VisIsSupported():
        print("Visible is not supported")
        return True

    global new_visible_cb
    new_visible_cb = MagSDK.MAG_VISFRAMECALLBACK(new_visible_frame)

    if not device.VisPlay_v2("rtsp://%s/camera1" % ip_addr, MagSDK.enumVideoPixFormat.pixFmtARGB,
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


def enum_cameras():
    """
    Find all available cameras
    """
    service = MagService()
    if service.Initialize():
        print("service init success\n")
    else:
        print("service init failed\n")

    # this is asynchronous function, just send a broadcast message,
    # so we need wait a moment for a reply
    service.EnumCameras()
    time.sleep(0.1)

    enum_count, enum_list = service.GetTerminalList()
    if not enum_count:
        print("enum 0 cam")
        return
    else:
        print("enum {} camera(s)".format(enum_count))

    for i in range(enum_count):
        print("Name:" + enum_list[i].charTerminalName.decode("gbk"),
              "IP:" + socket.inet_ntoa(struct.pack("=I", enum_list[i].intTerminalIp)))

    service.Deinitialize()


def  test_live_camera(ip_addr, timeout):
    # 1. login camera
    device = login_camera(ip_addr, timeout)
    if not device:
        return False

    # 2. config camera, not a must
    config_camera(device, timeout)

    # 3. start sensors in camera
    start_infrared(device)
    start_visible(device, ip_addr, timeout)

    # run a while
    time.sleep(5)

    # 4. save something, not a must
    save_something(device, timeout)

    # 5. stop sensors in camera
    stop_visible(device)
    stop_infrared(device)

    # 6. logout camera
    logout_camera(device)
    return True
