import platform
import os
import ctypes
from ctypes import *
from enum import Enum

IS_WINDOWS = platform.system() == 'Windows' or 'cygwin' in platform.system().lower()
IS_UNIX = os.name == 'posix'

if IS_WINDOWS:
	from ctypes.wintypes import HWND
	ir_lib_name = "ThermoGroupSDKLib.dll"
	vis_lib_name = "MagVisible.dll"
else: 
	HWND = ctypes.c_void_p
	WINFUNCTYPE = CFUNCTYPE
	windll = cdll
	ir_lib_name = "libthermogroupsdk.so"
	vis_lib_name = "libmagvisible.so"

# Constants
CAMNAME_PROTOCOLLEN = 32
TYPENAMELEN = 8
MAX_RECT_ROI_NUM = 4
STREAM_TEMPERATURE = 2
STREAM_VIDEO = 4
STREAM_HYBRID =	STREAM_TEMPERATURE | STREAM_VIDEO

DEFAULT_TIMEOUT = 500
MAX_OBJ_NUM = 30

# enum
class ColorPalette(Enum):
	Gray0to255 = 0
	Gray255to0 = 1
	IronBow = 2
	RainBow = 3
	GlowBow = 4
	Autumn = 5
	Winter = 6
	HotMetal = 7
	Jet = 8
	RedSaturation = 9
	HighContrast = 10
	Medicine = 11

class PTZCmd(Enum):
	PTZStop = 0
	PTZRight = 1
	PTZLeft = 2
	PTZUp = 3
	PTZDown = 4
	PTZUpRight = 5
	PTZUpLeft = 6
	PTZDownRight = 7
	PTZDownLeft = 8
	PTZSetPreset = 9
	PTZCallPreset = 10
	PTZClearPreset = 11
	PTZSetAuxiliary = 12
	PTZClearAuxiliary = 13
	PTZZoomStop = 14
	PTZZoomIn = 15
	PTZZoomOut = 16
	PTZFocusStop = 17
	PTZFocusAuto = 18
	PTZFocusFar = 19
	PTZFocusNear = 20
	PTZFocusGoto = 21

class PTZQuery(Enum):
	PTZQueryPan = 0
	PTZQueryTilt = 1
	PTZQueryZoomPosition = 2
	PTZQueryZoomState = 3
	PTZQueryFocusPosition = 4
	PTZQueryFocusState = 5

class SDStorageFileType(Enum):
	SDFileBMP = 0
	SDFileJPG = 1
	SDFileDDT = 2
	SDFileMDT = 3
	SDFileMGT = 4
	SDFileAVI = 5
	SDFileMGS = 6

class FixSelection(Enum):
	FixSelectionDisabled = 0
	FixSelectionManualReflect = 1
	FixSelectionSceneAverageReflect = 2
	FixSelectionCameraTempReflect = 3

# NPU
class enumNpuImgSrc(Enum):
	NpuImgSrcIr = 0
	NpuImgSrcVis = 1
	NpuImgSrcUser = 2

class enumNpuImgPixelFormat(Enum):
	NpuImgPlanarRGB = 0
	NpuImgPlanarBGR = 1
	NpuImgPlanarYCbCr422 = 2
	NpuImgPlanarYCbCr420 = 3
	NpuImgPlanarYCrCb422 = 4
	NpuImgPlanarYCrCb420 = 5
	NpuImgGray = 6

class enumNpuImgScan(Enum):
	NpuImgScanTopDown = 0
	NpuImgScanBottomUp = 1

class enumTensorType(Enum):
	TensorFloat32 = 0
	TensorFloat16 = 1
	TensorInt8 = 2
	TensorUint8 = 3
	TensorInt16 = 4

class enumVideoPixFormat(Enum):
	pixfmtUnknown = -1
	pixFmtYUV420P = 0   #< planar YUV 4:2:0, 12bpp, (1 Cr & Cb sample per 2x2 Y samples)
	pixFmtRGB24 = 2     #< packed RGB 8:8:8, 24bpp, RGBRGB...
	pixFmtYUV422P = 4   #< planar YUV 4:2:2, 16bpp, (1 Cr & Cb sample per 2x1 Y samples)
	pixFmtARGB = 25     #< packed ARGB 8:8:8:8, 32bpp, ARGBARGB...
	pixFmtRGBA = 26     #< packed RGBA 8:8:8:8, 32bpp, RGBARGBA...
	pixFmtABGR = 27     #< packed ABGR 8:8:8:8, 32bpp, ABGRABGR...
	pixFmtBGRA = 28     #< packed BGRA 8:8:8:8, 32bpp, BGRABGRA...

class enumRoiType(Enum):
	RoiPoint = 0
	RoiLine = 1
	RoiRect = 2
	RoiCircle = 3
	RoiEllipse = 4
	RoiPolygon = 5
	RoiDelta = 6
	RoiDelta3 = 7

class enumJpgExt(Enum):
	JpgExtNone = 0
	JpgExtMagnityDDT = 1
	JpgExtDL664 = 2

class enumInputIo(Enum):
	IoFFC = 0
	IoCaptureMGT = 1
	IoCaptureBMP = 2
	IoCustom = 3
	IoPulseImage = 4

class enumDisplayMode(Enum):
	DmIrOnly = 0,
	DmVisOnly = 1
	DmPipCenter = 2,
	DmPipTopLeft = 3
	DmPipTopRight = 4
	DmPipBottomLeft = 5
	DmPipBottomRight = 6
	DmFused0p25 = 7
	DmFused0p50 = 8
	DmFused0p75 = 9
	DmFused0p100 = 10
	DmFusedContour1 = 11
	DmFusedContour2 = 12
	DmFusedContour3 = 13

class enumAnalogPlot(Enum):
	AnalogPlotNone = 0,
	AnalogPlotCenterCross = 1
	AnalogPlotMaxTemperature = 2
	AnalogPlotROI = 3

class enumHDMISpec(Enum):
	Hdmi480P50Hz = 0
	Hdmi480P60Hz = 1
	Hdmi576P50Hz = 2
	Hdmi576P60Hz = 3
	Hdmi720P50Hz = 4
	Hdmi720P60Hz = 5
	Hdmi1080P50Hz = 6
	Hdmi1080P60Hz = 7
	Hdmi1080P30Hz = 8

# struct
class TimeUnit(Structure):
	_fields_ = [("StartHour", c_byte), ("StartMinute", c_byte), ("EndHour", c_byte), ("EndMinute", c_byte)]

class CamInfo(Structure):
	_fields_ = [("intFPAWidth", c_uint), ("intFPAHeight", c_uint), ("pad", c_uint * 2), ("charName", c_char * CAMNAME_PROTOCOLLEN),
	("charType", c_char * TYPENAMELEN), ("intMaxFPS", c_uint), ("intCurrentFPS", c_uint), ("intVideoWidth", c_uint), ("intVideoHeight", c_uint) ]

class CamInfoEx(Structure):
	_fields_ = [("BaseInfo", CamInfo), ("intCameraSN", c_uint), ("intCamTemperature", c_int * 4), ("charLensName", c_char * 32), ("fFocalLength", c_float), 
	("intCaliBlackbodyRange", c_int * 2), ("dwReserved0", c_uint * 10), ("timeCurrent", c_longlong), ("dblLatitude", c_double), ("dblLongitude", c_double), 
	("fAltitude", c_float), ("intPaletteIndex", c_int), ("intTempUnit", c_int), ("fEmissivity", c_float), ("fEnvTemp", c_float), ("fTaoAtm", c_float),
	("fTaoFilter", c_float), ("fObjDist", c_float), ("bSubSectionEnlarge", c_int), ("intEnlargeX1", c_int), ("intEnlargeX2", c_int), 
	("byteEnlargeY1", c_uint), ("byteEnlargeY2", c_uint), ("intAutoEnlargeRange", c_uint), ("intBrightOffset", c_int), ("intContrastOffset", c_int), 
	("dwReserved1", c_uint * 8), ("intEmbededVisWidth", c_uint), ("intEmbededVisHeight", c_uint), ("dwReserved2", c_uint * 2), ("bSuppportNPU", c_uint),
	("dwReserved3", c_uint * 19)]

class CeRegContent(Structure):
	_fields_ = [("pad0", c_uint), ("charName", c_char * CAMNAME_PROTOCOLLEN), ("bUseStaticIp", c_int), ("dwStaticIp", c_uint), ("dwStaticNetMask", c_uint), 
	("bMulticastImg", c_int), ("dwMulticastIp", c_uint), ("pad1", c_uint * 16), ("dwSN", c_uint), ("pad2", c_uint * 2), ("dwStaticGateWay", c_uint), 
	("pad3", c_uint * 8), ("intCurrentLensIndex", c_uint), ("pad4", c_uint * 2), ("intFFCFrameTrigger", c_uint), ("intFFCTemperatureTrigger", c_uint),
	("pad5", c_uint), ("intAccResponse", c_uint), ("pad6", c_uint * 3), ("intInputIoFunction", c_uint), ("intPaletteIndex", c_int), 
	("bColorBar", c_int), ("bSubSectionEnlarge", c_int), ("intEnlargeX1", c_int), ("intEnlargeX2", c_int), ("byteEnlargeY1", c_uint), 
	("byteEnlargeY2", c_uint), ("intAutoEnlargeRange", c_uint), ("intAnalogPlot", c_uint), ("intAlarmTemp", c_int), 
	("intTVStandard", c_int), ("bCheckHeartBeat", c_int), ("bAlwaysAnalogOutput", c_int), ("pad8", c_uint * 3), 
	("EXLevel", c_int), ("dwPartnerVisibleIp", c_uint), ("intDDE", c_uint), ("pad9", c_uint), ("dwSerialBaudRate", c_uint), ("dwSerialFeature", c_uint)]

class CfgPara(Structure):
	_fields_ = [("intSize", c_uint), ("charName", c_char * CAMNAME_PROTOCOLLEN), ("dwReserved", c_uint * 12), ("dwSN", c_uint), ("dwFlip", c_uint), 
	("dwRotate", c_uint), ("dwStabilizer", c_uint), ("bUseStaticIp", c_int), ("dwStaticIp", c_uint), ("dwStaticNetMask", c_uint), ("dwStaticGateWay", c_uint), 
	("dwStaticDNS", c_uint), ("dwReserved3", c_uint * 4), ("bCheckHeartBeat", c_int), ("dwPartnerVisibleIp", c_uint), ("bCloudEnable", c_int), 
	("dwCloudServerIp", c_uint), ("dwReserved4", c_uint * 3), ("InputIoFunction", c_int), ("dwReserved5", c_uint * 1), ("intAlarmTemp", c_int), 
	("intMotionDetectSensitivity", c_uint), ("intMotionDetectX", c_uint * 2), ("intMotionDetectY", c_uint * 2), ("dwReserved6", c_uint * 2), 
	("intPeopleCountingLinePos", c_uint), ("intPeopleCountingObjSize", c_uint), ("intEncoder", c_uint), ("intBitRateKBPS", c_uint), ("intIFrame", c_uint), 
	("HdmiSpec", c_int), ("dwReserved7", c_uint * 4), ("intPaletteIndex", c_uint), ("AnalogPlot", c_int), ("intTVStandard", c_uint), 
	("intDetailRatio", c_uint), ("intEX", c_uint), ("bSeparateMode", c_int), ("intBrightOffset", c_int), ("intContrastOffset", c_int), 
	("bOSDTime", c_int), ("bOSDCamName", c_int), ("bOSDMotionDetect", c_int), ("dwReserved9", c_uint * 6), ("dwSerialBaudRate", c_uint), 
	("dwSerialByteSize", c_uint), ("dwSerialStopBits", c_uint), ("dwSerilParity", c_uint), ("dwReserved10", c_uint * 3), ("protoPTZ", c_int),
	("intPTZAddress", c_uint), ("dwReserved11", c_uint), ("alarmTime", TimeUnit * 7)]

class CamRegs(Structure):
	_fields_ = [("intSize", c_uint), ("charNameUTF8", c_char * CAMNAME_PROTOCOLLEN), ("dwReserved0", c_uint * 4), ("charCamType", c_char * TYPENAMELEN),
				("intSN", c_uint), ("intFirmwareVersionMS", c_uint), ("intFirmwareVersionLS", c_uint), ("intFPGAVersion", c_uint),
				("dwReserved1", c_uint * 4), ("dwReserved2", c_uint * 16), ("intAccResponse", c_uint), ("intDropResponse", c_uint),("bHybridFullFps", c_int),
				("dwReserved3", c_uint * 4), ("intMeasureRange", c_uint), ("intCurrentLensIndex", c_uint), ("intAlarmTempmC", c_int), ("intSunArea", c_uint),
				("intSmartAlgorithm", c_uint), ("intTFilterIndex", c_uint), ("bShowConvertedBodyTemp", c_int), ("bShowFahrenheit", c_int),
				("bSunProtectionEnable", c_int), ("intSunProtectionTime", c_uint), ("dwReserved4", c_uint * 2), ("intFFCFrameTrigger", c_uint),
				("intFFCTemperatureTrigger", c_uint), ("dwReserved5", c_uint * 4), ("dwSerialBaudRate", c_uint), ("dwSerialByteSize", c_uint),
				("dwSerialStopBits", c_uint), ("dwSerilParity", c_uint), ("protoPTZ", c_int), ("intPTZAddress", c_uint), ("dwReserved6", c_uint * 4),
				("InputIoFunction", c_int), ("intCustomPara1", c_int), ("intCustomPara2", c_int), ("dwReserved7", c_uint * 4), ("dwReserved8", c_uint * 20),
				("intNetCatlog", c_uint), ("bUseStaticIp", c_int), ("dwStaticIp", c_uint), ("dwStaticNetMask", c_uint), ("dwStaticGateWay", c_uint),
				("dwReserved9", c_uint * 4), ("intWifiMode", c_uint), ("charWifiSSID", c_char * 32), ("charWifiPwd", c_char * 16), ("bWifiStaticIp", c_int),
				("dwWifiStaticIp", c_uint), ("dwWifiStaticNetMask", c_uint), ("dwWifiStaticGateWay", c_uint), ("dwReserved10", c_uint * 4),
				("intCmdPort", c_uint), ("intImgPort", c_uint), ("intHttpPort", c_uint), ("intGVCPPort", c_uint), ("intGVSPPort", c_uint),
				("intModbusPort", c_uint), ("intOnvifPort", c_uint), ("intRtspPort", c_uint), ("dwReserved11", c_uint * 4), ("bNatEnable", c_int),
				("dwNatIp", c_uint), ("intRtspNatPort", c_uint), ("dwReserved12", c_uint * 4), ("dwStaticDNS", c_uint), ("dwStaticNTP", c_uint),
				("bCloudEnable", c_uint), ("dwCloudServerIp", c_uint), ("dwReserved13", c_uint * 4), ("dwPartnerVisibleIp", c_uint), ("dwPartnerDcsIp", c_uint),
				("intPartnerDcsPort", c_uint), ("intPartnerDcsIo", c_uint), ("dwReserved14", c_uint * 4), ("dwReserved15", c_uint * 20), ("DisplayMode", c_int),
				("dwFlip", c_uint), ("dwRotate", c_uint), ("dwStabilizer", c_uint), ("intAntiFlicker", c_uint), ("intPaletteIndex", c_uint), ("AnalogPlot", c_int),
				("intEX", c_uint), ("bAlwaysAnalogOutput", c_int), ("bColorBar", c_int), ("bOSDTime", c_int), ("bOSDCamName", c_int), ("dwReserved16", c_uint * 8),
				("intBrightOffset", c_int), ("intContrastOffset", c_int), ("intDetailRatio", c_uint), ("intAutoEnlargeRange", c_uint), ("bSeparateMode", c_int),
				("bIsothermal", c_int), ("dwReserved17", c_uint * 4), ("bSubSectionEnlarge", c_int), ("intEnlargeX1", c_int), ("intEnlargeX2", c_int),
				("byteEnlargeY1", c_uint), ("byteEnlargeY2", c_uint), ("dwReserved18", c_uint * 8), ("intEncoderProfile", c_uint), ("intBitRateKBPS", c_uint),
				("intRcMode", c_uint), ("intIFrame", c_uint), ("dwReserved19", c_uint * 4), ("HdmiSpec", c_int), ("intTVStandard", c_uint),
				("dwReserved20", c_uint * 4), ("dwReserved21", c_uint * 20)]

class FixPara(Structure):
	_fields_ = [("fDistance", c_float), ("fEmissivity", c_float), ("fTemp", c_float), ("fRH", c_float), ("fVisDistance", c_float), ("fRain", c_float), 
	("fSnow", c_float), ("fExtrapara1", c_float), ("fExtrapara2", c_float), ("fTaoAtm", c_float), ("fTaoFilter", c_float)]

class IpV4Cfg(Structure):
	_fields_ = [("MAC", c_byte * 6), ("pad", c_byte * 2), ("charCamName", c_char * CAMNAME_PROTOCOLLEN), ("dwSN", c_uint), ("bEnableDHCP", c_int),
	("IPAddress", c_uint), ("Subnetmask", c_uint), ("DefaultGateway", c_uint), ("Reserved", c_uint * 16)]

class OutputPara(Structure):
	_fields_ = [("dwFPAWidth", c_uint), ("dwFPAHeight", c_uint), ("dwBMPWidth", c_uint), ("dwBMPHeight", c_uint), ("dwColorBarWidth", c_uint), 
	("dwColorBarHeight", c_uint)]

class RectROI(Structure):
	_fields_ = [("charROIName", c_char * 32), ("x0", c_int), ("y0", c_int), ("x1", c_int), ("y1", c_int), ("intEmissivity", c_int), 
	("intAlarmTemp", c_int), ("dwDraw", c_uint), ("intSamplePeriod", c_uint), ("dwReserved", c_uint * 8)]

class RectROIReport(Structure):
	_fields_ = [("charROIName", c_char * 32), ("x0", c_int), ("y0", c_int), ("x1", c_int), ("y1", c_int), ("bAlarm", c_int), 
	("intMinTemp", c_int), ("intMaxTemp", c_int), ("intAveTemp", c_int), ("intMaxPos", c_int), ("intAlarmThreshold", c_int * 2), 
	("intTextPos", c_int), ("intRoiType", c_int), ("fEmissivity", c_float), ("dwReserved", c_uint * 2)]

class Point(Structure):
	_fields_ = [("x", c_int), ("y", c_int)]

class IrregularROI(Structure):
	_fields_ = [("charROIName", c_char * 32), ("intRoiType", c_int), ("x0", c_int), ("y0", c_int), ("x1", c_int), ("y1", c_int),
				("intEmissivity", c_int), ("intAlarmTemp", c_int), ("intTextPos", c_int), ("intSamplePeriod", c_uint),
				("intPtNumber", c_uint), ("Points", Point * 7), ("dwReserved", c_uint), ("bNonvolatile", c_int)]

class IrregularROIReport(Structure):
	_fields_ = [("BaseInfo", RectROIReport), ("intColor", c_uint), ("intPtNumber", c_uint), ("Points", Point * 8), ("dwReserved", c_uint * 8)]

class State(Structure):
	_fields_ = [("pad0", c_uint * 6), ("intMaxTemperature", c_int), ("intMinTemperature", c_int), ("intAveTemperature", c_int), ("intSTDTemperature", c_int), 
	("intPosMax", c_uint), ("intPosMin", c_uint), ("pad1", c_uint * 3), ("intAveNETDt", c_uint), ("pad2", c_uint * 5), ("intHistTemperature", c_uint * 256)]

class TerminalList(Structure):
	_fields_ = [("charTerminalName", c_char * 32), ("intVersion", c_uint), ("intTerminalIp", c_uint), ("intControllerIp", c_uint), 
	("charCameraMAC", c_byte * 6), ("pad0", c_char * 2), ("pad1", c_uint * 2)]

class UserROIs(Structure):
	_fields_ = [("intValidRectROI", c_uint), ("ROI", RectROI * MAX_RECT_ROI_NUM)]

class ShortRect(Structure):
	_fields_ = [("left", c_short), ("top", c_short), ("right", c_short), ("bottom", c_short)]

class ObjInfo(Structure):
	_fields_ = [("intObjId", c_int), ("shortObjType", c_short), ("shortConfidence", c_short), ("rectIr", ShortRect), ("shortSamplePointIrX", c_short), 
	("shortSamplePointIrY", c_short), ("rectVis", ShortRect), ("shortSamplePointVisX", c_short), ("shortSamplePointVisY", c_short), ("intTemperature", c_int), 
	("intBodyTemp", c_int), ("intAlarmLevel", c_short), ("reserved1", c_short), ("reserved2", c_int * 3)]

# NPU
class NPUSrc(Structure):
	_fields_ = [("src", c_uint), ("intCropLeft", c_uint), ("intCropRight", c_uint), ("intCropTop", c_uint), ("intCropBottom", c_uint), 
	("intNetworkWidth", c_uint), ("intNetworkHeight", c_uint), ("PixelFormat", c_uint), ("PixelScan", c_uint), 
	("intPostProcessingHandle", c_uint), ("InputTensorType", c_uint), ("OutputTensorType", c_uint), ("reserved", c_uint * 20)]
	# _fields_ = [("src", enumNpuImgSrc), ("intCropLeft", c_uint), ("intCropRight", c_uint), ("intCropTop", c_uint), ("intCropBottom", c_uint), 
	# ("intNetworkWidth", c_uint), ("intNetworkHeight", c_uint), ("PixelFormat", enumNpuImgPixelFormat), ("PixelScan", enumNpuImgScan), 
	# ("intPostProcessingHandle", c_uint), ("InputTensorType", enumTensorType), ("OutputTensorType", enumTensorType), ("reserved", c_uint * 20)]

MAX_TENSOR_NUM = 16
MAX_DIM_NUM = 16

class TensorAttr(Structure):
	_fields_ = [("dim", c_uint * MAX_DIM_NUM), ("dimNum", c_uint), ("fl", c_int), ("zp", c_uint), ("scale", c_float), ("reserved", c_uint * 8)]

class TensorAttrs(Structure):
	_fields_ = [("attrs", TensorAttr * MAX_TENSOR_NUM), ("inNum", c_int), ("outNum", c_int)]

# Infrared camera callback
MAG_FRAMECALLBACK = WINFUNCTYPE(None, c_uint, c_int, c_uint, c_uint, c_uint, py_object)
MAG_DDTCOMPRESSCALLBACK = WINFUNCTYPE(None, c_uint, c_void_p, c_uint, py_object)
MAG_RECONNECTCALLBACK = WINFUNCTYPE(None, c_uint, c_uint, c_int, py_object)
MAG_ROICALLBACK = WINFUNCTYPE(None, c_uint, POINTER(RectROIReport), c_uint, py_object)
MAG_IRREGULARROICALLBACK = WINFUNCTYPE(None, c_uint, POINTER(IrregularROIReport), c_uint, py_object)
MAG_SERIALCALLBACK = WINFUNCTYPE(None, c_uint, c_void_p, c_uint, py_object)

# AI callback
MAG_OBJRECOCALLBACK = WINFUNCTYPE(None, c_uint, POINTER(ObjInfo), c_uint, py_object)
MAG_NPUCALLBACK = WINFUNCTYPE(None, c_uint, c_uint, c_void_p, c_uint, POINTER(TensorAttrs), c_void_p, c_uint, py_object)

# Day camera callback
MAG_INITCALLBACK = WINFUNCTYPE(None, c_uint, c_int, c_int, py_object)
MAG_VISFRAMECALLBACK = WINFUNCTYPE(None, c_uint, c_char_p, c_int, c_int, c_int, py_object)
MAG_VISRECONNECTCALLBACK = WINFUNCTYPE(None, c_uint, c_int, py_object)

# import C++ dll
lib = windll.LoadLibrary(os.path.join(os.path.abspath('.'), ir_lib_name))
vis_lib_path = os.path.join(os.path.abspath('.'), vis_lib_name)
if os.path.exists(vis_lib_path):
	vis_lib = windll.LoadLibrary(vis_lib_path)

# import functions

def CompressDDT(pDstBuffer, intDstBufferSize, pSrcBuffer, intSrcBufferSize, intQuality):
	fun=lib.MAG_CompressDDT
	fun.argtypes = [c_void_p, c_uint, c_void_p, c_uint, c_uint]
	fun.restype = c_int
	return fun(pDstBuffer, intDstBufferSize, pSrcBuffer, intSrcBufferSize, intQuality)

def DecodeVideoFrame(intChannelIndex, pBuffer, intBufferLen):
	fun=lib.MAG_DecodeVideoFrame
	fun.argtypes = [c_uint, c_char_p, c_uint]
	fun.restype = c_uint
	return fun(intChannelIndex, pBuffer, intBufferLen)

def DeCompressDDT(pDstBuffer, intDstBufferSize, pSrcBuffer, intSrcBufferSize):
	fun=lib.MAG_DeCompressDDT
	fun.argtypes = [c_void_p, c_uint, c_void_p, c_uint]
	fun.restype = c_int
	return fun(pDstBuffer, intDstBufferSize, pSrcBuffer, intSrcBufferSize)

def DelChannel(intChannelIndex):
	fun=lib.MAG_DelChannel
	fun.argtypes = [c_uint]
	fun.restype = None
	return fun(intChannelIndex)

def DisLinkCamera(intChannelIndex):
	fun=lib.MAG_DisLinkCamera
	fun.argtypes = [c_uint]
	fun.restype = None
	return fun(intChannelIndex)

def EnableAutoReConnect(bEnable):
	fun=lib.MAG_EnableAutoReConnect
	fun.argtypes = [c_int]
	fun.restype = None
	return fun(bEnable)

def EnumCameras():
	fun=lib.MAG_EnumCameras
	fun.argtypes = []
	fun.restype = c_int
	return fun()

def ExecAutoFocus(intChannelIndex):
	fun=lib.MAG_ExecAutoFocus
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def FixTemperature(intChannelIndex, intT, fEmissivity, dwPosX, dwPosY):
	fun=lib.MAG_FixTemperature
	fun.argtypes = [c_uint, c_int, c_float, c_uint, c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, intT, fEmissivity, dwPosX, dwPosY)

def Free(intChannelIndex):
	fun=lib.MAG_Free
	fun.argtypes = [c_uint]
	fun.restype = None
	return fun(intChannelIndex)

def GetCamInfo(intChannelIndex, pInfo, intSize):
	fun=lib.MAG_GetCamInfo
	fun.argtypes = [c_uint, POINTER(CamInfo), c_uint]
	fun.restype = None
	return fun(intChannelIndex, pInfo, intSize)

def GetCamInfoEx(intChannelIndex, pInfo, intSize):
	fun=lib.MAG_GetCamInfoEx
	fun.argtypes = [c_uint, POINTER(CamInfoEx), c_uint]
	fun.restype = None
	return fun(intChannelIndex, pInfo, intSize)

def GetCameraTemperature(intChannelIndex, intT, intTimeoutMS):
	fun=lib.MAG_GetCameraTemperature
	fun.argtypes = [c_uint, c_int * 4, c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, intT, intTimeoutMS)

def GetCurrentOffset(intChannelIndex, charReferenceDDT, pOffsetX, pOffsetY):
	fun=lib.MAG_GetCurrentOffset
	if IS_WINDOWS:
		fun.argtypes = [c_uint, c_wchar_p, POINTER(c_int), POINTER(c_int)]
	else:
		fun.argtypes = [c_uint, c_char_p, POINTER(c_int), POINTER(c_int)]
	fun.restype = c_int
	return fun(intChannelIndex, charReferenceDDT, pOffsetX, pOffsetY)

def GetEllipseTemperatureInfo(intChannelIndex, x0, y0, x1, y1, info):
	fun=lib.MAG_GetEllipseTemperatureInfo
	fun.argtypes = [c_uint, c_uint, c_uint, c_uint, c_uint, c_int * 5]
	fun.restype = c_int
	return fun(intChannelIndex, x0, y0, x1, y1, info)

def GetEXLevel(intChannelIndex):
	fun=lib.MAG_GetEXLevel
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def GetFilteredRaw(intChannelIndex):
	fun=lib.MAG_GetFilteredRaw
	fun.argtypes = [c_uint]
	fun.restype = POINTER(c_ushort)
	return fun(intChannelIndex)

def GetFixPara(intChannelIndex, pBuffer):
	fun=lib.MAG_GetFixPara
	fun.argtypes = [c_uint, POINTER(FixPara)]
	fun.restype = c_int
	return fun(intChannelIndex, pBuffer)

def GetFrameStatisticalData(intChannelIndex):
	fun=lib.MAG_GetFrameStatisticalData
	fun.argtypes = [c_uint]
	fun.restype = POINTER(State)
	return fun(intChannelIndex)

def GetLineTemperatureInfo(intChannelIndex, buffer_, intBufferSizeByte, info, x0, y0, x1, y1):
	fun=lib.MAG_GetLineTemperatureInfo
	fun.argtypes = [c_uint, POINTER(c_int), c_uint, c_int * 3, c_uint, c_uint, c_uint, c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, buffer_, intBufferSizeByte, info, x0, y0, x1, y1)

def GetLocalIp():
	fun=lib.MAG_GetLocalIp
	fun.argtypes = []
	fun.restype = c_uint
	return fun()

def GetMulticastState(intTargetIp, intMulticastIp, intMulticastPort, intTimeoutMS):
	fun=lib.MAG_GetMulticastState
	fun.argtypes = [c_uint, POINTER(c_uint), POINTER(c_uint), c_uint]
	fun.restype = c_int
	return fun(intTargetIp, intMulticastIp, intMulticastPort, intTimeoutMS)

def GetOutputBMPdata(intChannelIndex, pData, pInfo):
	fun=lib.MAG_GetOutputBMPdata
	fun.argtypes = [c_uint, POINTER(POINTER(c_ubyte)), POINTER(POINTER(c_ubyte))] #third is BitmapInfo
	fun.restype = c_int
	return fun(intChannelIndex, pData, pInfo)

def GetOutputBMPdata_copy(intChannelIndex, pBmp, intBufferSize):
	fun=lib.MAG_GetOutputBMPdata_copy
	fun.argtypes = [c_uint, POINTER(POINTER(c_ubyte)), c_uint] 
	fun.restype = c_int	
	return fun(intChannelIndex, pBmp, intBufferSize)

def GetOutputBMPDataRGB24(intChannelIndex, pData, intBufferSize, bOrderBGR):
	fun=lib.MAG_GetOutputBMPdataRGB24
	fun.argtypes = [c_uint, POINTER(c_ubyte), c_uint, c_int]
	fun.restype = c_int
	return fun(intChannelIndex, pData, intBufferSize, bOrderBGR)

def GetOutputColorBardata(intChannelIndex, pData, pInfo):
	fun=lib.MAG_GetOutputColorBardata
	fun.argtypes = [c_uint, POINTER(POINTER(c_ubyte)), POINTER(POINTER(c_ubyte))] #third is BitmapInfo
	fun.restype = c_int
	return fun(intChannelIndex, pData, pInfo)

def GetOutputColorBardata_copy(intChannelIndex, pColorBar, intBufferSize):
	fun=lib.MAG_GetOutputColorBardata_copy
	fun.argtypes = [c_uint, POINTER(POINTER(c_ubyte)), c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, pColorBar, intBufferSize)

def GetOutputColorBardataRGB24(intChannelIndex, pData, intBufferSize, bOrderBGR):
	fun=lib.MAG_GetOutputColorBardataRGB24
	fun.argtypes = [c_uint, POINTER(c_ubyte), c_uint, c_int]
	fun.restype = c_int
	return fun(intChannelIndex, pData, intBufferSize, bOrderBGR)

def GetOutputVideoData(intChannelIndex, pData, pInfo):
	fun=lib.MAG_GetOutputVideoData
	fun.argtypes = [c_uint, POINTER(POINTER(c_ubyte)), POINTER(POINTER(c_ubyte))] #third is BitmapInfo
	fun.restype = c_int
	return fun(intChannelIndex, pData, pInfo)

def GetOutputVideoData_copy(intChannelIndex, pBmp, intBufferSize):
	fun=lib.MAG_GetOutputVideoData_copy
	fun.argtypes = [c_uint, POINTER(POINTER(c_ubyte)), c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, pBmp, intBufferSize)

def GetOutputVideoDataRGB24(intChannelIndex, pData, intBufferSize, bOrderBGR):
	fun=lib.MAG_GetOutputVideoDataRGB24
	fun.argtypes = [c_uint, POINTER(c_ubyte), c_uint, c_int]
	fun.restype = c_int
	return fun(intChannelIndex, pData, intBufferSize, bOrderBGR)

def GetOutputVideoYV12(intChannelIndex):
	fun=lib.MAG_GetOutputVideoYV12
	fun.argtypes = [c_uint]
	fun.restype = POINTER(c_ubyte)
	return fun(intChannelIndex)

def GetPCounter(intChannelIndex, intUpCounter, intDownCounter, intTimeoutMS):
	fun=lib.MAG_GetPCounter
	fun.argtypes = [c_uint, POINTER(c_uint), POINTER(c_uint), c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, intUpCounter, intDownCounter, intTimeoutMS)

def GetRecentHeartBeat(intChannelIndex):
	fun=lib.MAG_GetRecentHeartBeat
	fun.argtypes = [c_uint]
	fun.restype = c_uint
	return fun(intChannelIndex)

def GetLineTemperatureInfo2(intChannelIndex, x0, y0, x1, y1, info):
	fun=lib.MAG_GetLineTemperatureInfo2
	fun.argtypes = [c_uint, c_uint, c_uint, c_uint, c_uint, c_int * 5]
	fun.restype = c_int
	return fun(intChannelIndex, x0, y0, x1, y1, info)

def GetRectTemperatureInfo(intChannelIndex, x0, y0, x1, y1, info):
	fun=lib.MAG_GetRectTemperatureInfo
	fun.argtypes = [c_uint, c_uint, c_uint, c_uint, c_uint, c_int * 5]
	fun.restype = c_int
	return fun(intChannelIndex, x0, y0, x1, y1, info)

def GetRgnTemperatureInfo(intChannelIndex, Pos, intPosNumber, info):
	fun=lib.MAG_GetRgnTemperatureInfo
	fun.argtypes = [c_uint, POINTER(c_uint), c_uint, c_int * 5]
	fun.restype = c_int
	return fun(intChannelIndex, Pos, intPosNumber, info)

def GetTemperatureData(intChannelIndex, pData, intBufferSize, bEnableExtCorrect):
	fun=lib.MAG_GetTemperatureData
	fun.argtypes = [c_uint, POINTER(c_int), c_uint, c_int]
	fun.restype = c_int
	return fun(intChannelIndex, pData,intBufferSize, bEnableExtCorrect)

def GetTemperatureData_Raw(intChannelIndex, pData, intBufferSize, bEnableExtCorrect):
	fun=lib.MAG_GetTemperatureData_Raw
	fun.argtypes = [c_uint, POINTER(c_int), c_uint, c_int]
	fun.restype = c_int
	return fun(intChannelIndex, pData, intBufferSize, bEnableExtCorrect)

def GetTemperatureProbe(intChannelIndex, dwPosX, dwPosY, intSize):	
	fun=lib.MAG_GetTemperatureProbe
	fun.argtypes = [c_uint, c_uint, c_uint, c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, dwPosX, dwPosY, intSize)

def GetTerminalList(pList, dwBufferSize):
	fun=lib.MAG_GetTerminalList
	fun.argtypes = [POINTER(TerminalList), c_uint]
	fun.restype = c_uint
	return fun(pList, dwBufferSize)

def GetVideoPPS(intChannelIndex, pBuffer, intBufferLen):
	fun=lib.MAG_GetVideoPPS
	fun.argtypes = [c_uint, c_char_p, c_uint]
	fun.restype = c_uint
	return fun(intChannelIndex, pBuffer, intBufferLen)

def GetVideoSPS(intChannelIndex, pBuffer, intBufferLen):
	fun=lib.MAG_GetVideoSPS
	fun.argtypes = [c_uint, c_char_p, c_uint]
	fun.restype = c_uint
	return fun(intChannelIndex, pBuffer, intBufferLen)

def GetVideoVCL(intChannelIndex, pBuffer, intBufferLen):
	fun=lib.MAG_GetVideoVCL
	fun.argtypes = [c_uint, c_char_p, c_uint]
	fun.restype = c_uint
	return fun(intChannelIndex, pBuffer, intBufferLen)

def GrabAndAsyncCompressDDT(intChannelIndex, pUserData):
	fun=lib.MAG_GrabAndAsyncCompressDDT
	fun.argtypes = [c_uint, c_void_p]
	fun.restype = c_int
	return fun(intChannelIndex, pUserData)

def Initialize(intChannelIndex, hWndMsg):
	fun=lib.MAG_Initialize
	fun.argtypes = [c_uint, HWND]
	fun.restype = c_int
	return fun(intChannelIndex, hWndMsg)

def IpCfg_EnumCamera():
	fun=lib.MAG_IpCfg_EnumCamera
	fun.argtypes = []
	fun.restype = None
	return fun()

def IpCfg_GetTerminalList(pList, dwBufferSize):
	fun=lib.MAG_IpCfg_GetTerminalList
	fun.argtypes = [POINTER(IpV4Cfg), c_uint]
	fun.restype = c_uint
	return fun(pList, dwBufferSize)

def IpCfg_SetIp(pPara):
	fun=lib.MAG_IpCfg_SetIp
	fun.argtypes = [POINTER(IpV4Cfg)]
	fun.restype = c_int
	return fun(pPara)

def IsChannelAvailable(intChannelIndex):
	fun=lib.MAG_IsChannelAvailable
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

if IS_WINDOWS:
	def IsDHCPServerRunning():
		fun=lib.MAG_IsDHCPServerRunning
		fun.argtypes = []
		fun.restype = c_int
		return fun()
else:
	def IsDHCPServerRunning():
		return False

def IsInitialized(intChannelIndex):
	fun=lib.MAG_IsInitialized
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

if IS_WINDOWS:
	def IsLanConnected():
		fun=lib.MAG_IsLanConnected
		fun.argtypes = []
		fun.restype = c_int
		return fun()
else:
	def IsLanConnected():
		return True

def IsLinked(intChannelIndex):
	fun=lib.MAG_IsLinked
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def IsListening(intChannelIndex):
	fun=lib.MAG_IsListening
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def IsProcessingImage(intChannelIndex):
	fun=lib.MAG_IsProcessingImage
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def IsUsingTemperatureMask(intChannelIndex):
	fun=lib.MAG_IsUsingTemperatureMask
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def IsUsingStaticIp():
	fun=lib.MAG_IsUsingStaticIp
	fun.argtypes = []
	fun.restype = c_int
	return fun()

def LinkCamera(intChannelIndex, intIP, intTimeoutMS):
	fun=lib.MAG_LinkCamera
	fun.argtypes = [c_uint, c_uint, c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, intIP, intTimeoutMS)

def LinkCameraEx(intChannelIndex, IndexOrIP, shortCmdPort, shortImgPort, charCloudUser, charCloudPwd, intCamSN, charCamUser, charCamPwd, intTimeoutMS):
	if IS_WINDOWS:
		fun=lib.MAG_LinkCameraEx
	else:
		fun=lib.MAG_LinkCameraEx2
	fun.argtypes = [c_uint, c_uint, c_ushort, c_ushort, c_char_p, c_char_p, c_uint, c_char_p, c_char_p, c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, IndexOrIP, shortCmdPort, shortImgPort, charCloudUser, charCloudPwd, intCamSN, charCamUser, charCamPwd, intTimeoutMS)

def ListenTo(intChannelIndex, intTargetIp):
	fun=lib.MAG_ListenTo
	fun.argtypes = [c_uint, c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, intTargetIp)

def LoadBufferedDDT(intChannelIndex, paraOut, pBuffer, intBufferSize, funcFrame, pUserData):
	fun=lib.MAG_LoadBufferedDDT
	fun.argtypes = [c_uint, POINTER(OutputPara), c_void_p, c_uint, MAG_FRAMECALLBACK, py_object]
	fun.restype = c_int
	return fun(intChannelIndex, paraOut, pBuffer, intBufferSize, funcFrame, pUserData)

def LoadDDT(intChannelIndex, paraOut, charFilename, funcFrame, pUserData):
	fun=lib.MAG_LoadDDT
	if IS_WINDOWS:
		fun.argtypes = [c_uint, POINTER(OutputPara), c_wchar_p, MAG_FRAMECALLBACK, py_object]
	else:
		fun.argtypes = [c_uint, POINTER(OutputPara), c_char_p, MAG_FRAMECALLBACK, py_object]
	fun.restype = c_int
	return fun(intChannelIndex, paraOut, charFilename, funcFrame, pUserData)

def LocalStorageAviStart(intChannelIndex, charFileName, intSamplePeriod):
	fun=lib.MAG_LocalStorageAviStart
	if IS_WINDOWS:
		fun.argtypes = [c_uint, c_wchar_p, c_uint]
	else:
		fun.argtypes = [c_uint, c_char_p, c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, charFileName, intSamplePeriod)

def LocalStorageAviStop(intChannelIndex):
	fun=lib.MAG_LocalStorageAviStop
	fun.argtypes = [c_uint]
	fun.restype = None
	return fun(intChannelIndex)

def LocalStorageMgsRecord(intChannelIndex, charFileName, intSamplePeriod):
	fun = lib.MAG_LocalStorageMgsRecord
	if IS_WINDOWS:
		fun.argtypes = [c_uint, c_wchar_p, c_uint]
	else:
		fun.argtypes = [c_uint, c_char_p, c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, charFileName, intSamplePeriod)

def LocalStorageMgsPlay(intChannelIndex, charFilename, funcFrame, pUserData):
	fun = lib.MAG_LocalStorageMgsPlay
	if IS_WINDOWS:
		fun.argtypes = [c_uint, c_wchar_p, MAG_FRAMECALLBACK, py_object]
	else:
		fun.argtypes = [c_uint, c_char_p, MAG_FRAMECALLBACK, py_object]
	fun.restype = c_int
	return fun(intChannelIndex, charFilename, funcFrame, pUserData)

def LocalStorageMgsPopFrame(intChannelIndex):
	fun = lib.MAG_LocalStorageMgsPopFrame
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def LocalStorageMgsSeekFrame(intChannelIndex, intFrameIndex):
	fun = lib.MAG_LocalStorageMgsSeekFrame
	fun.argtypes = [c_uint, c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, intFrameIndex)

def LocalStorageMgsStop(intChannelIndex):
	fun = lib.MAG_LocalStorageMgsStop
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)


def LockFrame(intChannelIndex):
	fun=lib.MAG_LockFrame
	fun.argtypes = [c_uint]
	fun.restype = None
	return fun(intChannelIndex)

def MoveLens(intChannelIndex, bDirectionFar, intMs):
	fun=lib.MAG_MoveLens
	fun.argtypes = [c_uint, c_int, c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, bDirectionFar, intMs)

def NewChannel(intChannelIndex):
	fun=lib.MAG_NewChannel
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def QueryPTZState(intChannelIndex, query, intValue, intTimeoutMS):
	fun=lib.MAG_QueryPTZState
	fun.argtypes = [c_uint, c_int, POINTER(c_int), c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, query, intValue, intTimeoutMS)

def ReadCameraRegContent(intChannelIndex, pContent, intTimeoutMS, bReadDefaultValue):
	fun=lib.MAG_ReadCameraRegContent
	fun.argtypes = [c_uint, POINTER(CeRegContent), c_uint, c_int]
	fun.restype = c_int
	return fun(intChannelIndex, pContent, intTimeoutMS, bReadDefaultValue)

def ReadCameraRegContent2(intChannelIndex, pContent, intTimeoutMS):
	fun=lib.MAG_ReadCameraRegContent2
	fun.argtypes = [c_uint, POINTER(CfgPara), c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, pContent, intTimeoutMS)

def ReadCamRegs(intChannelIndex, pContent, intTimeoutMS):
	fun=lib.MAG_ReadCamRegs
	fun.argtypes = [c_uint, POINTER(CamRegs), c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, pContent, intTimeoutMS)

def ResetCamera(intChannelIndex):
	fun=lib.MAG_ResetCamera
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def SaveBMP(intChannelIndex, dwIndex, charFilename):
	fun=lib.MAG_SaveBMP
	if IS_WINDOWS:
		fun.argtypes = [c_uint, c_uint, c_wchar_p]
	else:
		fun.argtypes = [c_uint, c_uint, c_char_p]
	fun.restype = c_int
	return fun(intChannelIndex, dwIndex, charFilename)

def SaveJPG(intChannelIndex, charFilename, ext, intTimeoutMS):
	fun=lib.MAG_SaveJPG
	if IS_WINDOWS:
		fun.argtypes = [c_uint, c_wchar_p, c_int, c_uint]
	else:
		fun.argtypes = [c_uint, c_char_p, c_int, c_uint]
	fun.restype = c_int
	print(intChannelIndex, charFilename, ext, intTimeoutMS)
	return fun(intChannelIndex, charFilename, ext, intTimeoutMS)

def SaveDDT(intChannelIndex, charFilename):
	fun=lib.MAG_SaveDDT
	if IS_WINDOWS:
		fun.argtypes = [c_uint, c_wchar_p]
	else:
		fun.argtypes = [c_uint, c_char_p]
	fun.restype = c_int
	return fun(intChannelIndex, charFilename)

def SaveDDT2Buffer(intChannelIndex, pBuffer, intBufferSize):
	fun=lib.MAG_SaveDDT2Buffer
	fun.argtypes = [c_uint, c_void_p, c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, pBuffer, intBufferSize)

def SaveMGT(intChannelIndex, charFilename):
	fun=lib.MAG_SaveMGT
	if IS_WINDOWS:
		fun.argtypes = [c_uint, c_wchar_p]
	else:
		fun.argtypes = [c_uint, c_char_p]
	fun.restype = c_int
	return fun(intChannelIndex, charFilename)

def SDCardStorage(intChannelIndex, filetype, para):
	fun=lib.MAG_SDCardStorage
	fun.argtypes = [c_uint, c_int, c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, filetype, para)

def SDStorageAviStart(intChannelIndex):
	fun=lib.MAG_SDStorageAviStart
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def SDStorageAviStop(intChannelIndex):
	fun=lib.MAG_SDStorageAviStop
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def SDStorageBMP(intChannelIndex):
	fun=lib.MAG_SDStorageBMP
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def SDStorageMGSStart(intChannelIndex):
	fun=lib.MAG_SDStorageMGSStart
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def SDStorageMGSStop(intChannelIndex):
	fun=lib.MAG_SDStorageMGSStop
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def SDStorageMGT(intChannelIndex):
	fun=lib.MAG_SDStorageMGT
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def SetAsyncCompressCallBack(intChannelIndex, pCallBack, intQuality):
	fun=lib.MAG_SetAsyncCompressCallBack
	# FIXME
	fun.argtypes = [c_uint, MAG_DDTCOMPRESSCALLBACK, py_object]
	fun.restype = c_int
	return fun(intChannelIndex, pCallBack, intQuality)

def SetAutoEnlargePara(intChannelIndex, dwAutoEnlargeRange, intBrightOffset, intContrastOffset):
	fun=lib.MAG_SetAutoEnlargePara
	fun.argtypes = [c_uint, c_uint, c_int, c_int]
	fun.restype = None
	return fun(intChannelIndex, dwAutoEnlargeRange, intBrightOffset, intContrastOffset)

def SetCameraRegContent(intChannelIndex, pContent):
	fun=lib.MAG_SetCameraRegContent
	fun.argtypes = [c_uint, POINTER(CeRegContent)]
	fun.restype = c_int
	return fun(intChannelIndex, pContent)

def SetCameraRegContent2(intChannelIndex, pContent):
	fun=lib.MAG_SetCameraRegContent2
	fun.argtypes = [c_uint, POINTER(CfgPara)]
	fun.restype = c_int
	return fun(intChannelIndex, pContent)

def SetCamRegs(intChannelIndex, pContent):
	fun=lib.MAG_SetCamRegs
	fun.argtypes = [c_uint, POINTER(CamRegs)]
	fun.restype = c_int
	return fun(intChannelIndex, pContent)

def SetColorPalette(intChannelIndex, ColorPaletteIndex):
	fun=lib.MAG_SetColorPalette
	fun.argtypes = [c_uint, c_int]
	fun.restype = None
	return fun(intChannelIndex, ColorPaletteIndex)

def SetDetailEnhancement(intChannelIndex, intDDE, bQuickDDE):
	fun=lib.MAG_SetDetailEnhancement
	fun.argtypes = [c_uint, c_int, c_int]
	fun.restype = None
	return fun(intChannelIndex, intDDE, bQuickDDE)

def SetEnhancedROI(intChannelIndex, intEnhancedRatio, x0, y0, x1, y1):
	fun=lib.MAG_SetEnhancedROI
	fun.argtypes = [c_uint, c_uint, c_uint, c_uint, c_uint, c_uint]
	fun.restype = None
	return fun(intChannelIndex, intEnhancedRatio, x0, y0, x1, y1)

def SetEXLevel(intChannelIndex, ExLevel, intCenterX, intCenterY):
	fun=lib.MAG_SetEXLevel
	fun.argtypes = [c_uint, c_int, c_int, c_int]
	fun.restype = None
	return fun(intChannelIndex, ExLevel, intCenterX, intCenterY)

def SetFilter(intFilter):
	fun=lib.MAG_SetFilter
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intFilter)

def SetFixedEnlargePara(intChannelIndex, intX1, intX2, byteY1, byteY2):
	fun=lib.MAG_SetFixedEnlargePara
	fun.argtypes = [c_uint, c_int, c_int, c_int]
	fun.restype = c_int
	return fun(intChannelIndex, intX1, intX2, byteY1, byteY2)

def SetFixPara(intChannelIndex, pBuffer, enumFixOption):
	fun=lib.MAG_SetFixPara
	fun.argtypes = [c_uint, POINTER(FixPara), c_int]
	fun.restype = c_float
	return fun(intChannelIndex, pBuffer, enumFixOption)

def SetIoAlarmState(intChannelIndex, bAlarm):
	fun=lib.MAG_SetIoAlarmState
	fun.argtypes = [c_uint, c_int]
	fun.restype = c_int
	return fun(intChannelIndex, bAlarm)

def SetIsothermalPara(intChannelIndex, intLowerLimit, intUpperLimit):
	fun=lib.MAG_SetIsothermalPara
	fun.argtypes = [c_uint, c_int, c_int]
	fun.restype = None
	return fun(intChannelIndex, intLowerLimit, intUpperLimit)

def SetPCounter(intChannelIndex, intUpCounter, intDownCounter):
	fun=lib.MAG_SetPCounter
	fun.argtypes = [c_uint, c_uint, c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, intUpCounter, intDownCounter)

def SetPTZCmd(intChannelIndex, cmd, dwPara):
	fun=lib.MAG_SetPTZCmd
	fun.argtypes = [c_uint, c_int, c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, cmd, dwPara)

def SetReConnectCallBack(intChannelIndex, pCallBack, pUserData):
	fun=lib.MAG_SetReConnectCallBack
	fun.argtypes = [c_uint, MAG_RECONNECTCALLBACK, py_object]
	fun.restype = c_int
	return fun(intChannelIndex, pCallBack, pUserData)

def SetROIReportCallBack(intChannelIndex, pCallBack, pUserData):
	fun=lib.MAG_SetROIReportCallBack
	fun.argtypes = [c_uint, MAG_ROICALLBACK, py_object]
	fun.restype = c_int
	return fun(intChannelIndex, pCallBack, pUserData)

def SetIrregularROIReportExCallBack(intChannelIndex, pCallBack, pUserData):
	fun=lib.MAG_SetIrregularROIReportExCallBack
	fun.argtypes = [c_uint, MAG_IRREGULARROICALLBACK, py_object]
	fun.restype = c_int
	return fun(intChannelIndex, pCallBack, pUserData)

def SetSerialCallBack(intChannelIndex, pCallBack, pUserData):
	fun=lib.MAG_SetSerialCallBack
	fun.argtypes = [c_uint, MAG_SERIALCALLBACK, py_object]
	fun.restype = c_int
	return fun(intChannelIndex, pCallBack, pUserData)

def SetSerialCmd(intChannelIndex, buff, intBufferLen):
	fun=lib.MAG_SetSerialCmd
	fun.argtypes = [c_uint, POINTER(c_ubyte), c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, buff, intBufferLen)

def SetSubsectionEnlargePara(intChannelIndex, intX1, intX2, byteY1, byteY2):
	fun=lib.MAG_SetSubsectionEnlargePara
	fun.argtypes = [c_uint, c_int, c_int, c_ubyte, c_ubyte]
	fun.restype = c_int
	return fun(intChannelIndex, intX1, intX2, byteY1, byteY2)

def SetUserROIs(intChannelIndex, pROI):
	fun=lib.MAG_SetUserROIs
	fun.argtypes = [c_uint, POINTER(UserROIs)]
	fun.restype = c_int
	return fun(intChannelIndex, pROI)

def SetUserROIsEx(intChannelIndex, pROIs, intROINum):
	fun=lib.MAG_SetUserROIsEx
	fun.argtypes = [c_uint, POINTER(RectROI), c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, pROIs, intROINum)

def SetIrregularROIs(intChannelIndex, pROIs, intROINum):
	fun = lib.MAG_SetIrregularROIs
	fun.argtypes = [c_uint, POINTER(IrregularROI), c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, pROIs, intROINum)

def SetVideoBrightness(intChannelIndex, intBrightnessOffset):
	fun=lib.MAG_SetVideoBrightness
	fun.argtypes = [c_uint, c_int]
	fun.restype = c_int
	return fun(intChannelIndex, intBrightnessOffset)

def SetVideoContrast(intChannelIndex, intContrastOffset):
	fun=lib.MAG_SetVideoContrast
	fun.argtypes = [c_uint, c_int]
	fun.restype = c_int
	return fun(intChannelIndex, intContrastOffset)

def StartDecodeVideo(intChannelIndex, paraOut, funcFrame, pUserData):
	fun=lib.MAG_StartDecodeVideo
	fun.argtypes = [c_uint, POINTER(OutputPara), MAG_FRAMECALLBACK, py_object]
	fun.restype = c_int
	return fun(intChannelIndex, paraOut, funcFrame, pUserData)

def StartDHCPServer(hWndMsg):
	fun=lib.MAG_StartDHCPServer
	fun.argtypes = [HWND]
	fun.restype = c_int
	return fun(hWndMsg)

def StartProcessImage(intChannelIndex, paraOut, funcFrame, dwStreamType, pUserData):
	fun=lib.MAG_StartProcessImage
	fun.argtypes = [c_uint, POINTER(OutputPara), MAG_FRAMECALLBACK, c_uint, py_object]
	fun.restype = c_int
	return fun(intChannelIndex, paraOut, funcFrame, dwStreamType, pUserData)

def StartProcessPulseImage(intChannelIndex, paraOut, funcFrame, dwStreamType, pUserData):
	fun=lib.MAG_StartProcessPulseImage
	fun.argtypes = [c_uint, POINTER(OutputPara), MAG_FRAMECALLBACK, c_uint, py_object]
	fun.restype = c_int
	return fun(intChannelIndex, paraOut, funcFrame, dwStreamType, pUserData)

def StopDHCPServer():
	fun=lib.MAG_StopDHCPServer
	fun.argtypes = []
	fun.restype = None
	return fun()

def StopLensMotor(intChannelIndex):
	fun=lib.MAG_StopLensMotor
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def StopListen(intChannelIndex):
	fun=lib.MAG_StopListen
	fun.argtypes = [c_uint]
	fun.restype = None
	return fun(intChannelIndex)

def StopProcessImage(intChannelIndex):
	fun=lib.MAG_StopProcessImage
	fun.argtypes = [c_uint]
	fun.restype = None
	return fun(intChannelIndex)

def TransferPulseImage(intChannelIndex):
	fun=lib.MAG_TransferPulseImage
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def TransferPulseImageWait(intChannelIndex, intTimeoutMS):
	fun=lib.MAG_TransferPulseImageWait
	fun.argtypes = [c_uint, c_int]
	fun.restype = c_int
	return fun(intChannelIndex, intTimeoutMS)

def TriggerFFC(intChannelIndex):
	fun=lib.MAG_TriggerFFC
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def UnLockFrame(intChannelIndex):
	fun=lib.MAG_UnLockFrame
	fun.argtypes = [c_uint]
	fun.restype = None
	return fun(intChannelIndex)

def UseTemperatureMask(intChannelIndex, bUse):
	fun=lib.MAG_UseTemperatureMask
	fun.argtypes = [c_uint, c_int]
	fun.restype = c_int
	return fun(intChannelIndex, bUse)

def SetObjRecoCallBack(intChannelIndex, pCallback, pUserData):
	fun = lib.MAG_SetObjRecoCallBack
	fun.argtypes = [c_uint, MAG_OBJRECOCALLBACK, py_object]
	fun.restype = c_int
	return fun(intChannelIndex, pCallback, pUserData)

# NPU 
def NPUSetNetWork(intChannelIndex, pSrcPara, wcharNetworkFile, intTimeoutMS):
	fun=lib.MAG_NPU_SetNetWork
	if IS_WINDOWS:
		fun.argtypes = [c_uint, POINTER(NPUSrc), c_wchar_p, c_uint]
	else:
		fun.argtypes = [c_uint, POINTER(NPUSrc), c_char_p, c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, pSrcPara, wcharNetworkFile, intTimeoutMS)

def NPUSetWeight(intChannelIndex, wcharWeightFile, intTimeoutMS):
	fun=lib.MAG_NPU_SetWeight
	if IS_WINDOWS:
		fun.argtypes = [c_uint, c_wchar_p, c_uint]
	else:
		fun.argtypes = [c_uint, c_char_p, c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, wcharWeightFile, intTimeoutMS)

def NPUUpdateTestImage(intChannelIndex, wcharTestFile, intTimeoutMS):
	fun=lib.MAG_NPU_UpdateTestImage
	if IS_WINDOWS:
		fun.argtypes = [c_uint, c_wchar_p, c_uint]
	else:
		fun.argtypes = [c_uint, c_char_p, c_uint]
	fun.restype = c_int
	return fun(intChannelIndex, wcharTestFile, intTimeoutMS)

def NPUStop(intChannelIndex):
	fun=lib.MAG_NPU_Stop
	fun.argtypes = [c_uint]
	fun.restype = None
	return fun(intChannelIndex)

def NPUStart(intChannelIndex, intInferencePriod, funcNPU, pUserData):
	fun=lib.MAG_NPU_Start
	fun.argtypes = [c_uint, c_uint, MAG_NPUCALLBACK, py_object]
	fun.restype = c_int
	return fun(intChannelIndex, intInferencePriod, funcNPU, pUserData)

def ConvertIrCorr2VisCorr(intChannelIndex, intIrX, intIrY, fDistance, intVisX, intVisY):
	fun = lib.MAG_ConvertIrCorr2VisCorr
	fun.argtypes = [c_uint, c_int, c_int, c_float, POINTER(c_int), POINTER(c_int)]
	fun.restype = c_int
	return fun(intChannelIndex, intIrX, intIrY, fDistance, intVisX, intVisY)

def ConvertVisCorr2IrCorr(intChannelIndex, intVisX, intVisY, fDistance, intIrX, intIrY):
	fun = lib.MAG_ConvertVisCorr2IrCorr
	fun.argtypes = [c_uint, c_int, c_int, c_float, POINTER(c_int), POINTER(c_int)]
	fun.restype = c_int
	return fun(intChannelIndex, intVisX, intVisY, fDistance, intIrX, intIrY)

def StartVis(intChannelIndex, rtspUrl, pixelFormat, frameCallback, pUserData, protocol, timeout):
	fun = vis_lib.MAG_StartVis
	fun.argtypes = [c_uint, c_char_p, c_int, MAG_VISFRAMECALLBACK, py_object, c_int, c_int]
	fun.restype = c_int
	return fun(intChannelIndex, rtspUrl.encode("utf-8"), pixelFormat, frameCallback, pUserData, protocol, timeout)

def StopVis(intChannelIndex):
	fun = vis_lib.MAG_StopVis
	fun.argtypes = [c_uint]
	fun.restype = None
	fun(intChannelIndex)

def IsVisStarted(intChannelIndex):
	fun = vis_lib.MAG_IsVisStarted
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def SetVisInitCallback(intChannelIndex, initCallback, pUserData):
	fun = vis_lib.MAG_SetVisInitCallback
	fun.argtypes = [c_uint, MAG_INITCALLBACK, py_object]
	fun.restype = c_int
	return fun(intChannelIndex, initCallback, pUserData) == 0

def SetVisReconnectCallback(intChannelIndex, reconnectCallback, pUserData):
	fun = vis_lib.MAG_SetVisReconnectCallback
	fun.argtypes = [c_uint, MAG_VISRECONNECTCALLBACK, py_object]
	fun.restype = None
	fun(intChannelIndex, reconnectCallback, pUserData)

def VisWidth(intChannelIndex):
	fun = vis_lib.MAG_VisWidth
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def VisHeight(intChannelIndex):
	fun = vis_lib.MAG_VisHeight
	fun.argtypes = [c_uint]
	fun.restype = c_int
	return fun(intChannelIndex)

def SaveVis(intChannelIndex, charFileName):
	fun = vis_lib.MAG_SaveVis
	fun.argtypes = [c_uint, c_char_p]
	fun.restype = c_int
	return fun(intChannelIndex, charFileName)

def LockVisFrame(intChannelIndex):
	fun = vis_lib.MAG_LockVisFrame
	fun.argtypes = [c_uint]
	fun.restype = None
	fun(intChannelIndex)

def UnlockVisFrame(intChannelIndex):
	fun = vis_lib.MAG_UnlockVisFrame
	fun.argtypes = [c_uint]
	fun.restype = None
	fun(intChannelIndex)

def GetVisData(intChannelIndex, data, info):
	fun = vis_lib.MAG_GetVisData
	fun.argtypes = [c_uint, POINTER(POINTER(c_ubyte)), POINTER(POINTER(c_ubyte))]
	fun.restype = c_int
	return fun(intChannelIndex, data, info)

def GetVisData_copy(intChannelIndex, data, size):
	fun = vis_lib.MAG_GetVisData_copy
	fun.argtypes = [c_uint, POINTER(c_ubyte), c_int]
	fun.restype = c_int
	return fun(intChannelIndex, data, size)
