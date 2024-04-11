import MagSDK

import ctypes


def singleton(cls):
    instances = {}

    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return getinstance


@singleton
class MagService(object):
    def __init__(self):
        self.__initialized = False
        self.__maxEnumDevice = 32

    def Deinitialize(self):
        if MagSDK.IsDHCPServerRunning():
            MagSDK.StopDHCPServer()

        if MagSDK.IsInitialized(0):
            MagSDK.Free(0)

        if MagSDK.IsChannelAvailable(0):
            MagSDK.DelChannel(0)

        self.__initialized = False

    def Initialize(self):
        if self.__initialized:
            return True

        if not (MagSDK.IsChannelAvailable(0)):
            bSuccess = MagSDK.NewChannel(0)

        if MagSDK.IsLanConnected():
            self.__initialized = MagSDK.Initialize(0, None)

        return self.__initialized

    def IsInitialized(self):
        return self.__initialized

    def IsLanConnected(self):
        return MagSDK.IsLanConnected()

    def IsUsingStaticIp(self):
        return MagSDK.IsUsingStaticIp()

    def GetLocalIp(self):
        if not self.__initialized:
            self.Initialize()

        return MagSDK.GetLocalIp() if self.__initialized else 0

    def IsDHCPServerRunning(self):
        return MagSDK.IsDHCPServerRunning()

    def StartDHCPServer(self):
        return MagSDK.StartDHCPServer(None)

    def StopDHCPServer(self):
        MagSDK.StopDHCPServer()

    def EnableAutoReConnect(self, bEnable):
        MagSDK.EnableAutoReConnect(bEnable)

    def EnumCameras(self):
        return MagSDK.EnumCameras() if self.__initialized else False

    def InfrastructureGetTerminalList(self, pList, dwBufferSize):
        return MagSDK.GetTerminalList(pList, dwBufferSize) if self.__initialized else 0

    def GetTerminalList(self):
        """Get terminal list.
		
		Enum terminal list then return enumCount and enumList.

		Returns:
			An integer type named enumCount and a ctypes pointer array (MagSDK.TerminalList structure) named enumList. 
			You can regard the enumList as a python List. For example:
			
			(enumCount, enumList) = GetTerminalList()
			charname = enumList[0].charTerminalName
			ip_address = enumList[0].intTerminalIp
		"""
        enumList = (MagSDK.TerminalList * self.__maxEnumDevice)()
        enumCount = self.InfrastructureGetTerminalList(enumList,
                                                       self.__maxEnumDevice * ctypes.sizeof(MagSDK.TerminalList))
        if enumCount > 0:
            return enumCount, enumList
        else:
            return 0, None

    def GetTerminalCount(self):
        return MagSDK.GetTerminalList(None, 0) if self.__initialized else 0

    def CompressDDT(self, pDstBuffer, intDstBufferSize, pSrcBuffer, intSrcBufferSize, intQuality):
        return MagSDK.CompressDDT(pDstBuffer, intDstBufferSize, pSrcBuffer, intSrcBufferSize, intQuality)

    def DeCompressDDT(self, pDstBuffer, intDstBufferSize, pSrcBuffer, intSrcBufferSize):
        return MagSDK.DeCompressDDT(pDstBuffer, intDstBufferSize, pSrcBuffer, intSrcBufferSize)
