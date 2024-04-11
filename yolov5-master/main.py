from sample_live import test_live_camera
from sample_ddt import test_ddt_file
from sample_mgs import test_mgs_file
from sample_live_opencv import test_live_camera_opencv
from sample_mgs_opencv import test_mgs_file_opencv
from sample_ddt_opencv import test_ddt_file_opencv


if __name__ == "__main__":
    print("Begin live ...\n")
    test_live_camera("192.168.1.161", 30000)
    test_live_camera_opencv("192.168.1.161", 30000)
    print("\nExit live ...\n")

    # print("Begin ddt ...\n")
    # test_ddt_file("test.ddt")
    # test_ddt_file_opencv("test.ddt")
    # print("\nExit ddt ...\n")

    # print("Begin mgs ...\n")
    # test_mgs_file("test.mgs")
    # test_mgs_file_opencv("test.mgs")
    # print("\nExit mgs ...\n")
