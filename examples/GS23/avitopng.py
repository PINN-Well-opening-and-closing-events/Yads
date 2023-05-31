import cv2 as cv


class Avi2jpg:
    vid = None

    def convert(self, video):
        vidcap = cv.VideoCapture(video)
        success, image = vidcap.read()
        count = 0
        while success:
            framecount = "{number:06}".format(number=count)
            cv.imwrite(framecount + ".jpg", image)  # save frame as JPEG file
            success, image = vidcap.read()
            print("Read a new frame: ", success)
            count += 1


if __name__ == "__main__":
    vid_path = "first_non_physical_video/test_idc.avi"
    converter = Avi2jpg()
    converter.convert(vid_path)
