import imageio
import os

path = "paraview_video/"

fileList = []
for file in os.listdir(path):
    complete_path = path + file
    fileList.append(complete_path)

writer = imageio.get_writer("paraview_shp_video.gif", fps=2)
images = []
for im in fileList:
    writer.append_data(imageio.imread(im))
    # images.append(imageio.imread(im))

writer.close()
