#!/usr/bin/env python3

import numpy as np
from PIL import Image
import argparse

def genLoadTrackDataTestMap():

    # is this correct??
    width = 10
    height = 20
    pixelsPerMeter = 2

    channel0 = np.zeros(width*pixelsPerMeter, *height*pixelsPerMeter)
    channel1 = np.zeros(width*pixelsPerMeter, *height*pixelsPerMeter)
    channel2 = np.zeros(width*pixelsPerMeter, *height*pixelsPerMeter)
    channel3 = np.zeros(width*pixelsPerMeter, *height*pixelsPerMeter)

    #Save data to numpy array, each channel is saved individually as an array in row major order.
    track_dict = {"xBounds":np.array([width/2, width/2], dtype = np.float32),
                  "yBounds":np.array([height/2, height/2], dtype = np.float32),
                  "pixelsPerMeter":np.array([pixelsPerMeter], dtype=np.float32),
                  "channel0":channel0,
                  "channel1":channel1,
                  "channel2":channel2,
                  "channel3":channel3}

    for i in range(0, width):
        for j in range(0, height):
            channel0 = height * i + j
            channel1 = (height * i + j) * 10
            channel2 = (height * i + j) * 100
            channel3 = (height * i + j) * 1000

    np.savez("../../../resource/autorally/test/test_map", **track_dict)


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("-i", "--input", type = str, help = "Costmap in old .txt format")
    #parser.add_argument("-d", "--display", type = str, help = "Name of image to save costmap as", default="display_image.jpg")
    #parser.add_argument("-o", "--output", type = str, help = "File to save map to", default = "map.npz")
    #args = vars(parser.parse_args())
    gen_costmap(args["input"], args["display"], args["output"])