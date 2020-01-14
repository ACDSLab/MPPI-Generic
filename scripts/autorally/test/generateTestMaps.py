#!/usr/bin/env python3

import numpy as np
from PIL import Image
import argparse

def genLoadTrackDataTestMap(args):

    # is this correct??
    width = 10
    height = 20
    pixelsPerMeter = 2

    channel0 = np.zeros((width*pixelsPerMeter, height*pixelsPerMeter), np.float32)
    channel1 = np.zeros((width*pixelsPerMeter, height*pixelsPerMeter), np.float32)
    channel2 = np.zeros((width*pixelsPerMeter, height*pixelsPerMeter), np.float32)
    channel3 = np.zeros((width*pixelsPerMeter, height*pixelsPerMeter), np.float32)

    counter = 0
    for i in range(0, width*pixelsPerMeter):
        for j in range(0, height*pixelsPerMeter):
            print("putting width and height, ", pixelsPerMeter * height * i + j, " ", counter)
            counter+= 1
            channel0[i, j] = pixelsPerMeter * height * i + j
            channel1[i, j] = (pixelsPerMeter * height * i + j) * 10
            channel2[i, j] = (pixelsPerMeter * height * i + j) * 100
            channel3[i, j] = (pixelsPerMeter * height * i + j) * 1000
    print(channel0)

    #Save data to numpy array, each channel is saved individually as an array in row major order.
    track_dict = {"xBounds":np.array([-width/2, width/2], dtype = np.float32),
                  "yBounds":np.array([-height/2, height/2], dtype = np.float32),
                  "pixelsPerMeter":np.array([pixelsPerMeter], dtype=np.float32),
                  "channel0":channel0,
                  "channel1":channel1,
                  "channel2":channel2,
                  "channel3":channel3}


    np.savez(args.output, **track_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates a npz map file for testing purposes")
    # parser.add_argument("-i", "--input", type = str, help = "Costmap in old .txt format")
    # parser.add_argument("-d", "--display", type = str,
    #                     help = "Name of image to save costmap as",
    #                     default="display_image.jpg")
    parser.add_argument("-o", "--output", type = str,
                        help = "File to save map to",
                        default = "../../../resource/autorally/test/test_map.npz")
    args = parser.parse_args()
    genLoadTrackDataTestMap(args)
