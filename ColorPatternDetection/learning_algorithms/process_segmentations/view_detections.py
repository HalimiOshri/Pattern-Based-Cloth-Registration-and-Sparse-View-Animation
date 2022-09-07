import pickle
import os

if __name__ == '__main__':
    with open('/Users/oshrihalimi/Downloads/detections_cam401532_552.pkl', 'rb') as inp:
        d = pickle.load(inp)
    d.plot_edges('graph_401532_552.png', 'color_401532_552.png')