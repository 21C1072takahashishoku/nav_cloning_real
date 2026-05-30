#!/usr/bin/env python3
import roslib
roslib.load_manifest('nav_cloning')
import rospy
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from PIL import Image
from matplotlib.patches import Circle
import csv
from std_msgs.msg import Bool, String

class nav_cloning_node2:
    def __init__(self):
        rospy.init_node('nav_cloning_node2', anonymous=True)
        
       
        self.num = 0
        self.i = 1
        self.laps = False
        self.dir_sub = rospy.Subscriber("/dir", String, self.callback_dir)
        self.dir = str()
        self.mode_sub = rospy.Subscriber("/mode", Bool, self.callback_mode)
        self.mode = str()
        self.trigger = False
        self.laps_sub = rospy.Subscriber("/laps", Bool, self.callback_laps)
        self.reset = 0

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim([-5, 17.3])
        self.ax.set_ylim([-2, 24])
        self.ax.text(13, -4.8, "Total: " + str(self.num), color="b", fontsize="xx-large")
        self.ax.text(-4.2, -4.8, "Number of laps: " + str(self.i), color="r", fontsize="xx-large")
        self.image = Image.open('/home/orne_beta/imai_gamma_ws/src/nav_cloning/maps/cit_3_map_plot.pgm').convert("L")
        self.arr = np.asarray(self.image)
        self.map_plot = self.ax.imshow(self.arr, cmap='gray', extent=[-20.5,17.5,-9,29])
        self.patches = [] 
    
    def callback_dir(self, data):
        self.dir = data.data

    def callback_mode(self, data):
        self.mode = data.data

    def callback_laps(self, data):
        self.laps = data.data

   

    def update(self, frame):
        if self.reset == 0:
            with open(self.dir + '/' + 'training.csv', 'w') as f:
                f.write("")
            self.reset = 1
        self.ax.clear()
        self.ax.set_xlim([-5, 17.3])
        self.ax.set_ylim([-2, 24])        
        self.ax.text(13, -4.8, "Total: " + str(self.num), color="b", fontsize="xx-large")
        self.ax.text(-4.2, -4.8, "Number of laps: " + str(self.i), color="r", fontsize="xx-large")
        self.map_plot = self.ax.imshow(self.arr, cmap='gray', extent=[-20.5,17.5,-9,29])

        if self.trigger:
            self.ax.text(24, -4.8, "Test", color="c", fontsize="xx-large")
        
        else:
            self.ax.text(24, -4.8, "Train", color="m", fontsize="xx-large")
            

        with open(self.dir + '/' + 'training.csv', 'r') as f:
            for row in csv.reader(f):
                str_step, mode, distance, str_x, str_y, str_the = row
                x, y = float(str_x), float(str_y)
                if self.trigger:
                    self.patches.append(Circle(xy=(x, y), radius=0.10, facecolor="b"))
                    self.num += 1
                elif mode == "test":
                    pass
                else:
                    self.patches.append(Circle(xy=(x, y), radius=0.10, facecolor="b"))
                    self.num += 1
                
        
        with open(self.dir + '/' + 'training.csv', 'w') as f:
            f.write("")
                    
        for patch in self.patches:
            self.ax.add_patch(patch)
            



        if self.laps:
            plt.savefig(self.dir + "/" + str(self.i) + ".png") 
            self.laps = False
            self.i += 1
            self.num = 0
            self.patches.clear()
            if self.mode == False:
                self.trigger = True
            
        
       

    def animate(self):
        animation = FuncAnimation(self.fig, self.update, frames=10, interval=3000)
        plt.show()

if __name__ == '__main__':
    l = nav_cloning_node2()
    l.animate()