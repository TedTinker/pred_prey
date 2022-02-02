# What is an agent?
import torch

class Agent:
    def __init__(self, predator, p_num, energy, pos, yaw, spe):
        self.predator = predator; self.p_num = p_num; self.energy = energy
        self.pos = pos; self.yaw = yaw; self.spe = spe
        self.age = 0
        self.action = torch.tensor([0.0, 0.0])
        self.hidden = None
        self.to_push = []



# How to make physicsClients.
import pybullet as p
import pybullet_data



def get_physics(GUI, w, h):
    if(GUI):
        physicsClient = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(1,90,-89,(w/2,h/2,w), physicsClientId = physicsClient)
    else:   
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId = physicsClient)
    return(physicsClient)



# Get arena from image.
import numpy as np
from math import pi, sin, cos
import cv2
from itertools import product
from scipy.stats import percentileofscore
import random

from utils import parameters as para
from utils import get_arg

def pythagorean(pos_1, pos_2):
    return ((pos_1[0] - pos_2[0])**2 + (pos_1[1] - pos_2[1])**2)**.5

class Arena():
    def __init__(self, para = para, GUI = False):
        self.para = para
        self.arena_map = cv2.imread("arenas/" + para.arena_name + ".png")
        self.w, self.h, _ = self.arena_map.shape
        self.physicsClient = get_physics(GUI, self.w, self.h)
        self.open_spots = [(x,y) for x, y in product(range(self.w), range(self.h)) \
                          if self.arena_map[x,y].tolist() == [255, 255, 255]]
        num_open_spots = range(len(self.open_spots))
        self.pairs = [(self.open_spots[i], self.open_spots[j]) for \
                      i, j in product(num_open_spots, num_open_spots) if i != j]
        distances = [pythagorean(self.pairs[i][0], self.pairs[i][1]) for \
                      i in range(len(self.pairs))]
        self.difficulties = [percentileofscore(distances, d) for d in distances]
        self.already_constructed = False
  
    def get_pair_with_difficulty(self, min_dif = 0, max_dif = 100, verbose = False):
        if(max_dif == None):   max_dif = min_dif
        if(min_dif > max_dif): min_dif = max_dif
        pair_list = []
        while(len(pair_list) == 0):
            pair_list = [self.pairs[i] for i in range(len(self.pairs)) if \
                        self.difficulties[i] >= min_dif and self.difficulties[i] <= max_dif]
            min_dif -= 1; max_dif += 1
        if(verbose):
            print("Predator/Prey positions with minimum difficulty {}, maximum difficulty {}.".format(
                min_dif, max_dif))
            for pair in pair_list:
                print(pair)
        return(random.choice(pair_list))

    def start_arena(self, min_dif = None, max_dif = None):
        if(not self.already_constructed):
            self.wall_ids = []
            for loc in ((x,y) for x in range(self.w) for y in range(self.h)):
                if(not (self.arena_map[loc] == [255]).all()):
                  pos = [loc[0],loc[1],.5]
                  ors = p.getQuaternionFromEuler([0,0,0])
                  cube = p.loadURDF("cube.urdf",pos,ors, useFixedBase = True, physicsClientId = self.physicsClient)
                  color = self.arena_map[loc][::-1] / 255
                  color = np.append(color, 1)
                  p.changeVisualShape(cube, -1, rgbaColor=color, physicsClientId = self.physicsClient)
                  self.wall_ids.append(cube)
                  self.already_constructed = True
                    
        pred_pos, prey_pos = self.get_pair_with_difficulty(min_dif, max_dif)
        pred = self.make_agent(True, pred_pos)
        prey = self.make_agent(False, prey_pos)
        return([pred, prey])
    
    def make_agent(self, predator, pos):
        yaw = random.uniform(0, 2*pi)
        spe = get_arg(self.para, predator, "min_speed")
        energy = get_arg(self.para, predator, "energy")
        color = [1,0,0,1] if predator else [0,0,1,1]
        file = "sphere2red.urdf"
        
        pos = (pos[0], pos[1], .5)
        ors = p.getQuaternionFromEuler([0,0,yaw])
        p_num = p.loadURDF(file,pos,ors,
                           globalScaling = self.para.pred_size if predator else self.para.prey_size, 
                           physicsClientId = self.physicsClient)
        x, y = cos(yaw)*spe, sin(yaw)*spe
        p.resetBaseVelocity(p_num, (x,y,0),(0,0,0), physicsClientId = self.physicsClient)
        p.changeVisualShape(p_num, -1, rgbaColor = color, physicsClientId = self.physicsClient)
        agent = Agent(predator, p_num, energy, pos, yaw, spe)
        return(agent)
    
    def make_flower(self):
        pos   = random.choice(self.open_spots)
        roll  = random.uniform(0, 2*pi)
        pitch = random.uniform(0, 2*pi)
        yaw   = random.uniform(0, 2*pi)
        file  = "duck_vhacd.urdf"
        
        pos = (pos[0], pos[1], .5)
        ors = p.getQuaternionFromEuler([roll,pitch,yaw])
        color = [random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),1]
        flower = p.loadURDF(file,pos,ors,
                            globalScaling = self.para.flower_size,
                            physicsClientId = self.physicsClient)
        p.resetBaseVelocity(flower, (0,0,0),(0,0,0), physicsClientId = self.physicsClient)
        p.changeVisualShape(flower, -1, rgbaColor = color, physicsClientId = self.physicsClient)
        return(flower)
    
    def wall_collisions(self, p_num):
        col = False
        for wall in self.wall_ids:
            if 0 < len(p.getContactPoints(p_num, wall, physicsClientId = self.physicsClient)):
              col = True
        return(col)

    def agent_collisions(self, p_num_1, p_num_2):
        return(0 < len(p.getContactPoints(
            p_num_1, p_num_2, physicsClientId = self.physicsClient)))
    
    def agent_dist(self, p_num_1, p_num_2):
        pos_1,_ ,_ = self.get_pos_yaw_spe(p_num_1)
        pos_2,_ ,_ = self.get_pos_yaw_spe(p_num_2)
        x = pos_1[0] - pos_2[0]
        y = pos_1[1] - pos_2[1]
        return((x**2 + y**2)**.5)
    
    def get_pos_yaw_spe(self, p_num):
        pos, ors = p.getBasePositionAndOrientation(p_num, physicsClientId = self.physicsClient)
        yaw = p.getEulerFromQuaternion(ors)[-1]
        (x, y, _), _ = p.getBaseVelocity(p_num, physicsClientId = self.physicsClient)
        spe = (x**2 + y**2)**.5
        return(pos, yaw, spe)



if __name__ == "__main__":
    arena = Arena()
    arena.get_pair_with_difficulty(min_dif = 0, max_dif = 0, verbose = True)
    arena.get_pair_with_difficulty(min_dif = 50, max_dif = 50, verbose = True)
    arena.get_pair_with_difficulty(min_dif = 100, max_dif = 100, verbose = True)
