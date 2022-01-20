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
import os

file_1 = r"C:\Users\tedjt\Desktop\pred_prey"

def pythagorean(pos_1, pos_2):
  return ((pos_1[0] - pos_2[0])**2 + (pos_1[1] - pos_2[1])**2)**.5

class Arena():
  def __init__(self, arena_name,
               agent_size = .5):
    self.arena_name = arena_name
    os.chdir(file_1)
    self.arena = cv2.imread("arenas/" + arena_name); self.w, self.h, _ = self.arena.shape
    #file_2 = r"C:\Users\tedjt"
    #os.chdir(file_2)
    self.agent_size = agent_size
    self.open_spots = [(x,y) for x, y in product(range(self.w), range(self.h)) \
                      if self.arena[x,y].tolist() == [255, 255, 255]]
    num_open_spots = range(len(self.open_spots))
    self.pairs = [(self.open_spots[i], self.open_spots[j]) for \
                  i, j in product(num_open_spots, num_open_spots) if i != j]
    distances = [pythagorean(self.pairs[i][0], self.pairs[i][1]) for \
                  i in range(len(self.pairs))]
    self.difficulties = [percentileofscore(distances, d) for d in distances]
  
  def get_pair_with_difficulty(self, min_dif = 0, max_dif = 100, verbose = False):
    if(max_dif == None): max_dif = min_dif
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

  def start_arena(
    self, physicsClient, already_constructed = False,
    min_dif = None, max_dif = None, start_speed = 0):
    
    wall_ids = []
    if(not already_constructed):
      for loc in ((x,y) for x in range(self.w) for y in range(self.h)):
        if(not (self.arena[loc] == [255]).all()):
          pos = [loc[0],loc[1],.5]
          ors = p.getQuaternionFromEuler([0,0,0])
          cube = p.loadURDF("cube.urdf",pos,ors, useFixedBase = True, physicsClientId = physicsClient)
          color = self.arena[loc][::-1] / 255
          color = np.append(color, 1)
          p.changeVisualShape(cube, -1, rgbaColor=color, physicsClientId = physicsClient)
          wall_ids.append(cube)
                
    pred_pos, prey_pos = self.get_pair_with_difficulty(min_dif, max_dif)
    pred_yaw, prey_yaw = random.uniform(0, 2*pi), random.uniform(0, 2*pi)
    pred_spe, prey_spe = start_speed, start_speed
    
    file = "sphere2red.urdf"
    #file = "pred_prey.urdf" # How can I make my own robot-shape? 
    
    pos = (pred_pos[0], pred_pos[1], .5)
    ors = p.getQuaternionFromEuler([0,0,pred_yaw])
    pred = p.loadURDF(file,pos,ors,globalScaling = self.agent_size, physicsClientId = physicsClient)
    x, y = cos(pred_yaw)*pred_spe, sin(pred_yaw)*pred_spe
    p.resetBaseVelocity(pred, (x,y,0),(0,0,0), physicsClientId = physicsClient)
    p.changeVisualShape(pred, -1, rgbaColor = [1,0,0,1], physicsClientId = physicsClient)
    
    pos = (prey_pos[0], prey_pos[1], .5)
    ors = p.getQuaternionFromEuler([0,0,prey_yaw])
    prey = p.loadURDF(file,pos,ors,globalScaling = self.agent_size, physicsClientId = physicsClient)
    x, y = cos(prey_yaw)*prey_spe, sin(prey_yaw)*prey_spe
    p.resetBaseVelocity(prey, (x,y,0),(0,0,0), physicsClientId = physicsClient)
    p.changeVisualShape(prey, -1, rgbaColor = [0,0,1,1], physicsClientId = physicsClient)
    
    return((pred, pred_pos, pred_yaw, pred_spe),
           (prey, prey_pos, prey_yaw, prey_spe), wall_ids)



if __name__ == "__main__":
  os.chdir(file_1)
  arena = Arena("empty_arena.png")
  arena.get_pair_with_difficulty(min_dif = 0, max_dif = 0, verbose = True)
  arena.get_pair_with_difficulty(min_dif = 50, max_dif = 50, verbose = True)
  arena.get_pair_with_difficulty(min_dif = 100, max_dif = 100, verbose = True)
