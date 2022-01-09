### How to get arena from image
import numpy as np
import cv2
from itertools import product
from math import pi, sin, cos

file_1 = r"C:\Users\tedjt\Desktop\pred_prey"
import os
os.chdir(file_1)
from utils import file_2




def get_arena(arena = "arena.png"):

    # Get PNG image.
    os.chdir(file_1) 
    arena = cv2.imread(arena)
    os.chdir(file_2) 
    w,h,_ = arena.shape
    
    # Totally white (255, 255, 255) pixels are open space. 
    open_spots = [(x,y) for x, y in product(range(w), range(h)) if arena[x,y].tolist() == [255, 255, 255]]
    
    # Starting close to the prey is easy for the predator.
    # Starting far from the predator is easy for the prey. 
    distances = []
    open_spot_combinations = [(spot_1, spot_2) for spot_1, spot_2 in product(open_spots, open_spots) if spot_1 != spot_2]
    for spot_1, spot_2 in open_spot_combinations:
        x, y = spot_1[0] - spot_2[0], spot_1[1] - spot_2[1]
        distances.append((x**2 + y**2)**.5)
    hard_dist_for_pred = np.percentile([d for d in distances if d != 0], 75)
    hard_dist_for_prey = np.percentile([d for d in distances if d != 0], 33)
    hard_spot_pairs_for_pred, hard_spot_pairs_for_prey = [], []
    for (spot_1, spot_2), dist in zip(open_spot_combinations, distances):
        if dist > hard_dist_for_pred:
            hard_spot_pairs_for_pred.append((spot_1, spot_2))
        if dist < hard_dist_for_prey and spot_1 != spot_2:
            hard_spot_pairs_for_prey.append((spot_1, spot_2))
            
    return(arena, open_spots, hard_spot_pairs_for_pred, hard_spot_pairs_for_prey, w, h)



if __name__ == "__main__":
    arena, open_spots, hard_spot_pairs_for_pred, hard_spot_pairs_for_prey, w, h = get_arena(arena = "bigger_arena.png")
    print("\nArena shape:", arena.shape)
    print("\nOpen spots:")
    for spot in open_spots:
        print("\t", spot)
    print("\nHard spots for pred:")
    for spots in hard_spot_pairs_for_pred:
        print("\t", spots)
    print("\nHard spots for prey:")
    for spots in hard_spot_pairs_for_prey:
        print("\t", spots)

    
    
    
### How to make arena in pybullet and place agents

agent_size = .5             # How big are agents?
image_size = 16             # How big are agent observations?
rgbd_input = (image_size, image_size, 4)
max_velocity = 30           # Maximum velocity?
max_velocity_change = 10    # How much can an agent change velocity?
max_angle_change = pi / 2   # How much can an agent change angle?
too_close = .6              # How close must the predator be to win?

import pybullet as p
import pybullet_data

# Return a physics client. If using GUI, move camera above arena looking down. 
def get_physics(GUI = False, name = "arena.png"):
    if(GUI):
        _, _, _, _, w, h = get_arena(name)
        physicsClient = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(1,90,-89,(w/2,h/2,w), physicsClientId = physicsClient)
    else:   physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId = physicsClient)
    return(physicsClient)

# We place agents at the center of arena-spots. Wiggle them around a little. 
import random
def wiggle(x, y):
    #x_, y_ = random.uniform(-agent_size/2, agent_size/2), random.uniform(-agent_size/2, agent_size/2)
    #x += x_; y += y_
    return([x, y, .5])

# Make arena! 
def start_arena(test = False, physicsClient = 0, name = "arena.png", already_constructed = False):
    arena, open_spots, hard_spot_pairs_for_pred, hard_spot_pairs_for_prey, w, h = get_arena(name)
    wall_ids = []
    # Construct with cubes the same color as the image-pixels.
    if(not already_constructed):
        for loc in ((x,y) for x in range(w) for y in range(h)):
            if(not (arena[loc] == [255]).all()):
                pos = [loc[0],loc[1],.5]
                ors = p.getQuaternionFromEuler([0,0,0])
                cube = p.loadURDF("cube.urdf",pos,ors, useFixedBase = True, physicsClientId = physicsClient)
                color = arena[loc][::-1] / 255
                color = np.append(color, 1)
                p.changeVisualShape(cube, -1, rgbaColor=color, physicsClientId = physicsClient)
                wall_ids.append(cube)
                
    # Pick random positions and angles for pred, prey. 
    if(  test == "pred"):   pos_pred, pos_prey = random.sample(hard_spot_pairs_for_pred,1)[0]
    elif(test == "prey"):   pos_pred, pos_prey = random.sample(hard_spot_pairs_for_prey,1)[0]
    else:                   pos_pred, pos_prey = random.sample(open_spots,2)
    yaw_pred, yaw_prey = random.uniform(0, 2*pi), random.uniform(0, 2*pi)
    vel_pred, vel_prey  = random.uniform(0,max_velocity), random.uniform(0,max_velocity)
    
    os.chdir(file_1)
    file = "sphere2red.urdf"
    #file = "pred_prey.urdf" # How can I make my own robot-shape? 
    
    # Place pred.
    pos = wiggle(pos_pred[0], pos_pred[1])
    ors = p.getQuaternionFromEuler([0,0,yaw_pred])
    pred = p.loadURDF(file,pos,ors,globalScaling = agent_size, physicsClientId = physicsClient)
    x, y = cos(yaw_pred)*vel_pred, sin(yaw_pred)*vel_pred
    p.resetBaseVelocity(pred, (x,y,0),(0,0,0), physicsClientId = physicsClient)
    p.changeVisualShape(pred, -1, rgbaColor = [1,0,0,1], physicsClientId = physicsClient)
    
    # Place prey.
    pos = wiggle(pos_prey[0], pos_prey[1])
    ors = p.getQuaternionFromEuler([0,0,yaw_prey])
    prey = p.loadURDF(file,pos,ors,globalScaling = agent_size, physicsClientId = physicsClient)
    x, y = cos(yaw_prey)*vel_prey, sin(yaw_prey)*vel_prey
    p.resetBaseVelocity(prey, (x,y,0),(0,0,0), physicsClientId = physicsClient)
    p.changeVisualShape(prey, -1, rgbaColor = [0,0,1,1], physicsClientId = physicsClient)
    
    os.chdir(file_2)
    return(pred, prey, pos_pred, pos_prey, yaw_pred, yaw_prey, vel_pred, vel_prey, wall_ids)