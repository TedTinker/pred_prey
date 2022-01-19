import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from torch.distributions import Normal
import torch.nn.functional as f
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import degrees
from copy import deepcopy
import keyboard

file = r"C:\Users\tedjt\Desktop\pred_prey"
os.chdir(file) 
from utils import file_1, file_2, device, get_free_mem, delete_these, \
    duration, save_plot, plot_wins, plot_losses, plot_rewards, \
    save_pred_prey, load_pred_prey
from arena import rgbd_input, too_close
from pred_prey_env import PredPreyEnv, run_with_GUI
from how_to_play import episode
from rtd3 import RecurrentTD3
os.chdir(r"C:\Users\tedjt")

pred = RecurrentTD3()
prey = RecurrentTD3()

arena_name = "empty_arena.png"

folder = "default"



e = 0





    
    

max_len = 300
per_epoch = 33



  
def get_rolling_average(wins, roll = 100):
  if(len(wins) < roll):
    return(sum(wins)/len(wins))
  return(sum(wins[-roll:])/roll)

win_easy = []
win_med = []
win_hard= []
win_easy_rolled = []
win_med_rolled = []
win_hard_rolled = []
losses = np.array([[None, None, None, None, None, None]])

starting_explorations = 1, 1
explorations = starting_explorations

while(e <= 100):
    e += 1
    
    print()
    print("Explorations:", explorations)
    for i in tqdm(range(per_epoch), "Epoch {}".format(e)):
        explorations = explorations[0]*.99, explorations[1]*.99
        win, rewards = episode(pred = pred, prey = prey,
                              GUI = keyboard.is_pressed('q'), 
                              train = "pred", test = "prey", 
                              pred_condition = False, prey_condition = "pin", arena_name = arena_name,
                              pred_exploration=explorations[0], prey_exploration=explorations[1])
        win_easy.append(win)
        win_easy_rolled.append(get_rolling_average(win_easy))
        if(keyboard.is_pressed('q')):
            plot_rewards(rewards)
        
        win, rewards = episode(pred = pred, prey = prey,
                              GUI = keyboard.is_pressed('q'), 
                              train = "pred", test = False, 
                              pred_condition = False, prey_condition = "pin", arena_name = arena_name,
                              pred_exploration=explorations[0], prey_exploration=explorations[1])
        win_med.append(win)
        win_med_rolled.append(get_rolling_average(win_med))
        if(keyboard.is_pressed('q')):
            plot_rewards(rewards)

        win, rewards = episode(pred = pred, prey = prey,
                              GUI = keyboard.is_pressed('q'), 
                              train = "pred", test = "pred", 
                              pred_condition = False, prey_condition = "pin", arena_name = arena_name,
                              pred_exploration=explorations[0], prey_exploration=explorations[1])
        win_hard.append(win)
        win_hard_rolled.append(get_rolling_average(win_hard))
        if(keyboard.is_pressed('q')):
            plot_rewards(rewards)
        
        pred_losses = pred.update_networks(batch_size = 16, iterations = 4)
        #prey_losses = prey.update_networks(batch_size = 16, iterations = 4)
        losses = np.concatenate([losses, np.concatenate([pred_losses, pred_losses], axis = 1)])
        if(keyboard.is_pressed('q')):
            print("Explorations:", explorations)
            plot_losses(losses, 500)
        
    print("Total duration: {}.".format(duration()))
    plot_wins(win_easy_rolled, win_med_rolled, win_hard_rolled, name = "wins_{}.png".format(e))
    save_pred_prey(pred, prey, "_{}_epochs".format(e), folder)

    if(win_easy_rolled[-1] >= .9 and win_med_rolled[-1] >= .9 and win_hard_rolled[-1] >= .9): 
        if(explorations[0] > .01 and explorations[0] < .05): explorations = 0, 0
        elif(explorations[0] > .01): explorations = explorations[0]/2, explorations[1]/2
        else: plot_losses(losses); break
    if(     (e >= 10 and win_hard_rolled[-1] < .1) or
            (e >= 15 and win_hard_rolled[-1] < .2)):
        print("\n\nNot great. Starting again!\n\n")
        e = 0
        explorations = starting_explorations
        win_easy = []
        win_med = []
        win_hard= []
        win_easy_rolled = []
        win_med_rolled = []
        win_hard_rolled = []
        losses = np.array([[None, None, None, None, None, None]])
        pred = RecurrentTD3()
        prey = RecurrentTD3()


        
    save_pred_prey(pred, prey, "_{}_epochs".format(e), folder)
    print()
    
    
    
    
    
    
    


run_with_GUI(test = "pred", pred = pred, prey = prey, 
             pred_condition = None, prey_condition = "pin", 
             GUI = True, episodes = 100, arena_name = arena_name, render = False)

