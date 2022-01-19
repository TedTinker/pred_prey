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
from rtd3 import RecurrentTD3
os.chdir(r"C:\Users\tedjt")

pred = RecurrentTD3()
prey = RecurrentTD3()

arena_name = "empty_arena.png"

folder = "empty_prey_pinned"



e = 0



from itertools import product
all_env_params = [_ for _ in product((False, "pred", "prey"), (False, "random", "pin"), (False, "random", "pin"))]
env_dict = {(test, pred_condition, prey_condition) : PredPreyEnv(GUI = False, test = test, pred_condition = pred_condition, prey_condition = prey_condition, arena_name = arena_name) for test, pred_condition, prey_condition in all_env_params}

def add_discount(rewards, last, GAMMA = .9):
    discounts = [last * (GAMMA**i) for i in range(len(rewards))]
    discounts.reverse()
    #for r, d in zip(rewards, discounts):
    #    print("{} + {} = {}".format(r, d, r+d))
    return([r + d for r, d in zip(rewards, discounts)])


def session(
        GUI = False, 
        train = "both", test = False, 
        pred = pred, prey = prey, 
        pred_condition = False, prey_condition = False, 
        arena_name = arena_name, plotting_rewards = False,
        pred_exploration = 0, prey_exploration = 0):
    
    if(GUI):    env = PredPreyEnv(GUI = True, test = test, pred_condition = pred_condition, prey_condition = prey_condition, arena_name = arena_name)
    else:       env = env_dict[(test, pred_condition, prey_condition)]
    obs = env.reset()  
    pred.train(), prey.train()
    done = False
    pred_hc, prey_hc  = None, None
    ang_speed_1, ang_speed_2 = None, None
    reward_list = []
    to_push_pred, to_push_prey = [], []
    while(done == False):
        with torch.no_grad():
            pred_speed_before = env.speed_pred
            pred_energy_before = env.pred_energy
            prey_speed_before = env.speed_prey
            prey_energy_before = env.prey_energy

            ang_speed_1, new_pred_hc = pred.get_action(obs[0], pred_speed_before, pred_energy_before, ang_speed_1, pred_hc, pred_exploration)
            ang_speed_2, new_prey_hc = prey.get_action(obs[1], prey_speed_before, prey_energy_before, ang_speed_2, prey_hc, prey_exploration)
                
            new_obs, (r_pred, r_prey), done, dist_after = env.step(ang_speed_1, ang_speed_2)
                        
            # o, s, e, a, r, no, ns, d, cutoff
            to_push_pred.append(
                (obs[0].cpu(), torch.tensor(pred_speed_before), torch.tensor(pred_energy_before), ang_speed_1.cpu(), r_pred, 
                new_obs[0].cpu(), torch.tensor(env.speed_pred), torch.tensor(env.pred_energy), torch.tensor(done).int(), torch.tensor(done)))
                
            to_push_prey.append(
                (obs[1].cpu(), torch.tensor(prey_speed_before), torch.tensor(prey_energy_before), ang_speed_2.cpu(), r_prey, 
                new_obs[1].cpu(), torch.tensor(env.speed_prey), torch.tensor(env.prey_energy), torch.tensor(done), torch.tensor(done)))
                
            reward_list.append((r_pred, r_prey))
            obs = new_obs
            pred_hc  = new_pred_hc
            prey_hc  = new_prey_hc
        
    env.close(forever = GUI)
    win = dist_after < too_close
    game_length = len(to_push_pred)
    
    r=1
    reward_list_pred = add_discount([p[4] for p in to_push_pred], r if win else -r)
    reward_list_prey = add_discount([p[4] for p in to_push_prey], -r if win else r)
    
    to_push_pred = [(p[0], p[1], p[2], p[3], torch.tensor(r), p[5], p[6], p[7], p[8], p[9]) for p, r in zip(to_push_pred, reward_list_pred)]
    to_push_prey = [(p[0], p[1], p[2], p[3], torch.tensor(r), p[5], p[6], p[7], p[8], p[9]) for p, r in zip(to_push_prey, reward_list_prey)]
    
    for i in range(len(to_push_pred)):
        pred.episodes.push(to_push_pred[i], pred)
        prey.episodes.push(to_push_prey[i], prey)
    
    if(plotting_rewards):
        plot_rewards([(preds, preys) for preds, preys in zip(reward_list_pred, reward_list_prey)])
    return(win, game_length)





    
    

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
        win, length = session(GUI = keyboard.is_pressed('q'), plotting_rewards = keyboard.is_pressed('q'),
                              train = "pred", test = "prey", 
                              pred = pred, prey = prey,
                              pred_condition = False, prey_condition = "pin", arena_name = arena_name,
                              pred_exploration=explorations[0], prey_exploration=explorations[1])
        win_easy.append(win)
        win_easy_rolled.append(get_rolling_average(win_easy))
        
        win, length = session(GUI = keyboard.is_pressed('q'), plotting_rewards = keyboard.is_pressed('q'),
                              train = "pred", test = False, 
                              pred = pred, prey = prey, 
                              pred_condition = False, prey_condition = "pin", arena_name = arena_name,
                              pred_exploration=explorations[0], prey_exploration=explorations[1])
        win_med.append(win)
        win_med_rolled.append(get_rolling_average(win_med))

        win, length = session(GUI = keyboard.is_pressed('q'), plotting_rewards = keyboard.is_pressed('q'),
                              train = "pred", test = "pred", 
                              pred = pred, prey = prey, 
                              pred_condition = False, prey_condition = "pin", arena_name = arena_name,
                              pred_exploration=explorations[0], prey_exploration=explorations[1])
        win_hard.append(win)
        win_hard_rolled.append(get_rolling_average(win_hard))
        pred_losses = pred.update_networks(batch_size = 16, iterations = 4)
        #prey_losses = prey.update_networks(batch_size = 16, iterations = 4)
        losses = np.concatenate([losses, np.concatenate([pred_losses, pred_losses], axis = 1)])
        if(keyboard.is_pressed('q')):
            print("Explorations:", explorations)
            plot_losses(losses, 500)
        
    print("Total duration: {}.".format(duration()))
    plot_wins(win_easy_rolled, win_med_rolled, win_hard_rolled, name = "wins_{}.png".format(e))
    save_pred_prey(pred, prey, folder, "_{}_epochs".format(e))

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


        
    save_pred_prey(pred, prey, folder, "_{}_epochs".format(e))
    print()
    
    
    
    
    
    
    


run_with_GUI(test = "pred", pred = pred, prey = prey, 
             pred_condition = None, prey_condition = "pin", 
             GUI = True, episodes = 100, arena_name = arena_name, render = False)

