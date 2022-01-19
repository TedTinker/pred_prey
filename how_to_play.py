import torch
import os

file = r"C:\Users\tedjt\Desktop\pred_prey"
os.chdir(file) 
from arena import rgbd_input, too_close
from pred_prey_env import PredPreyEnv, run_with_GUI

from itertools import product
all_env_params = [_ for _ in product((False, "pred", "prey"), (False, "random", "pin"), (False, "random", "pin"))]
env_dict = {(test, pred_condition, prey_condition) : PredPreyEnv(GUI = False, test = test, pred_condition = pred_condition, prey_condition = prey_condition, arena_name = "empty_arena.png") for test, pred_condition, prey_condition in all_env_params}

def add_discount(rewards, last, GAMMA = .9):
    discounts = [last * (GAMMA**i) for i in range(len(rewards))]
    discounts.reverse()
    #for r, d in zip(rewards, discounts):
    #    print("{} + {} = {}".format(r, d, r+d))
    return([r + d for r, d in zip(rewards, discounts)])


def episode(
        pred, prey, 
        GUI = False, 
        train = "both", test = False, 
        pred_condition = False, prey_condition = False, 
        arena_name = "empty_arena.png", 
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
    
    r=1
    reward_list_pred = add_discount([p[4] for p in to_push_pred], r if win else -r)
    reward_list_prey = add_discount([p[4] for p in to_push_prey], -r if win else r)
    
    to_push_pred = [(p[0], p[1], p[2], p[3], torch.tensor(r), p[5], p[6], p[7], p[8], p[9]) for p, r in zip(to_push_pred, reward_list_pred)]
    to_push_prey = [(p[0], p[1], p[2], p[3], torch.tensor(r), p[5], p[6], p[7], p[8], p[9]) for p, r in zip(to_push_prey, reward_list_prey)]
    
    for i in range(len(to_push_pred)):
        pred.episodes.push(to_push_pred[i], pred)
        prey.episodes.push(to_push_prey[i], prey)
    
    rewards = [(preds, preys) for preds, preys in zip(reward_list_pred, reward_list_prey)]
    return(win, rewards)

