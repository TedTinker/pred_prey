import torch
import os

file = r"C:\Users\tedjt\Desktop\pred_prey"
os.chdir(file) 
from pred_prey_env import PredPreyEnv, add_discount





def episode(
        pred, prey, 
        GUI = False, 
        min_dif = 0, max_dif = 100, energy = 3000,
        pred_condition = 0, prey_condition = 0, 
        arena_name = "empty_arena.png"):
    
    env = PredPreyEnv(GUI = GUI, arena_name = arena_name)
    obs = env.reset(min_dif, max_dif, energy)  
    obs = torch.stack([obs[0][0], obs[1][0]])
    pred.train(), prey.train()
    done = False
    pred_hc, prey_hc  = None, None
    ang_speed_1, ang_speed_2 = None, None
    reward_list = []
    to_push_pred, to_push_prey = [], []
    while(done == False):
        with torch.no_grad():
            pred_speed_before = env.pred_spe
            pred_energy_before = env.pred_energy
            prey_speed_before = env.prey_spe
            prey_energy_before = env.prey_energy

            ang_speed_1, new_pred_hc = pred.act(obs[0], pred_speed_before, pred_energy_before, ang_speed_1, pred_hc, pred_condition)
            ang_speed_2, new_prey_hc = prey.act(obs[1], prey_speed_before, prey_energy_before, ang_speed_2, prey_hc, prey_condition)
                
            new_obs, (r_pred, r_prey), done, dist_after = env.step(ang_speed_1, ang_speed_2)
            new_obs = torch.stack([new_obs[0][0], new_obs[1][0]])     
            
            # o, s, e, a, r, no, ns, d, cutoff
            to_push_pred.append(
                (obs[0], torch.tensor(pred_speed_before), torch.tensor(pred_energy_before), ang_speed_1.cpu(), r_pred, 
                new_obs[0], torch.tensor(env.pred_spe), torch.tensor(env.pred_energy), torch.tensor(done).int(), torch.tensor(done)))
                
            to_push_prey.append(
                (obs[1], torch.tensor(prey_speed_before), torch.tensor(prey_energy_before), ang_speed_2.cpu(), r_prey, 
                new_obs[1], torch.tensor(env.prey_spe), torch.tensor(env.prey_energy), torch.tensor(done), torch.tensor(done)))
                
            reward_list.append((r_pred, r_prey))
            obs = new_obs
            pred_hc  = new_pred_hc
            prey_hc  = new_prey_hc
            
    win = dist_after < env.too_close
    env.close(forever = True)
    
    r=1
    reward_list_pred = add_discount([p[4] for p in to_push_pred], r if win else -r)
    reward_list_prey = add_discount([p[4] for p in to_push_prey], -r if win else r)
    
    to_push_pred = [(p[0], p[1], p[2], p[3], torch.tensor(r), p[5], p[6], p[7], p[8], p[9]) for p, r in zip(to_push_pred, reward_list_pred)]
    to_push_prey = [(p[0], p[1], p[2], p[3], torch.tensor(r), p[5], p[6], p[7], p[8], p[9]) for p, r in zip(to_push_prey, reward_list_prey)]
    
    for i in range(len(to_push_pred)):
        pred.episodes.push(to_push_pred[i])
        prey.episodes.push(to_push_prey[i])
    
    rewards = [(preds, preys) for preds, preys in zip(reward_list_pred, reward_list_prey)]
    return(win, rewards)

