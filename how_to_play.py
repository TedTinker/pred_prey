import torch
import os

file = r"C:\Users\tedjt\Desktop\pred_prey"
os.chdir(file) 
from pred_prey_env import add_discount





def episode(
        env, pred, prey, min_dif = 0, max_dif = 100, energy = 3000,
        pred_condition = 0, prey_condition = 0, GUI = False):
    
    (pred_rgbd, pred_speed, pred_energy, pred_action), \
    (prey_rgbd, prey_speed, prey_energy, prey_action) = env.reset(min_dif, max_dif, energy)  
    pred_hc, prey_hc  = None, None
    to_push_pred, to_push_prey = [], []
    done = False
    while(done == False):
        with torch.no_grad():

            pred_action, pred_hc = pred.act(pred_rgbd, pred_speed, pred_energy, pred_action, pred_hc, pred_condition)
            prey_action, prey_hc = prey.act(prey_rgbd, prey_speed, prey_energy, prey_action, prey_hc, prey_condition)
                
            new_obs, (r_pred, r_prey), done, dist_after = env.step(pred_action, prey_action)
            (new_pred_rgbd, new_pred_speed, new_pred_energy, new_pred_action), \
            (new_prey_rgbd, new_prey_speed, new_prey_energy, new_prey_action) = new_obs  
            
            # o, s, e, a, r, no, ns, d, cutoff
            to_push_pred.append(
                (pred_rgbd, torch.tensor(pred_speed), torch.tensor(pred_energy), pred_action.cpu(), r_pred, 
                new_pred_rgbd, torch.tensor(new_pred_speed), torch.tensor(env.pred_energy), torch.tensor(done).int(), torch.tensor(done)))
                
            to_push_prey.append(
                (prey_rgbd, torch.tensor(prey_speed), torch.tensor(prey_energy), prey_action.cpu(), r_prey, 
                new_prey_rgbd, torch.tensor(new_prey_speed), torch.tensor(env.prey_energy), torch.tensor(done), torch.tensor(done)))
                
            (pred_rgbd, pred_speed, pred_energy, pred_action),  \
            (prey_rgbd, prey_speed, prey_energy, prey_action) = \
              (new_pred_rgbd, new_pred_speed, new_pred_energy, new_pred_action), \
              (new_prey_rgbd, new_prey_speed, new_prey_energy, new_prey_action)
              
    env.close()
    win = dist_after < env.too_close
    
    r=1
    reward_list_pred = add_discount([p[4] for p in to_push_pred], r if win else -r)
    reward_list_prey = add_discount([p[4] for p in to_push_prey], -r if win else r)
    
    to_push_pred = [(p[0], p[1], p[2], p[3], torch.tensor(r), p[5], p[6], p[7], p[8], p[9]) for p, r in zip(to_push_pred, reward_list_pred)]
    to_push_prey = [(p[0], p[1], p[2], p[3], torch.tensor(r), p[5], p[6], p[7], p[8], p[9]) for p, r in zip(to_push_prey, reward_list_prey)]
    
    rewards = [(preds, preys) for preds, preys in zip(reward_list_pred, reward_list_prey)]
    return(to_push_pred, to_push_prey, win, rewards)