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
    
    pred.train(), prey.train()
    env = PredPreyEnv(GUI = GUI, arena_name = arena_name)
    (pred_rgbd, pred_speed, pred_energy, pred_action), \
    (prey_rgbd, prey_speed, prey_energy, prey_action) = env.reset(min_dif, max_dif, energy)  
    pred_hc, prey_hc  = None, None
    ang_speed_1, ang_speed_2 = None, None
    to_push_pred, to_push_prey = [], []
    done = False
    while(done == False):
        with torch.no_grad():
            pred_speed_before = env.pred_spe
            pred_energy_before = env.pred_energy
            prey_speed_before = env.prey_spe
            prey_energy_before = env.prey_energy

            ang_speed_1, pred_hc = pred.act(pred_rgbd, pred_speed_before, pred_energy_before, ang_speed_1, pred_hc, pred_condition)
            ang_speed_2, prey_hc = prey.act(prey_rgbd, prey_speed_before, prey_energy_before, ang_speed_2, prey_hc, prey_condition)
                
            new_obs, (r_pred, r_prey), done, dist_after = env.step(ang_speed_1, ang_speed_2)
            (new_pred_rgbd, new_pred_speed, new_pred_energy, new_pred_action), \
            (new_prey_rgbd, new_prey_speed, new_prey_energy, new_prey_action) = new_obs  
            
            # o, s, e, a, r, no, ns, d, cutoff
            to_push_pred.append(
                (pred_rgbd, torch.tensor(pred_speed_before), torch.tensor(pred_energy_before), ang_speed_1.cpu(), r_pred, 
                new_pred_rgbd, torch.tensor(env.pred_spe), torch.tensor(env.pred_energy), torch.tensor(done).int(), torch.tensor(done)))
                
            to_push_prey.append(
                (prey_rgbd, torch.tensor(prey_speed_before), torch.tensor(prey_energy_before), ang_speed_2.cpu(), r_prey, 
                new_prey_rgbd, torch.tensor(env.prey_spe), torch.tensor(env.prey_energy), torch.tensor(done), torch.tensor(done)))
                
            (pred_rgbd, pred_speed, pred_energy, pred_action), \
            (prey_rgbd, prey_speed, prey_energy, prey_action) = \
              (new_pred_rgbd, new_pred_speed, new_pred_energy, new_pred_action), \
              (new_prey_rgbd, new_prey_speed, new_prey_energy, new_prey_action)
    env.close(forever = True)
    win = dist_after < env.too_close
    
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







"""
def episode(
    env, pred, prey, energy, min_dif = 0, max_dif = 100,
    pred_condition = 0, prey_condition = 0, GUI = False, render = False):
  (pred_rgbd, pred_speed, pred_energy, pred_action), \
  (prey_rgbd, prey_speed, prey_energy, prey_action) = env.reset(min_dif, max_dif, energy)  
  pred_hc, prey_hc = None, None
  to_push_pred, to_push_prey = [], []
  done = False
  while(done == False):
    with torch.no_grad():
      if(render): 
        clear_output()
        env.render("above")

      if(pred_condition == "by_hand"):  current_pred_action = move_by_hand(env, "pred")
      else:
        current_pred_action, pred_hc = pred.act(pred_rgbd, pred_speed, pred_energy, pred_action, pred_hc, pred_condition)
        current_pred_action = current_pred_action.cpu()

      if(prey_condition == "by_hand"):  current_prey_action = move_by_hand(env, "prey")
      else:
        current_prey_action, prey_hc = prey.act(prey_rgbd, prey_speed, prey_energy, prey_action, prey_hc, prey_condition)
        current_prey_action = current_prey_action.cpu()

      new_obs, (pred_reward, prey_reward), done, dist_after = env.step(current_pred_action, current_prey_action)
      (new_pred_rgbd, new_pred_speed, new_pred_energy, new_pred_action), \
      (new_prey_rgbd, new_prey_speed, new_prey_energy, new_prey_action) = new_obs
                  
      # o, s, e, a, r, no, ns, ne, d, cutoff
      to_push_pred.append(
          (pred_rgbd, pred_speed, pred_energy, pred_action, pred_reward, 
          new_pred_rgbd, new_pred_speed, new_pred_energy, torch.tensor(done).int(), torch.tensor(done)))
          
      to_push_prey.append(
          (prey_rgbd, prey_speed, prey_energy, prey_action, prey_reward, 
          new_prey_rgbd, new_prey_speed, new_prey_energy, torch.tensor(done).int(), torch.tensor(done)))
          
      (pred_rgbd, pred_speed, pred_energy, pred_action), \
      (prey_rgbd, prey_speed, prey_energy, prey_action) = \
        (new_pred_rgbd, new_pred_speed, new_pred_energy, new_pred_action), \
        (new_prey_rgbd, new_prey_speed, new_prey_energy, new_prey_action)
  env.close(forever=False)
    
  pred_win = dist_after < env.too_close
  r=1
  reward_list_pred = add_discount([p[4] for p in to_push_pred], r if pred_win else -r)
  reward_list_prey = add_discount([p[4] for p in to_push_prey], -r if pred_win else r)
  
  to_push_pred = [(p[0], p[1], p[2], p[3], torch.tensor(r), p[5], p[6], p[7], p[8], p[9]) for p, r in zip(to_push_pred, reward_list_pred)]
  to_push_prey = [(p[0], p[1], p[2], p[3], torch.tensor(r), p[5], p[6], p[7], p[8], p[9]) for p, r in zip(to_push_prey, reward_list_prey)]
  
  rewards = [(preds, preys) for preds, preys in zip(reward_list_pred, reward_list_prey)]

  return(to_push_pred, to_push_prey, pred_win, rewards)
"""



