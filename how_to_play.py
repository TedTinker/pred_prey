import torch

from utils import get_input, plot_rewards
from pred_prey_env import add_discount





def episode(
        env, pred, prey, min_dif = 0, max_dif = 100, GUI = False):
    
    (pred_rgbd, pred_speed, pred_energy, pred_action), \
    (prey_rgbd, prey_speed, prey_energy, prey_action) = env.reset(min_dif, max_dif)  
    pred_hc, prey_hc  = None, None
    to_push_pred, to_push_prey = [], []
    done = False
    while(done == False):
        with torch.no_grad():
            if(env.para.pred_condition == "by_hand"):  pred_action = move_by_hand(env, "pred")
            else:
                pred_action, pred_hc = pred.act(pred_rgbd, pred_speed, pred_energy, pred_action, pred_hc, env.para.pred_condition)
            if(env.para.prey_condition == "by_hand"):  prey_action = move_by_hand(env, "prey")
            else:
                prey_action, prey_hc = prey.act(prey_rgbd, prey_speed, prey_energy, prey_action, prey_hc, env.para.prey_condition)
                
            new_obs, (r_pred, r_prey), done, win = env.step(pred_action, prey_action)
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
    
    r=1
    reward_list_pred = add_discount([p[4] for p in to_push_pred], r if win else -r)
    reward_list_prey = add_discount([p[4] for p in to_push_prey], -r if win else r)
    
    to_push_pred = [(p[0], p[1], p[2], p[3], torch.tensor(r), p[5], p[6], p[7], p[8], p[9]) for p, r in zip(to_push_pred, reward_list_pred)]
    to_push_prey = [(p[0], p[1], p[2], p[3], torch.tensor(r), p[5], p[6], p[7], p[8], p[9]) for p, r in zip(to_push_prey, reward_list_prey)]
    
    rewards = [(preds, preys) for preds, preys in zip(reward_list_pred, reward_list_prey)]
    return(to_push_pred, to_push_prey, win, rewards)







# How to play one move by hand.
def move_by_hand(env, agent_name):
  print("Observations:")
  if(agent_name == "pred"):    agent, speed, energy = env.pred, env.pred_spe, env.pred_energy
  else:                        agent, speed, energy = env.prey, env.prey_spe, env.prey_energy
  env.render(agent_name)
  print("Speed: {}, Energy: {}".format(speed, energy))
  action = None
  while(action == None):
    yaw   = input("\nChange in yaw? Min {}, max {}.\n".format(-1, 1))
    if(yaw == ""): yaw = 0
    speed = input("\nChange in speed? Min {}, max {}.\n".format(-1, 1))
    if(speed == ""): speed = 1
    try:
      yaw = float(yaw); speed = float(speed)
      assert(yaw >= -1 and yaw <= 1 and \
             speed >= -1 and speed <= 1)
      action = torch.tensor([yaw, speed])
    except: print("'{}, {}' not a valid move. Try again.".format(yaw, speed))
  return(action)














# How to play a whole episode by hand.
def hand_episodes(env, pred, prey, energy = None, pred_condition = None, prey_condition = None):
  play = get_input("Play a game by hand?", ["y", "n"], default = "n")
  if(play == "n"): return(None, None)
  while(play == "y"):
    if(energy == None):
        energy = int(get_input("Energy?", ["2000", "3000", "4000"], default = "3000"))
    min_dif = get_input("Minimum pred-difficulty?", [str(i) for i in range(101)])
    max_dif = get_input("Maximum pred-difficulty?", [str(i) for i in range(101)])
    if(pred_condition == None):
        pred_condition = get_input("Predator condition?", ["by_hand", "pin", "random", "none"], default = 1)
    if(prey_condition == None):
        prey_condition = get_input("Prey condition?", ["by_hand", "pin", "random", "none"], default = 2)
    to_push_pred, to_push_prey, pred_win, rewards = episode(
      env, pred, prey, min_dif = int(min_dif), max_dif = int(max_dif), energy = energy, 
      pred_condition = pred_condition, prey_condition = prey_condition)
    print("\nGAME OVER! {}\n".format("Predator wins." if pred_win else "Prey wins."))
    plot_rewards(rewards)
    max_repeats = 32
    repeat = get_input("Remember this how many times?", [str(i) for i in range(1,max_repeats+1)])
    for _ in range(int(repeat)):
      for i in range(len(to_push_pred)):
        pred.episodes.push(to_push_pred[i])
        prey.episodes.push(to_push_prey[i])
    play = get_input("Play another game by hand?", ["y", "n"], default = "n")
  return(pred.episodes, prey.episodes)