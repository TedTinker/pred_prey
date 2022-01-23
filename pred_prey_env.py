import torch
import numpy as np
import pybullet as p
from math import degrees, pi, cos, sin, sqrt


import os
file = r"C:\Users\tedjt\Desktop\pred_prey"
os.chdir(file) 
from arena import get_physics, Arena





  

### The arena environment
    
import gym # This is not deeply used yet. 
from matplotlib import pyplot as plt
from torchvision.transforms.functional import resize


# How are agents rewarded/punished each step? 
dist_d      = .1     # Based on distance
closer_d    = 10     # Based on distance closer
col_d       = .1     # Based on collision with walls

def get_reward(agent, dist, closer, collision, printing = False):
    r_dist = (1/dist - .7) * dist_d if agent == "pred" else (-1/dist + .7) * dist_d
    r_closer = closer * closer_d if agent == "pred" else -closer * closer_d
    r_col    = -col_d if collision else 0
    r = r_dist + r_closer + r_col
    if(printing):
        print("\n{} reward {}:\n\t{} from dist,\n\t{} from dist closer,\n\t{} from collision.".format(
            agent, round(r,3), round(r_dist,3), round(r_closer,3), round(r_col, 3)))
    return(r)

def add_discount(rewards, last, GAMMA = .9):
    discounts = [last * (GAMMA**i) for i in range(len(rewards))]
    discounts.reverse()
    #for r, d in zip(rewards, discounts):
    #    print("{} + {} = {}".format(r, d, r+d))
    return([r + d for r, d in zip(rewards, discounts)])



# Made an environment! 
class PredPreyEnv():   
  def __init__(
      self, arena_name = "arena.png", GUI = False, image_size = 16, 
      min_speed = 10, max_speed = 50, max_angle_change = pi/2, too_close = .6):
    
    self.image_size = image_size; self.min_speed = min_speed; self.max_speed = max_speed
    self.max_angle_change = max_angle_change; self.too_close = too_close
    self.arena = Arena(arena_name); self.GUI = GUI
    self.already_constructed = False
    self.physicsClient = get_physics(self.GUI, self.arena.w, self.arena.h)
    self.pred, self.prey = None, None
    self.steps, self.resets = 0, 0

  def close(self, forever = False):
    if(self.pred != None):
      p.removeBody(self.pred, physicsClientId = self.physicsClient)
      p.removeBody(self.prey, physicsClientId = self.physicsClient)
    if(self.resets % 100 == 99 and self.GUI and not forever):
      p.disconnect(self.physicsClient)
      self.already_constructed = False
      self.physicsClient = get_physics(self.GUI, self.arena.w, self.arena.h)
    if(forever):
      p.disconnect(self.physicsClient)  

  def reset(self, min_dif = 0, max_dif = 100, energy = 3000):
    self.close()
    self.resets += 1; self.steps = 0
    (self.pred, self.pred_pos, self.pred_yaw, self.pred_spe),\
    (self.prey, self.prey_pos, self.prey_yaw, self.prey_spe), wall_ids = \
        self.arena.start_arena(self.physicsClient, self.already_constructed, min_dif, max_dif, self.min_speed)
    if(self.already_constructed == False): self.wall_ids = wall_ids; self.already_constructed = True
    self.pred_energy, self.prey_energy = energy, energy
    self.pred_action, self.prey_action = torch.tensor([0,0]), torch.tensor([0,0])
    return(self.get_obs())

  def get_obs(self, agent_name = "both"):
    if(agent_name == "both"): return(self.get_obs("pred"), self.get_obs("prey"))
    elif(agent_name == "pred"): 
      yaw, pos = self.pred_yaw, self.pred_pos
      speed, energy, action = self.pred_spe, self.pred_energy, self.pred_action
    elif(agent_name == "prey"): 
      yaw, pos = self.prey_yaw, self.prey_pos
      speed, energy, action = self.prey_spe, self.prey_energy, self.prey_action
    else: print("Not a good agent."); return

    x, y = cos(yaw), sin(yaw)
    view_matrix = p.computeViewMatrix(
      cameraEyePosition = [pos[0], pos[1], .5], 
      cameraTargetPosition = [pos[0] + x, pos[1] + y, .5], 
      cameraUpVector = [0, 0, 1], physicsClientId = self.physicsClient)
    proj_matrix = p.computeProjectionMatrixFOV(
      fov = 90, aspect = 1, nearVal = 0.2, 
      farVal = 10, physicsClientId = self.physicsClient)
    _, _, rgba, depth, _ = p.getCameraImage(
      width=64, height=64,
      projectionMatrix=proj_matrix, viewMatrix=view_matrix, 
      physicsClientId = self.physicsClient)
    rgb = np.divide(rgba[:,:,:-1], 255) * 2 - 1
    d = np.expand_dims(depth, axis=-1)
    rgbd = np.concatenate([rgb, d], axis = -1)
    rgbd = torch.from_numpy(rgbd).float()
    rgbd = resize(rgbd.permute(-1,0,1), (self.image_size, self.image_size)).permute(1,2,0)
    return(rgbd, speed, energy, action)

  def render(self, agent_name = "both"):

    if(agent_name == "both"): 
      self.render("pred"); self.render("prey"); self.render("above")
      return()

    if(agent_name != "above"):
      rgbd, _, _, _ = self.get_obs(agent_name)
      rgb = (rgbd[:,:,0:3] + 1)/2
      plt.figure(figsize = (5,5))
      plt.imshow(rgb)
      plt.title("{}'s view".format(agent_name))
      plt.show()
      plt.close()
      plt.ioff()
      return()
    
    x_1, y_1 = self.pred_pos[0], self.pred_pos[1]
    x_2, y_2 = self.prey_pos[0], self.prey_pos[1]
    x = (x_1 + x_2)/2
    y = (y_1 + y_2)/2
    
    view_matrix = p.computeViewMatrix(
        cameraEyePosition = [x, y, self.agent_dist() + 1], 
        cameraTargetPosition = [x, y, 0], 
        cameraUpVector = [1, 0, 0], physicsClientId = self.physicsClient)
    proj_matrix = p.computeProjectionMatrixFOV(
        fov = 90, aspect = 1, nearVal = 0.001, 
        farVal = self.agent_dist() + 2, physicsClientId = self.physicsClient)
    _, _, rgba, _, _ = p.getCameraImage(
        width=64, height=64,
        projectionMatrix=proj_matrix, viewMatrix=view_matrix, 
        physicsClientId = self.physicsClient)
    rgb = rgba[:,:,:-1]
    rgb = np.divide(rgb,255)
    plt.figure(figsize = (10,10))
    plt.imshow(rgb)
    plt.title("View from above")
    plt.show()
    plt.close()
    plt.ioff()

  def agent_dist(self):
    x = self.pred_pos[0] - self.prey_pos[0]
    y = self.pred_pos[1] - self.prey_pos[1]
    return((x**2 + y**2)**.5)

  def get_position_and_speed(self, agent):
    pos, _ = p.getBasePositionAndOrientation(agent, physicsClientId = self.physicsClient)
    (x, y, _), _ = p.getBaseVelocity(agent, physicsClientId = self.physicsClient)
    speed = (x**2 + y**2)**.5
    return(pos, speed)

  def collisions(self, agent = "both"):
    if(agent == "both"): return(self.collisions(self.pred), self.collisions(self.prey))
    col = False
    for wall in self.wall_ids:
      if len(p.getContactPoints(agent, wall, physicsClientId = self.physicsClient)) > 0:
        col = True
    return(col)

  def change_velocity(self, agent_name, yaw_change, speed, printing = False):
    if(agent_name == "pred"): agent, old_yaw, old_speed, pos = self.pred, self.pred_yaw, self.pred_spe, self.pred_pos
    if(agent_name == "prey"): agent, old_yaw, old_speed, pos = self.prey, self.prey_yaw, self.pred_spe, self.prey_pos
    
    new_yaw = old_yaw + yaw_change
    new_yaw %= 2*pi
    orn = p.getQuaternionFromEuler([0,0,new_yaw])
    p.resetBasePositionAndOrientation(agent,(pos[0], pos[1], .5), orn, physicsClientId = self.physicsClient)
    
    x = cos(new_yaw)*speed
    y = sin(new_yaw)*speed
    p.resetBaseVelocity(agent, (x,y,0), (0,0,0), physicsClientId = self.physicsClient)
    
    if(agent_name == "pred"): self.pred_yaw = new_yaw; self.pred_spe = speed
    if(agent_name == "prey"): self.prey_yaw = new_yaw; self.prey_spe = speed
    
    if(printing):
      print("\n{}:\nOld yaw:\t{}\nChange:\t\t{}\nNew yaw:\t{}".format(
        agent_name, round(degrees(old_yaw)), round(degrees(yaw_change)), round(degrees(new_yaw))))
      print("Old speed:\t{}\nNew speed:\t{}".format(old_speed, speed))

  def unnormalize(self, action): # from (-1, 1) to (min, max)
    yaw = action[0].clip(-1,1) * self.max_angle_change
    spe = self.min_speed + ((action[1].clip(-1,1) + 1)/2) * (self.max_speed - self.min_speed)
    return(yaw, spe)
  
  def step(self, pred_action, prey_action):
    self.pred_action, self.prey_action = pred_action, prey_action
    self.steps += 1
    yaw_1, spe_1 = self.unnormalize(pred_action)
    yaw_2, spe_2 = self.unnormalize(prey_action)
    if(self.pred_energy <= 0): spe_1 = 0
    if(self.prey_energy <= 0): spe_2 = 0
    self.change_velocity("pred", yaw_1, spe_1)
    self.change_velocity("prey", yaw_2, spe_2)

    dist_before = self.agent_dist()
    p.stepSimulation(physicsClientId = self.physicsClient)
    self.pred_energy -= spe_1
    self.prey_energy -= spe_2
    self.pred_pos, self.pred_spe = self.get_position_and_speed(self.pred)
    self.prey_pos, self.prey_spe = self.get_position_and_speed(self.prey)
    dist_after = self.agent_dist()
    dist_closer = dist_before - dist_after

    pred_collision, prey_collision = self.collisions()
    rewards = (
        get_reward("pred", dist_after, dist_closer, pred_collision), 
        get_reward("prey", dist_after, dist_closer, prey_collision))
    observations = self.get_obs(agent_name = "both")
    done = True if dist_after <= self.too_close or self.pred_energy <= 0 else False
    return(observations, rewards, done, dist_after)
    

    


from torch.distributions import Normal
from tqdm import tqdm


    
    
    

if __name__ == "__main__":
    env = PredPreyEnv(arena_name = "big_arena.png")
    env.reset()   
    env.render() 
    env.close(forever = True)
