import torch
import numpy as np
import pybullet as p
import random
from math import degrees, pi, cos, sin, sqrt


import os
file = r"C:\Users\tedjt\Desktop\pred_prey"
os.chdir(file) 
from utils import device, file_1, file_2, duration
from arena import get_physics, start_arena, max_speed, max_speed_change, max_angle_change, too_close, image_size










### The arena environment
    
import gym # This is not deeply used yet. 
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from copy import deepcopy
from itertools import product

# How are agents rewarded/punished each step? 
dist_d      = .1     # Based on distance
closer_d    = 10     # Based on distance closer
speed_d       = 0     # Based on speed
step_d      = 0     # Based on how many steps have passed
col_d       = .1     # Based on collision with walls
win_lose    = 0     # Based on predator victory

def get_reward(agent, dist, closer, speed, step, collision, done, printing = False):
    try: speed = speed.item() 
    except: pass
    r_dist = (1/dist - .7) * dist_d if agent == "pred" else (-1/dist + .7) * dist_d
    r_closer = closer * closer_d if agent == "pred" else -closer * closer_d
    r_speed    = speed * speed_d
    r_step   = -step * step_d if agent == "pred" else step * step_d
    r_col    = -col_d if collision else 0
    r = r_dist + r_closer + r_speed + r_step + r_col
    if(done):
        if(dist <= too_close): r += win_lose if agent == "pred" else 0# -win_lose
        else:                  r += win_lose if agent == "prey" else 0# -win_lose 
    if(printing):
        print("\n{} reward {}:\n\t{} from dist,\n\t{} from dist closer,\n\t{} from speed,\n\t{} from step,\n\t{} from collision.".format(
            agent, round(r,3), round(r_dist,3), round(r_closer,3), round(r_speed,3), round(r_step, 3), round(r_col, 3)))
    return(r)

def add_discount(rewards, last, GAMMA = .9):
    discounts = [last * (GAMMA**i) for i in range(len(rewards))]
    discounts.reverse()
    #for r, d in zip(rewards, discounts):
    #    print("{} + {} = {}".format(r, d, r+d))
    return([r + d for r, d in zip(rewards, discounts)])

def include_steps(rewards, loss_per_step = .003, add = False):
    steps = len(rewards)
    if(add):
        return([r + steps*loss_per_step for r in rewards])
    return([r - steps*loss_per_step for r in rewards])

# Made an environment! 
indexes = [i for i in range(10000,0,-1)]
class PredPreyEnv():   
    def __init__(
            self, 
            test = False,               # Is this a test for either agent? "pred" or "prey"
            GUI = False,                # Is it rendered on-screen?
            pred_condition = False,     # [None, "still" or "random"]
            prey_condition = False,     # [None, "still" or "random"]
            arena_name = "arena.png"):
        self.arena_name = arena_name
        self.already_constructed = False
        self.index = indexes.pop()
        self.test = test
        self.GUI = GUI
        if(self.GUI):   self.physicsClient = get_physics(self.GUI, self.arena_name)
        else:           self.physicsClient = get_physics()            
        self.observation_space = gym.spaces.box.Box(
            low=-1.0, high=1.0, shape=(2, image_size, image_size, 4), dtype=np.float64)
        self.action_space = gym.spaces.box.Box(
            low=-max_angle_change, high=max_angle_change, shape=(2,), dtype=np.float64)
        self.np_random, _ = gym.utils.seeding.np_random()
        self.pred, self.prey = None, None
        self.steps = 0
        self.resets = 0
        self.pred_condition = pred_condition
        self.prey_condition = prey_condition
        self.normalize = False
        
    def reset(self):
        self.close()
        self.resets += 1
        self.pred, self.prey, self.pos_pred, self.pos_prey, self.yaw_pred, self.yaw_prey, self.speed_pred, self.speed_prey, wall_ids = \
            start_arena(self.test, self.physicsClient, self.arena_name, self.already_constructed)
        if(self.already_constructed == False): self.wall_ids = wall_ids
        self.already_constructed = True
        self.steps = 0
        self.pred_energy, self.prey_energy = 3000, 3000
        return(self.get_obs())

    def agent_dist(self):
        x = self.pos_pred[0] - self.pos_prey[0]
        y = self.pos_pred[1] - self.pos_prey[1]
        return((x**2 + y**2)**.5)
    
    def collisions(self, agent = "both"):
        if(agent == "both"): return(self.collisions(self.pred), self.collisions(self.prey))
        col = False
        for wall in self.wall_ids:
            if len(p.getContactPoints(agent, wall, physicsClientId = self.physicsClient)) > 0:
                col = True
        return(col)
        
    def change_angle_speed(self, agent, yaw, speed):
        if(agent == self.pred): old_yaw, old_speed, pos = self.yaw_pred, self.speed_pred, self.pos_pred
        if(agent == self.prey): old_yaw, old_speed, pos = self.yaw_prey, self.speed_prey, self.pos_prey
        
        new_yaw = old_yaw + yaw
        new_yaw %= 2*pi
        orn = p.getQuaternionFromEuler([0,0,new_yaw])
        p.resetBasePositionAndOrientation(agent,(pos[0], pos[1], .5), orn, physicsClientId = self.physicsClient)
        #print("\n", agent)
        #print("Change angle:", round(degrees(yaw)))
        #print("Old:", round(degrees(old_yaw)))
        #print("New:", round(degrees(new_yaw)))
        
        new_speed = old_speed + speed
        new_speed = sorted((0, max_speed, new_speed))[1]
        x = cos(new_yaw)*new_speed
        y = sin(new_yaw)*new_speed
        p.resetBaseVelocity(agent, (x,y,0), (0,0,0), physicsClientId = self.physicsClient)
        #print("Change speed:", round(speed))
        #print("Old:", round(old_speed))
        #print("New:", round(new_speed))
        
        if(agent == self.pred): self.yaw_pred = new_yaw; self.speed_pred = new_speed
        if(agent == self.prey): self.yaw_prey = new_yaw; self.speed_prey = new_speed
        
    def get_speed(self, agent):
        (x, y, _), _ = p.getBaseVelocity(agent, self.physicsClient)
        return(sqrt(x**2 + y**2))
        
        
    def get_obs(self, agent = "both"):
        if(agent == "both"): return(self.get_obs(self.pred), self.get_obs(self.prey))
        if(agent == self.pred): yaw, pos = self.yaw_pred, self.pos_pred
        if(agent == self.prey): yaw, pos = self.yaw_prey, self.pos_prey
    
        x, y = cos(yaw), sin(yaw)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [pos[0], pos[1], .5], 
            cameraTargetPosition = [pos[0] + x, pos[1] + y, .5], 
            cameraUpVector = [0, 0, 1], physicsClientId = self.physicsClient
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov = 90, 
            aspect = 1, 
            nearVal = 0.2, 
            farVal = 10, physicsClientId = self.physicsClient
        )
        w, h, rgba, depth, mask = p.getCameraImage(
            width=image_size,
            height=image_size,
            projectionMatrix=proj_matrix,
            viewMatrix=view_matrix, physicsClientId = self.physicsClient
        )
        rgb = rgba[:,:,0:-1]
        rgb = np.divide(rgb,255)
        rgb = rgb * 2 - 1
        d = np.expand_dims(depth, axis=-1)
        rgbd = np.concatenate([rgb, d], axis = -1)
        rgbd = torch.from_numpy(rgbd).to(device).float()
        return(rgbd)
    
    def step(self, ang_speed_1 = None, ang_speed_2 = None):
        # If an agent is moving randomly, pick random movement.
        if(self.pred_condition == "random"): 
            angle_1 = random.uniform(-max_angle_change,max_angle_change)
            speed_1 = random.uniform(-max_speed_change,max_speed_change)
        # Otherwise, make sure the given movement is within allowance. 
        else:
            if(self.normalize):
                angle_1 = -max_angle_change + (ang_speed_1[0] + 1.0) * 0.5 * (max_angle_change - -max_angle_change)
                speed_1   = -max_speed_change + (ang_speed_1[1] + 1.0) * 0.5 * (max_speed_change - -max_speed_change)
            else:   
                angle_1 = ang_speed_1[0] 
                speed_1 = ang_speed_1[1] 
            angle_1 = sorted((-max_angle_change, max_angle_change, angle_1))[1]
            speed_1 = sorted((-max_speed_change, max_speed_change, speed_1))[1]
        if(self.prey_condition == "random"): 
            angle_2 = random.uniform(-max_angle_change,max_angle_change)
            speed_2 = random.uniform(-max_speed_change,max_speed_change)
        else:
            if(self.normalize):
                angle_2 = -max_angle_change + (ang_speed_2[0] + 1.0) * 0.5 * (max_angle_change - -max_angle_change)
                speed_2 = -max_speed_change + (ang_speed_2[1] + 1.0) * 0.5 * (max_speed_change - -max_speed_change)
            else:   
                angle_2 = ang_speed_2[0] 
                speed_2 = ang_speed_2[1] 
            angle_2 = sorted((-max_angle_change, max_angle_change, angle_2))[1]
            speed_2 = sorted((-max_speed_change, max_speed_change, speed_2))[1]
        self.steps += 1
        dist_before = self.agent_dist()
        self.change_angle_speed(self.pred, angle_1, speed_1)
        self.change_angle_speed(self.prey, angle_2, speed_2)
        if(self.pred_condition == "still" or self.pred_energy <= 0): p.resetBaseVelocity(self.pred, (0,0,0), (0,0,0), physicsClientId = self.physicsClient); self.speed_pred = 0
        if(self.prey_condition == "still" or self.prey_energy <= 0): p.resetBaseVelocity(self.prey, (0,0,0), (0,0,0), physicsClientId = self.physicsClient); self.speed_prey = 0
        p.stepSimulation(physicsClientId = self.physicsClient)
        self.pos_pred, _ = p.getBasePositionAndOrientation(self.pred, physicsClientId = self.physicsClient)
        self.pos_prey, _ = p.getBasePositionAndOrientation(self.prey, physicsClientId = self.physicsClient)
        self.speed_pred = self.get_speed(self.pred)
        self.speed_prey = self.get_speed(self.prey)
        dist_after = self.agent_dist()
        dist_closer = dist_before - dist_after
        pred_collision, prey_collision = self.collisions()
        self.pred_energy -= self.speed_pred+5
        self.prey_energy -= self.speed_prey+5
        done = True if dist_after <= too_close or self.pred_energy <= 0 else False
        reward = (
            get_reward("pred", dist_after, dist_closer, self.speed_pred, self.steps, pred_collision, done), 
            get_reward("prey", dist_after, dist_closer, self.speed_prey, self.steps, prey_collision, done))
        #if(done): print("\n\n\tEnded!\n\n")
        observations = self.get_obs(agent = "both")
        return(observations, reward, done, dist_after)
    
    def render(self, agent = "both", name = "image", save_folder = None):
        if(agent == "both"): return(
                self.render(self.pred, name + "_predator"), 
                self.render(self.prey, name + "_prey"),
                self.render("above", name + "_above"))
        if(agent != "above"):
            rgbd = self.get_obs(agent)
            rgb = rgbd[:,:,0:3]
            rgb = (rgb + 1)/2
            rgb = rgb.cpu()
            plt.figure(figsize = (5,5))
            plt.imshow(rgb)
            plt.axis('off')
            os.chdir(file_1)
            #plt.savefig('images/{}.png'.format(name), bbox_inches='tight', pad_inches = 0)
            os.chdir(file_2)
            plt.show()
            plt.close()
            plt.ioff()
            return()
        
        x_1, y_1 = self.pos_pred[0], self.pos_pred[1]
        x_2, y_2 = self.pos_prey[0], self.pos_prey[1]
        x = (x_1 + x_2)/2
        y = (y_1 + y_2)/2
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [x, y, self.agent_dist() + 1], 
            cameraTargetPosition = [x, y, 0], 
            cameraUpVector = [1, 0, 0], physicsClientId = self.physicsClient
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov = 90, 
            aspect = 1, 
            nearVal = 0.001, 
            farVal = self.agent_dist() + 2, physicsClientId = self.physicsClient
        )
        w, h, rgba, depth, mask = p.getCameraImage(
            width=128,
            height=128,
            projectionMatrix=proj_matrix,
            viewMatrix=view_matrix, physicsClientId = self.physicsClient
        )
        rgb = rgba[:,:,0:-1]
        rgb = np.divide(rgb,255)
        plt.figure(figsize = (10,10))
        plt.imshow(rgb)
        plt.axis('off')
        if(save_folder != None):
            plt.savefig('{}/{}.png'.format(save_folder, name), bbox_inches='tight', pad_inches = 0)
        plt.show()
        plt.close()
        plt.ioff()
        print("Dist:", self.agent_dist())

    def close(self, forever = False):
        if(self.pred != None):
            p.removeBody(self.pred, physicsClientId = self.physicsClient)
            p.removeBody(self.prey, physicsClientId = self.physicsClient)
        if(self.resets == 100 and self.GUI):
            p.disconnect(self.physicsClient)
            self.physicsClient = get_physics(True, name = self.arena_name) 
        if(forever):
            p.disconnect(self.physicsClient)
            indexes.append(self.index)
    

    


from torch.distributions import Normal
from tqdm import tqdm



def run_with_GUI(test = False, pred = None, prey = None, GUI = True, episodes = 100, pred_condition = False, prey_condition = False, arena_name = "arena.png", render = False):
    env = PredPreyEnv(test = test, GUI = GUI, pred_condition = pred_condition, prey_condition = prey_condition, arena_name = arena_name)
    win_list = []
    if(pred != None): pred.eval()
    if(prey != None): prey.eval()
    for i in tqdm(range(episodes), desc = "Testing", position=0, leave=True):
        done = False
        pred_actor_hc, prey_actor_hc = None, None
        ang_speed_1, ang_speed_2 = None, None
        steps = 0
        obs = env.reset()  
        while(done == False):
            steps += 1
            if(pred != None):
                ang_speed_1, pred_actor_hc = pred.get_action(obs[0], env.speed_pred, env.pred_energy, ang_speed_1, pred_actor_hc)
            else:
                ang_speed_1 = (None, None)
            if(prey != None):
                ang_speed_2, prey_actor_hc = prey.get_action(obs[1], env.speed_prey, env.prey_energy, ang_speed_2, prey_actor_hc)
            else:
                ang_speed_2 = (None, None)
            obs, _, done, dist_after = env.step(ang_speed_1, ang_speed_2)
        win = dist_after < too_close
        win_list.append(win)
    env.close(forever = True)
    win_percent = round(100*sum(win_list)/episodes)
    print("\nWinning: {}%. Duration: {}.".format(win_percent, duration()))
    return(win_percent)


    
    
    

if __name__ == "__main__":
    env = PredPreyEnv(arena_name = "big_arena.png")
    env.reset()   
    env.render() 
    print("\nObservation space:")
    print(env.observation_space.shape)
    print("\nAction space:")
    print(env.action_space.shape)
    print()
    env.close(forever = True)
    run_with_GUI(episodes = 100, arena_name = "empty_arena.png", 
                 pred_condition = "random", prey_condition = "random",
                 render = False)
