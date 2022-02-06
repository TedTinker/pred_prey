import torch
import numpy as np
import pybullet as p
from math import degrees, pi, cos, sin
from itertools import product

from utils import get_arg, parameters as para, add_discount
from arena import get_physics, Arena

    
from matplotlib import pyplot as plt
from torchvision.transforms.functional import resize





# Made an environment! 
class PredPreyEnv():   
    def __init__(self, para = para, GUI = False):
        self.para = para
        self.GUI = GUI
        self.arena = Arena(para, self.GUI)
        self.flower_list = []
        self.agent_list = []
        self.dead_agents = []
        self.steps, self.resets = 0, 0

    def close(self, forever = False):
        self.arena.used_spots = []
        for agent in self.agent_list:
            p.removeBody(agent.p_num, physicsClientId = self.arena.physicsClient)
        self.agent_list = []
        self.dead_agents = []
        for flower in self.flower_list:
            p.removeBody(flower, physicsClientId = self.arena.physicsClient)
        self.flower_list = []
        if(self.resets % 100 == 99 and self.GUI and not forever):
            p.disconnect(self.arena.physicsClient)
            self.arena.already_constructed = False
            self.arena.physicsClient = get_physics(self.GUI, self.arena.w, self.arena.h)
        if(forever):
            p.disconnect(self.arena.physicsClient)  

    def reset(self):
        self.close()
        self.resets += 1; self.steps = 0
        self.arena.start_arena()
        for _ in range(self.para.pred_start):
            self.agent_list.append(self.arena.make_agent(True))
        for _ in range(self.para.prey_start):
            self.agent_list.append(self.arena.make_agent(False))
        for _ in range(self.para.flowers):
            self.flower_list.append(self.arena.make_flower())
        return([self.get_obs(agent) for agent in self.agent_list])

    def get_obs(self, agent):
        image_size = get_arg(self.para, agent.predator, "image_size")

        x, y = cos(agent.yaw), sin(agent.yaw)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [agent.pos[0], agent.pos[1], .5], 
            cameraTargetPosition = [agent.pos[0] + x, agent.pos[1] + y, .5], 
            cameraUpVector = [0, 0, 1], physicsClientId = self.arena.physicsClient)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov = 90, aspect = 1, nearVal = 0.2, 
            farVal = 10, physicsClientId = self.arena.physicsClient)
        _, _, rgba, depth, _ = p.getCameraImage(
            width=64, height=64,
            projectionMatrix=proj_matrix, viewMatrix=view_matrix, 
            physicsClientId = self.arena.physicsClient)
        
        rgb = np.divide(rgba[:,:,:-1], 255) * 2 - 1
        d = np.expand_dims(depth, axis=-1)
        rgbd = np.concatenate([rgb, d], axis = -1)
        rgbd = torch.from_numpy(rgbd).float()
        rgbd = resize(rgbd.permute(-1,0,1), (image_size, image_size)).permute(1,2,0)
        return(rgbd, agent.spe, agent.energy, agent.action)

    def render(self, agent = "all"):
        if(agent == "all"): 
            self.render("above")
            for agent in self.agent_list:
                self.render(agent)
            return()
      
        if(agent != "above"):
            rgbd, _, _, _ = self.get_obs(agent)
            rgb = (rgbd[:,:,0:3] + 1)/2
            plt.figure(figsize = (5,5))
            plt.imshow(rgb)
            plt.title("{} {}'s view".format("Predator" if agent.predator else "Prey", agent.p_num))
            plt.show()
            plt.close()
            plt.ioff()
            return()
    
        xs = [agent.pos[0] for agent in self.agent_list]
        ys = [agent.pos[1] for agent in self.agent_list]
        x = sum(xs)/len(xs)
        y = sum(ys)/len(ys)
        dist = max([self.arena.agent_dist(agent_1.p_num, agent_2.p_num) for agent_1, agent_2 in \
                product(self.agent_list, self.agent_list)])
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [x, y, dist + 1], 
            cameraTargetPosition = [x, y, 0], 
            cameraUpVector = [1, 0, 0], physicsClientId = self.arena.physicsClient)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov = 90, aspect = 1, nearVal = 0.001, 
            farVal = dist + 2, physicsClientId = self.arena.physicsClient)
        _, _, rgba, _, _ = p.getCameraImage(
            width=64, height=64,
            projectionMatrix=proj_matrix, viewMatrix=view_matrix, 
            physicsClientId = self.arena.physicsClient)
        
        rgb = rgba[:,:,:-1]
        rgb = np.divide(rgb,255)
        plt.figure(figsize = (10,10))
        plt.imshow(rgb)
        plt.title("View from above")
        plt.show()
        plt.close()
        plt.ioff()
        
    def maintain_flowers(self):
        for flower in self.flower_list:
            pos, orn = p.getBasePositionAndOrientation(flower, physicsClientId = self.arena.physicsClient)
            p.resetBasePositionAndOrientation(flower,(pos[0], pos[1], .5), orn, physicsClientId = self.arena.physicsClient)

        
    def update_pos_yaw_spe(self):
        for agent in self.agent_list:
            agent.pos, agent.yaw, agent.spe = self.arena.get_pos_yaw_spe(agent.p_num)

    def change_velocity(self, agent, yaw_change, speed, verbose = False):
        old_yaw = agent.yaw
        new_yaw = old_yaw + yaw_change
        new_yaw %= 2*pi
        orn = p.getQuaternionFromEuler([pi/2,0,new_yaw])
        p.resetBasePositionAndOrientation(agent.p_num,(agent.pos[0], agent.pos[1], .5), orn, physicsClientId = self.arena.physicsClient)
        agent.yaw = new_yaw
        
        old_speed = agent.spe
        x = cos(new_yaw)*speed
        y = sin(new_yaw)*speed
        p.resetBaseVelocity(agent.p_num, (x,y,0), (0,0,0), physicsClientId = self.arena.physicsClient)
        agent.spe = speed
                
        if(verbose):
            print("\n{} {}:\nOld yaw:\t{}\nChange:\t\t{}\nNew yaw:\t{}".format(
                "Predator" if agent.predator else "Prey", agent.p_num, round(degrees(old_yaw)), round(degrees(yaw_change)), round(degrees(new_yaw))))
            print("Old speed:\t{}\nNew speed:\t{}".format(old_speed, speed))

    def unnormalize(self, action, predator): # from (-1, 1) to (min, max)
        max_angle_change = get_arg(self.para, predator, "max_yaw_change")
        min_speed = get_arg(self.para, predator, "min_speed")
        max_speed = get_arg(self.para, predator, "max_speed")
        yaw = action[0].clip(-1,1).item() * max_angle_change
        spe = min_speed + ((action[1].clip(-1,1).item() + 1)/2) * (max_speed - min_speed)
        return(yaw, spe)
    
    def get_action(self, agent, brain, obs = None):
        if(obs == None): obs = self.get_obs(agent)
        agent.action, agent.hidden = brain.act(
            obs[0], obs[1], obs[2], obs[3], agent.hidden, 
            get_arg(self.para, agent.predator, "condition"))

    def get_reward(self, agent, dist, closer, flower_dist, flower_closer, collision, pred_hits_prey, prey_hits_flower):
        dist_d = get_arg(self.para, agent.predator, "reward_dist")
        closer_d = get_arg(self.para, agent.predator, "reward_dist_closer")
        f_dist_d = get_arg(self.para, agent.predator, "reward_flower_dist")
        f_closer_d = get_arg(self.para, agent.predator, "reward_flower_dist_closer")
        col_d = get_arg(self.para, agent.predator, "reward_collision")
        
        r_dist = (1/dist - .7) * dist_d
        if(pred_hits_prey): 
            r_dist = 2.5 if agent.predator else -2.5
        r_closer = closer * closer_d
        r_f_dist = (1/flower_dist - .7) * f_dist_d
        if(prey_hits_flower): 
            r_f_dist = 2.5 if agent.predator else -2.5
        r_f_closer = flower_closer * f_closer_d
        r_col    = -col_d if collision else 0
        r = r_dist + r_closer + r_f_dist + r_f_closer + r_col
        return(r)
    
    def update_rewards(self, agent, r, pred_win):
        r = r if (pred_win and agent.predator) or (not pred_win and not agent.predator) else -r
        reward_list = add_discount([p[4] for p in agent.to_push], r)
        agent.to_push = [(p[0], p[1], p[2], p[3], r, p[5], p[6], p[7], p[8], p[9]) for p, r in zip(agent.to_push, reward_list)]
  
    def step(self, obs_list, pred_brain, prey_brain, push):
        self.steps += 1
        
        for i, agent in enumerate(self.agent_list):
            agent.age += 1
            brain = pred_brain if agent.predator else prey_brain
            self.get_action(agent, brain, obs_list[i])
            yaw, spe = self.unnormalize(agent.action, agent.predator)
            agent.energy -= spe * get_arg(self.para, agent.predator, "energy_per_speed")
            self.change_velocity(agent, yaw, spe)
      
        agent_dists_before = self.arena.all_agent_dists(self.agent_list)
        flower_dists_before = self.arena.all_flower_dists(self.agent_list, self.flower_list)
        self.maintain_flowers()
        p.stepSimulation(physicsClientId = self.arena.physicsClient)
        self.update_pos_yaw_spe()
        agent_dists_after = self.arena.all_agent_dists(self.agent_list)
        agent_dists_closer = [before - after for before, after in zip(agent_dists_before, agent_dists_after)]
        flower_dists_after = self.arena.all_flower_dists(self.agent_list, self.flower_list)
        flower_dists_closer = [before - after for before, after in zip(flower_dists_before, flower_dists_after)]
        
        wall_collisions = self.arena.all_wall_collisions(self.agent_list)
        pred_prey_collisions = [False]*len(self.agent_list)
        prey_flower_collisions = [False]*len(self.agent_list)
        dead_flowers = []
        for i, agent in enumerate(self.agent_list):
            if(agent.predator):
                for j, agent_2 in enumerate(self.agent_list):
                    if(self.arena.agent_collisions(agent.p_num, agent_2.p_num) and
                       not agent_2.predator):
                        agent.energy += self.para.pred_energy_from_prey
                        pred_prey_collisions[i] = True
                        agent_2.energy -= self.para.pred_energy_from_prey
                        pred_prey_collisions[j] = True
            if(not agent.predator):
                for j, flower in enumerate(self.flower_list):
                    if(self.arena.agent_collisions(agent.p_num, flower)):
                        agent.energy += self.para.prey_energy_from_flower
                        prey_flower_collisions[i] = True
                        dead_flowers.append(j)
        for f in dead_flowers:
            p.removeBody(self.flower_list[f], physicsClientId = self.arena.physicsClient)
        self.flower_list = [flower for i, flower in enumerate(self.flower_list) if i not in dead_flowers]
        while(len(self.flower_list) < self.para.flowers):
            self.arena.used_spots = []
            self.flower_list.append(self.arena.make_flower())
            self.arena.used_spots = []
        
        rewards = [self.get_reward(self.agent_list[i], 
                                   agent_dists_after[i], agent_dists_closer[i], 
                                   flower_dists_after[i], flower_dists_closer[i],
                                   wall_collisions[i], pred_prey_collisions[i], prey_flower_collisions[i])
                   for i in range(len(self.agent_list))]
        new_obs_list = [self.get_obs(agent) for agent in self.agent_list]
        
        dones = [True if agent.energy <= 0 or agent.age >= get_arg(self.para, agent.predator, "max_age") else False for agent in self.agent_list]
                
        for i, agent in enumerate(self.agent_list):
            agent.to_push.append(
                (obs_list[i][0], obs_list[i][1], obs_list[i][2], new_obs_list[i][3], rewards[i], 
                new_obs_list[i][0], new_obs_list[i][1], new_obs_list[i][2], torch.tensor(dones[i]), torch.tensor(dones[i])))
            if(dones[i]):
                p.removeBody(agent.p_num, physicsClientId = self.arena.physicsClient)
                self.dead_agents.append(agent)
                if(push):
                    r=1
                    self.update_rewards(agent, r, pred_prey_collisions[i])
                    brain = pred_brain if agent.predator else prey_brain
                    for i in range(len(agent.to_push)):
                        brain.episodes.push(agent.to_push[i])
        
        self.agent_list = [agent for i, agent in enumerate(self.agent_list) if not dones[i]]
        done = True if (self.para.pred_start > 0 and 0 == len([agent for agent in self.agent_list if agent.predator])) or \
                       (self.para.prey_start > 0 and 0 == len([agent for agent in self.agent_list if not agent.predator])) or \
                       0 == len(self.agent_list) else False
        pred_win = False
        if(done):
            pred_win = True if len([agent for agent in self.agent_list if agent.predator]) > \
                               len([agent for agent in self.agent_list if not agent.predator]) else False
            for i, agent in enumerate(self.agent_list):
                self.dead_agents.append(agent)
                if(push):
                    r=1
                    self.update_rewards(agent, r, pred_win)
                    agent.to_push[-1] = (agent.to_push[-1][0], agent.to_push[-1][1], agent.to_push[-1][2], agent.to_push[-1][3], agent.to_push[-1][4], 
                                         agent.to_push[-1][5], agent.to_push[-1][6], agent.to_push[-1][7], torch.tensor(done), torch.tensor(done))
                    brain = pred_brain if agent.predator else prey_brain
                    for j in range(len(agent.to_push)):
                        brain.episodes.push(agent.to_push[j])
        return(new_obs_list, rewards, done, pred_win)
      

    
    

if __name__ == "__main__":
    env = PredPreyEnv()
    env.reset()   
    env.render() 
    env.close(forever = True)
