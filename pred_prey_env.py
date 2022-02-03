import torch
import numpy as np
import pybullet as p
from math import degrees, pi, cos, sin

from utils import get_arg, parameters as para
from arena import get_physics, Arena

    
from matplotlib import pyplot as plt
from torchvision.transforms.functional import resize





# Made an environment! 
class PredPreyEnv():   
    def __init__(self, para = para, GUI = False):
        self.para = para
        self.GUI = GUI
        self.arena = Arena(para, self.GUI)
        self.agent_list = []
        self.steps, self.resets = 0, 0

    def close(self, forever = False):
        self.arena.used_spots = []
        for agent in self.agent_list:
            p.removeBody(agent.p_num, physicsClientId = self.arena.physicsClient)
        self.agent_list = []
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
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [x, y, self.agent_dist() + 1], 
            cameraTargetPosition = [x, y, 0], 
            cameraUpVector = [1, 0, 0], physicsClientId = self.arena.physicsClient)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov = 90, aspect = 1, nearVal = 0.001, 
            farVal = self.agent_dist() + 2, physicsClientId = self.arena.physicsClient)
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

    def change_velocity(self, agent, yaw_change, speed, verbose = False):
        old_yaw = agent.yaw
        new_yaw = old_yaw + yaw_change
        new_yaw %= 2*pi
        orn = p.getQuaternionFromEuler([0,0,new_yaw])
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

    def get_reward(self, agent, dist, closer, collision, pred_hits_prey, verbose = False):
        dist_d = get_arg(self.para, agent.predator, "reward_dist")
        closer_d = get_arg(self.para, agent.predator, "reward_dist_closer")
        col_d = get_arg(self.para, agent.predator, "reward_collision")
        
        r_dist = (1/dist - .7) * dist_d
        if(pred_hits_prey): r_dist = 2.5 if agent.predator else -2.5
        r_closer = closer * closer_d
        r_col    = -col_d if collision else 0
        r = r_dist + r_closer + r_col
        if(verbose):
            print("\n{} {} reward {}:\n\t{} from dist,\n\t{} from dist closer,\n\t{} from collision.".format(
                "Predator" if agent.predator else "Prey", agent.p_num, round(r,3), round(r_dist,3), round(r_closer,3), round(r_col, 3)))
        return(r)
    
    def get_action(self, agent, brain, obs = None):
        if(obs == None): obs = self.get_obs(agent)
        agent.action, agent.hidden = brain.act(
            obs[0], obs[1], obs[2], obs[3], agent.hidden, 
            get_arg(self.para, agent.predator, "condition"))
  
    def step(self, obs_list, pred_brain, prey_brain):
        self.steps += 1
        
        for i, agent in enumerate(self.agent_list):
            brain = pred_brain if agent.predator else prey_brain
            self.get_action(agent, brain, obs_list[i])
            yaw, spe = self.unnormalize(agent.action, agent.predator)
            if(agent.energy <= 0): spe = 0
            agent.energy -= spe * get_arg(self.para, agent.predator, "energy_per_speed")
            self.change_velocity(agent, yaw, spe)
      
        dist_before = self.arena.agent_dist(self.agent_list[0].p_num, self.agent_list[1].p_num)
        p.stepSimulation(physicsClientId = self.arena.physicsClient)
        for agent in self.agent_list:
            agent.pos, agent.yaw, agent.spe = self.arena.get_pos_yaw_spe(agent.p_num)
        dist_after = self.arena.agent_dist(self.agent_list[0].p_num, self.agent_list[1].p_num)
        dist_closer = dist_before - dist_after
      
        pred_collision = self.arena.wall_collisions(self.agent_list[0].p_num)
        prey_collision = self.arena.wall_collisions(self.agent_list[1].p_num)
        pred_hits_prey = self.arena.agent_collisions(self.agent_list[0].p_num, self.agent_list[1].p_num)
        rewards = (
            self.get_reward(self.agent_list[0], dist_after, dist_closer, pred_collision, pred_hits_prey), 
            self.get_reward(self.agent_list[1], dist_after, dist_closer, prey_collision, pred_hits_prey))
        observations = [self.get_obs(agent) for agent in self.agent_list]
        done = True if pred_hits_prey or self.agent_list[0].energy <= 0 else False
        return(observations, rewards, done, pred_hits_prey)
      

    
    

if __name__ == "__main__":
    env = PredPreyEnv()
    env.reset()   
    env.render() 
    env.close(forever = True)
