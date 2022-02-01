import torch
import numpy as np
import pybullet as p
from math import degrees, pi, cos, sin

from utils import add_discount, parameters as para
from arena import get_physics, Arena

    
from matplotlib import pyplot as plt
from torchvision.transforms.functional import resize





# Made an environment! 
class PredPreyEnv():   
    def __init__(self, para = para, GUI = False):
        self.para = para
        self.GUI = GUI
        self.arena = Arena(para, self.GUI)
        self.already_constructed = False
        self.pred, self.prey = None, None
        self.steps, self.resets = 0, 0

    def close(self, forever = False):
        if(self.pred != None):
            p.removeBody(self.pred.p_num, physicsClientId = self.arena.physicsClient)
            p.removeBody(self.prey.p_num, physicsClientId = self.arena.physicsClient)
        if(self.resets % 100 == 99 and self.GUI and not forever):
            p.disconnect(self.arena.physicsClient)
            self.already_constructed = False
            self.arena.physicsClient = get_physics(self.GUI, self.arena.w, self.arena.h)
        if(forever):
            p.disconnect(self.arena.physicsClient)  

    def reset(self, min_dif = 0, max_dif = 100):
        self.close()
        self.resets += 1; self.steps = 0
        self.pred, self.prey, wall_ids = \
            self.arena.start_arena(self.already_constructed, min_dif, max_dif)
        if(self.already_constructed == False): 
            self.wall_ids = wall_ids; self.already_constructed = True
        return(self.get_obs())

    def get_obs(self, agent_name = "both"):
        if(agent_name == "both"): return(self.get_obs("pred"), self.get_obs("prey"))
        elif(agent_name == "pred"): 
            agent = self.pred
            image_size = self.para.pred_image_size
        elif(agent_name == "prey"): 
            agent = self.prey
            image_size = self.para.prey_image_size
        else: print("Not a good agent."); return
      
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
    
        x_1, y_1 = self.pred.pos[0], self.pred.pos[1]
        x_2, y_2 = self.prey.pos[0], self.prey.pos[1]
        x = (x_1 + x_2)/2
        y = (y_1 + y_2)/2
        
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

    def agent_dist(self):
        x = self.pred.pos[0] - self.prey.pos[0]
        y = self.pred.pos[1] - self.prey.pos[1]
        return((x**2 + y**2)**.5)

    def get_position_and_speed(self, agent):
        pos, _ = p.getBasePositionAndOrientation(agent.p_num, physicsClientId = self.arena.physicsClient)
        (x, y, _), _ = p.getBaseVelocity(agent.p_num, physicsClientId = self.arena.physicsClient)
        speed = (x**2 + y**2)**.5
        return(pos, speed)

    def collisions(self, agent = "both"):
        if(agent == "both"): return(self.collisions(self.pred.p_num), self.collisions(self.prey.p_num))
        col = False
        for wall in self.wall_ids:
            if 0 < len(p.getContactPoints(agent.p_num, wall, physicsClientId = self.arena.physicsClient)):
                col = True
        return(col)

    def pred_hits_prey(self):
        return(0 < len(p.getContactPoints(
            self.pred.p_num, self.prey.p_num, physicsClientId = self.arena.physicsClient)))

    def change_velocity(self, agent_name, yaw_change, speed, verbose = False):
        if(agent_name == "pred"): agent = self.pred
        if(agent_name == "prey"): agent = self.prey
        
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
            print("\n{}:\nOld yaw:\t{}\nChange:\t\t{}\nNew yaw:\t{}".format(
                agent_name, round(degrees(old_yaw)), round(degrees(yaw_change)), round(degrees(new_yaw))))
            print("Old speed:\t{}\nNew speed:\t{}".format(old_speed, speed))

    def unnormalize(self, action, agent_name): # from (-1, 1) to (min, max)
        if(agent_name == "pred"): 
            max_angle_change = self.para.pred_max_yaw_change
            min_speed = self.para.pred_min_speed
            max_speed = self.para.pred_max_speed
        else:                     
            max_angle_change = self.para.prey_max_yaw_change
            min_speed = self.para.prey_min_speed
            max_speed = self.para.prey_max_speed
        yaw = action[0].clip(-1,1).item() * max_angle_change
        spe = min_speed + ((action[1].clip(-1,1).item() + 1)/2) * (max_speed - min_speed)
        return(yaw, spe)

    def get_reward(self, agent_name, dist, closer, collision, pred_hits_prey, verbose = False):
        dist_d = self.para.pred_reward_dist if agent_name == "pred" else self.para.prey_reward_dist
        closer_d = self.para.pred_reward_dist_closer if agent_name == "pred" else self.para.prey_reward_dist_closer
        col_d = self.para.pred_reward_collision if agent_name == "pred" else self.para.prey_reward_collision
        
        r_dist = (1/dist - .7) * dist_d
        if(pred_hits_prey): r_dist = 2.5 if agent_name == "pred" else -2.5
        r_closer = closer * closer_d
        r_col    = -col_d if collision else 0
        r = r_dist + r_closer + r_col
        if(verbose):
            print("\n{} reward {}:\n\t{} from dist,\n\t{} from dist closer,\n\t{} from collision.".format(
                agent_name, round(r,3), round(r_dist,3), round(r_closer,3), round(r_col, 3)))
        return(r)
  
    def step(self, pred_action, prey_action):
        self.pred.action, self.prey.action = pred_action, prey_action
        self.steps += 1
        yaw_1, spe_1 = self.unnormalize(self.pred.action, "pred")
        yaw_2, spe_2 = self.unnormalize(self.prey.action, "prey")
        if(self.pred.energy <= 0): spe_1 = 0
        if(self.prey.energy <= 0): spe_2 = 0
        self.change_velocity("pred", yaw_1, spe_1)
        self.change_velocity("prey", yaw_2, spe_2)
      
        dist_before = self.agent_dist()
        p.stepSimulation(physicsClientId = self.arena.physicsClient)
        self.pred.energy -= spe_1
        self.prey.energy -= spe_2
        self.pred.pos, self.pred.spe = self.get_position_and_speed(self.pred)
        self.prey.pos, self.prey.spe = self.get_position_and_speed(self.prey)
        dist_after = self.agent_dist()
        dist_closer = dist_before - dist_after
      
        pred_collision, prey_collision = self.collisions()
        pred_hits_prey = self.pred_hits_prey()
        rewards = (
            self.get_reward("pred", dist_after, dist_closer, pred_collision, pred_hits_prey), 
            self.get_reward("prey", dist_after, dist_closer, prey_collision, pred_hits_prey))
        observations = self.get_obs(agent_name = "both")
        done = True if pred_hits_prey or self.pred.energy <= 0 else False
        return(observations, rewards, done, pred_hits_prey)
      

    
    

if __name__ == "__main__":
    env = PredPreyEnv()
    env.reset()   
    env.render() 
    env.close(forever = True)
