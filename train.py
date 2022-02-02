import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
from tqdm import tqdm
import keyboard
from math import pi

from utils import parameters as para
from utils import parameters, delete_these, get_rolling_average, \
    duration, reset_start_time, remove_folder, make_folder, plot_wins, plot_losses, plot_rewards, \
    save_pred_prey, load_pred_prey
from pred_prey_env import PredPreyEnv
from how_to_play import episode, hand_episodes
from rtd3 import RecurrentTD3




  


class Trainer():
    def __init__(
            self, para = para, training_agent = "both", play_by_hand = False,
            save_folder = "default", load_folder = None, load_name = "last",
            restart_if = {"pred" : {500 : {"hard" : .1}}},
            done_if =    {"pred" : {250 : {"easy" : .99, "med" : .99, "hard" : .95}}},
            difficulty_dic = {"easy" : (0,  0), "med"  : (0,  100), "hard" : (100,100)}):
        
        self.para = para
        self.training_agent = training_agent
        self.attempts = 0
        self.start_pred_condition = self.para.pred_condition
        self.start_prey_condition = self.para.prey_condition
        self.save_folder = save_folder
        self.load_folder = load_folder; self.load_name = load_name
        
        self.env = PredPreyEnv(para, GUI = False)
        self.env_gui = PredPreyEnv(para, GUI = True)
        self.restart_if = restart_if; self.done_if = done_if

        self.pred_episodes, self.prey_episodes = None, None
        self.restart()
        if(play_by_hand):
            self.pred_episodes, self.prey_episodes = hand_episodes(
                self.env_gui, self.pred, self.prey,
                "by_hand", self.prey_condition)
        self.difficulty_dic = difficulty_dic
    
    def restart(self):
        reset_start_time()
        remove_folder(self.save_folder)
        make_folder(self.save_folder)
        self.attempts += 1
        self.e = 0
        self.pred = RecurrentTD3(); self.prey = RecurrentTD3()
        if(self.load_folder != None):
            self.pred, self.prey = load_pred_prey(
                self.pred, self.prey, post = self.load_name, folder = self.load_folder)
        if(self.pred_episodes != None and self.prey_episodes != None):
            self.pred.episodes = self.pred_episodes
            self.prey.episodes = self.prey_episodes
        save_pred_prey(self.pred, self.prey, post = "0", folder = self.save_folder)
        self.para.pred_condition = self.start_pred_condition
        self.para.prey_condition = self.start_prey_condition
        self.easy_wins = []; self.med_wins = []; self.hard_wins = []
        self.easy_wins_rolled = []; self.med_wins_rolled = []; self.hard_wins_rolled = []
        self.pred_losses = np.array([[None]*3])
        self.prey_losses = np.array([[None]*3])
      
    def close(self):
        self.env.close(forever = True)
        self.env_gui.close(forever = True)
        

      
    def one_episode(self, difficulty = "med", push = True, GUI = False):
        min_dif, max_dif = self.difficulty_dic[difficulty]
        
        if(GUI == False): GUI = keyboard.is_pressed('q') 
        if(GUI): env = self.env_gui
        else:    env = self.env
        
        to_push_pred, to_push_prey, pred_win, rewards = episode(
            env, self.pred, self.prey, min_dif = min_dif, max_dif = max_dif, GUI = GUI)
        if(keyboard.is_pressed('q') ): plot_rewards(rewards)
        
        if(push):
            for i in range(len(to_push_pred)):
                self.pred.episodes.push(to_push_pred[i])
                self.prey.episodes.push(to_push_prey[i])
            
        return(int(pred_win), rewards)


    def epoch(self):
        win, rewards = self.one_episode("easy")
        self.easy_wins.append(win)
        self.easy_wins_rolled.append(get_rolling_average(self.easy_wins))
      
        win, rewards = self.one_episode("med")
        self.med_wins.append(win)
        self.med_wins_rolled.append(get_rolling_average(self.med_wins))
      
        win, rewards = self.one_episode("hard")
        self.hard_wins.append(win)
        self.hard_wins_rolled.append(get_rolling_average(self.hard_wins))
      
        if(type(self.para.pred_condition) in [int, float]):
            self.para.pred_condition *= .99
        if(type(self.para.prey_condition) in [int, float]):
            self.para.prey_condition *= .99
        
        iterations = 4
        if(self.training_agent in ["pred", "both"]):
            pred_losses = self.pred.update_networks(batch_size = 32, iterations = iterations)
        else: pred_losses = np.array([[None]*3]*iterations)
        self.pred_losses = np.concatenate([self.pred_losses, pred_losses])
        
        if(self.training_agent in ["prey", "both"]):
            prey_losses = self.prey.update_networks(batch_size = 32, iterations = iterations)
        else: prey_losses = np.array([[None]*3]*iterations)
        if(iterations == 1):  pred_losses = np.expand_dims(pred_losses,0); prey_losses = np.expand_dims(prey_losses,0)
        self.pred_losses = np.concatenate([self.pred_losses, pred_losses])
        self.prey_losses = np.concatenate([self.prey_losses, prey_losses])
    
        if(keyboard.is_pressed('q') ): plot_losses(self.pred_losses, self.prey_losses, too_long = 300)


    def restart_or_done(self):
        
        restart = False
        for agent in self.restart_if.keys():
            for epochs in self.restart_if[agent].keys():
                for difficulty in self.restart_if[agent][epochs].keys():
                    if self.e > epochs:
                        if(difficulty == "easy"):   pred_wins = self.easy_wins_rolled[-1]
                        elif(difficulty == "med"):  pred_wins = self.med_wins_rolled[-1]
                        elif(difficulty == "hard"): pred_wins = self.hard_wins_rolled[-1]
                        if((agent == "pred" and pred_wins < self.restart_if[agent][epochs][difficulty]) or
                           (agent == "prey" and pred_wins > self.restart_if[agent][epochs][difficulty])):
                            restart = True
                            
        done = False
        for agent in self.done_if.keys():
            for epochs in self.done_if[agent].keys():
                difficulties_done = [False] * len(self.done_if[agent][epochs].keys())
                for i, difficulty in enumerate(self.done_if[agent][epochs].keys()):
                    if self.e > epochs:
                        if(difficulty == "easy"):   pred_wins = self.easy_wins_rolled[-1]
                        elif(difficulty == "med"):  pred_wins = self.med_wins_rolled[-1]
                        elif(difficulty == "hard"): pred_wins = self.hard_wins_rolled[-1]
                        if((agent == "pred" and pred_wins >= self.done_if[agent][epochs][difficulty]) or
                           (agent == "prey" and pred_wins <= self.done_if[agent][epochs][difficulty])):
                            difficulties_done[i] = True
                if(sum(difficulties_done) == len(difficulties_done)): done = True
        return(restart, done)
                        
        

    def train(
            self, 
            max_epochs = 1000, how_often_to_show_and_save = 25):
        
        self.pred.train(); self.prey.train()
        while(self.e < max_epochs):
            self.e += 1
            if(self.e % 5 == 0):  
                print("\nEpoch {}, {} attempt(s). {}.".format(self.e, self.attempts, duration()))
                print("Predator condition: {}. Prey condition: {}.".format(
                    self.para.pred_condition, self.para.prey_condition))
            self.epoch()
            if(self.e % how_often_to_show_and_save == 0): 
                plot_wins(self.easy_wins_rolled, self.med_wins_rolled, self.hard_wins_rolled, name = "wins_{}".format(self.e), folder = self.save_folder)
                plot_losses(self.pred_losses, self.prey_losses, too_long = 300)
                save_pred_prey(self.pred, self.prey, post = "{}".format(self.e), folder = self.save_folder)
            
            restart, done = self.restart_or_done()
            if(restart):
                print("This isn't working. Starting again!")
                delete_these(True, self.pred, self.prey, self.easy_wins, self.med_wins, self.hard_wins,
                   self.easy_wins_rolled, self.med_wins_rolled, self.hard_wins_rolled, self.pred_losses, self.prey_losses)
                self.restart()
            if(done or self.e >= max_epochs):
                print("\n\nFinished!\n\n")
                print("\n\nPredator condition: {}. Prey condition: {}.\n".format(
                    self.para.pred_condition, self.para.prey_condition))
                save_pred_prey(self.pred, self.prey, post = "last", folder = self.save_folder)
                plot_wins(self.easy_wins_rolled, self.med_wins_rolled, self.hard_wins_rolled, name = "wins_last".format(self.e), folder = self.save_folder)
                plot_losses(self.pred_losses, self.prey_losses, too_long = None, name = "losses".format(self.e), folder = self.save_folder)
                break
    
    def test(self, size = 100):
        self.pred.eval(); self.prey.eval()
        pred_wins = 0
        for i in range(size):
            w, _ = self.one_episode(difficulty = "hard", push = False, GUI = True)
            pred_wins += w
        print("Predator wins {} out of {} games ({}%).".format(pred_wins, size, round(100*(pred_wins/size))))
