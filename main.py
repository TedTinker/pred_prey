import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
from tqdm import tqdm
import keyboard

file = r"C:\Users\tedjt\Desktop\pred_prey"
os.chdir(file) 
from utils import delete_these, get_rolling_average, \
    duration, reset_start_time, empty_folder, make_folder, plot_wins, plot_losses, plot_rewards, \
    save_pred_prey, load_pred_prey
from pred_prey_env import PredPreyEnv
from how_to_play import episode, hand_episodes
from rtd3 import RecurrentTD3
#os.chdir(r"C:\Users\tedjt")




  


class Trainer():
    def __init__(
            self, arena_name, energy = 3000,
            pred_condition = 0, prey_condition = 0, folder = "default"):
        
        self.attempts = 0
        self.arena_name = arena_name
        self.env = PredPreyEnv(self.arena_name)
        self.env_gui = PredPreyEnv(self.arena_name, GUI = True)
        self.folder = folder
        self.energy = energy
        self.pred_condition, self.start_pred_condition = pred_condition, pred_condition
        self.prey_condition, self.start_prey_condition = prey_condition, prey_condition
        self.pred_episodes, self.prey_episodes = None, None
        self.restart()
        self.pred_episodes, self.prey_episodes = hand_episodes(self.env_gui, self.pred, self.prey, energy = 3000)
    
    def restart(self):
      reset_start_time()
      empty_folder(self.folder)
      make_folder(self.folder)
      self.attempts += 1
      self.e = 0
      self.pred = RecurrentTD3()
      self.prey = RecurrentTD3()
      if(self.pred_episodes != None and self.prey_episodes != None):
        self.pred.episodes = self.pred_episodes
        self.prey.episodes = self.prey_episodes
      save_pred_prey(self.pred, self.prey, post = "_0", folder = self.folder)
      self.pred_condition = self.start_pred_condition
      self.prey_condition = self.start_prey_condition
      self.easy_wins = []
      self.med_wins = []
      self.hard_wins = []
      self.easy_wins_rolled = []
      self.med_wins_rolled = []
      self.hard_wins_rolled = []
      self.losses = np.array([[None, None, None, None, None, None]])
      
    def one_episode(self, difficulty = "med", push = True):
      if(difficulty == "easy"): min_dif = 0; max_dif = 0
      if(difficulty == "med"):  min_dif = 0; max_dif = 100
      if(difficulty == "hard"): min_dif = 100; max_dif = 100
      
      GUI = keyboard.is_pressed('q') 
      if(GUI): env = self.env_gui
      else:    env = self.env
      
      to_push_pred, to_push_prey, pred_win, rewards = episode(
          env, self.pred, self.prey, min_dif = min_dif, max_dif = max_dif, energy = self.energy,
          pred_condition = self.pred_condition, prey_condition = self.prey_condition, GUI = GUI)
      
      if(push):
          for i in range(len(to_push_pred)):
              self.pred.episodes.push(to_push_pred[i])
              self.prey.episodes.push(to_push_prey[i])
          
      return(int(pred_win), rewards)


    def epoch(self):
      win, rewards = self.one_episode("easy")
      self.easy_wins.append(win)
      self.easy_wins_rolled.append(get_rolling_average(self.easy_wins))
      if(keyboard.is_pressed('q') ): plot_rewards(rewards)
    
      win, rewards = self.one_episode("med")
      self.med_wins.append(win)
      self.med_wins_rolled.append(get_rolling_average(self.med_wins))
    
      win, rewards = self.one_episode("hard")
      self.hard_wins.append(win)
      self.hard_wins_rolled.append(get_rolling_average(self.hard_wins))
    
      if(type(self.pred_condition) in [int, float]):
          self.pred_condition *= .99
      if(type(self.prey_condition) in [int, float]):
          self.prey_condition *= .99
      
      iterations = 4
      pred_losses = self.pred.update_networks(batch_size = 32, iterations = iterations)
      #prey_losses = self.prey.update_networks(batch_size = 32, iterations = iterations)
      if(iterations == 1): 
          pred_losses = np.expand_dims(pred_losses,0)#; prey_losses = np.expand_dims(prey_losses,0)
      self.losses = np.concatenate([self.losses, np.concatenate([pred_losses, pred_losses], axis = 1)])



    def train(
            self, 
            max_epochs = 1000,
            restarts = ((500, .1, .1, .1), (1000, .5, .3, .2)),
            done = ("pred", .99, .99, .95)):
        
        self.pred.train(); self.prey.train()
        while(self.e < max_epochs):
            self.e += 1
            if(self.e % 5 == 0):  print("Epoch {}, {} attempt(s). {}.".format(self.e, self.attempts, duration()))
            if(self.e % 25 == 0): print("\n\nPredator condition: {}. Prey condition: {}.\n".format(
                    self.pred_condition, self.prey_condition))
            self.epoch()
            if(self.e % 25 == 0): 
                plot_wins(self.easy_wins_rolled, self.med_wins_rolled, self.hard_wins_rolled, name = "wins_{}".format(self.e))
                plot_losses(self.losses, too_long = 300)
                save_pred_prey(self.pred, self.prey, post = "_{}".format(self.e), folder = self.folder)
            
            for r in restarts:
                if(self.e >= r[0]):
                    if(self.easy_wins_rolled[-1] < r[1] or
                       self.med_wins_rolled[-1]  < r[2] or
                       self.hard_wins_rolled[-1] < r[3]):
                        print("This isn't working. Starting again!")
                        delete_these(True, self.pred, self.prey, self.easy_wins, self.med_wins, self.hard_wins,
                           self.easy_wins_rolled, self.med_wins_rolled, self.hard_wins_rolled, self.losses)
                        self.restart()
            
            if(type(self.pred_condition) not in [int, float] or self.pred_condition < .05):
                if(done[0] == "pred"):
                    if(self.easy_wins_rolled[-1] >= done[1] and
                       self.med_wins_rolled[-1]  >= done[2] and
                       self.hard_wins_rolled[-1] >= done[3]):
                        print("\n\nFinished!\n\n")
                        print("\n\nPredator condition: {}. Prey condition: {}.\n".format(
                            self.pred_condition, self.prey_condition))
                        plot_wins(self.easy_wins_rolled, self.med_wins_rolled, self.hard_wins_rolled, name = "wins_last".format(self.e))
                        plot_losses(self.losses, too_long = None, name = "losses".format(self.e))
                        break
    
    def test(self, size = 100):
      self.pred.eval(); self.prey.eval()
      pred_wins = 0
      for i in range(size):
          w, _ = self.one_episode(difficulty = "hard", push = False)
          pred_wins += w
      print("Predator wins {} out of {} games ({}).".format(pred_wins, size, round(100*(pred_wins/size))))
    

# Train!
trainer = Trainer("empty_arena.png", energy = 3000, pred_condition = 1, prey_condition = "pin")
trainer.train()
trainer.test()