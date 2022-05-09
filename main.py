from utils import args, change_args
from train import Trainer



trainer_dict = {    
    "pred_easy" : lambda test: 
        Trainer(
            change_args(
                flowers = 0, pred_condition = 0 if test else 1),
            training_agent = "pred", play_by_hand = False,
            save_folder = "pred_easy" if not test else None,
            pred_load_folder = "pred_easy" if test else None),
    
    "prey_easy" : lambda test: 
        Trainer(
            change_args(
                pred_start = 0, prey_condition = 0 if test else 1), 
            training_agent = "prey", play_by_hand = False,
            restart_if = {"prey" : {400 : .95}},
            done_if =    {"prey" : {200 : .01}},
            save_folder = "prey_easy" if not test else None,
            prey_load_folder = "prey_easy" if test else None),
    
    "pred_prey_easy" : lambda test: 
        Trainer(
            change_args(
                pred_condition = 0 if test else 1, prey_condition = 0 if test else 1),
            save_folder = "pred_prey_easy" if not test else None,
            pred_load_folder = "pred_easy",
            prey_load_folder = "prey_easy"),
        
    "pred_prey_hard" : lambda test: 
        Trainer(
            change_args(
                arena_name = "final_arena", 
                pred_condition = 0 if test else 1, 
                prey_condition = 0 if test else 1),
            restart_if = {},
            done_if =    {"pred" : {200 : .0},
                          "prey" : {200 : 1}},
            save_folder = "pred_prey_hard" if not test else None,
            pred_load_folder = "pred_easy" if not test else "pred_prey_hard",
            prey_load_folder = "prey_easy" if not test else "pred_prey_hard") 
    }





def train(trainer_name):
    trainer = trainer_dict[trainer_name](False)
    trainer.train()
    trainer.close()

def test(trainer_name):
    trainer = trainer_dict[trainer_name](True)
    trainer.test()
    trainer.close()
    
"""
train("pred_easy") 
train("prey_easy")
test("pred_prey_easy")
"""

train("pred_prey_hard")
test("pred_prey_hard")