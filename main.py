from utils import parameters as para, change_para
from train import Trainer



trainer_dict = {    
    "pred_easy" : lambda : Trainer(change_para(flowers = 1), 
                      training_agent = "pred", play_by_hand = False,
                      save_folder = "empty_prey_pinned_no_flower"),
    
    "pred_easy_test" : lambda : Trainer(change_para(flowers = 0, pred_condition = 0), 
                      training_agent = "pred", play_by_hand = False,
                      pred_load_folder = "empty_prey_pinned_no_flower"),
    
    "prey_easy" : lambda : Trainer(change_para(pred_condition = "pin", prey_condition = 1), 
                      training_agent = "prey", play_by_hand = False,
                      restart_if = {},
                      done_if =    {"prey" : {200 : .01}},
                      save_folder = "empty_prey_alone"),
    
    "prey_easy_test" : lambda : Trainer(change_para(pred_start = 0, prey_condition = 0), 
                      training_agent = "prey", play_by_hand = False,
                      prey_load_folder = "empty_prey_alone"),
    
    "test" : lambda : Trainer(change_para(pred_condition = 0, prey_condition = 0),
                                  save_folder = "default", 
                                  pred_load_folder = "empty_prey_pinned_no_flower",
                                  prey_load_folder = "empty_prey_alone") 
    }





def train(trainer_name):
    trainer = trainer_dict[trainer_name]()
    trainer.train()
    trainer.close()

def test(trainer_name):
    trainer = trainer_dict[trainer_name]()
    trainer.test()
    trainer.close()
    

train("pred_easy") 
train("prey_easy")
test("test")