from utils import parameters as para, change_para
from train import Trainer



trainer_dict = {    
    "pred_easy" : lambda test: 
        Trainer(
            change_para(
                flowers = 0, pred_condition = 0 if test else 1),
            training_agent = "pred", play_by_hand = False,
            save_folder = "empty_prey_pinned_no_flower" if not test else None,
            pred_load_folder = "empty_prey_pinned_no_flower" if test else None),
    
    "prey_easy" : lambda test: 
        Trainer(
            change_para(
                pred_start = 0, prey_condition = 0 if test else 1), 
            training_agent = "prey", play_by_hand = False,
            restart_if = {"prey" : {400 : .95}},
            done_if =    {"prey" : {200 : .01}},
            save_folder = "empty_prey_alone" if not test else None,
            prey_load_folder = "empty_prey_alone" if test else None),
    
    "test" : lambda test: 
        Trainer(
            change_para(
                pred_condition = 0 if test else 1, prey_condition = 0 if test else 1),
            save_folder = "both_paired" if not test else None,
            pred_load_folder = "empty_prey_pinned_no_flower",
            prey_load_folder = "empty_prey_alone") 
    }





def train(trainer_name):
    trainer = trainer_dict[trainer_name](False)
    trainer.train()
    trainer.close()

def test(trainer_name):
    trainer = trainer_dict[trainer_name](True)
    trainer.test()
    trainer.close()
    

train("pred_easy") 
train("prey_easy")
test("test")