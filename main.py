from utils import parameters as para
from utils import change_para
from train import Trainer

# Train!
trainer = Trainer(para, 
                  training_agent = "pred", play_by_hand = False,
                  save_folder = "empty_with_prey_pinned")
trainer.train()
trainer.close()

trainer = Trainer(change_para(pred_condition = "pin", prey_condition = 1), 
                  training_agent = "prey", play_by_hand = False,
                  done_if =    {"prey" : {200 : .01}},
                  save_folder = "empty_with_pred_pinned")
trainer.train()
trainer.close()








# Test!
trainer = Trainer(change_para(arena_name = "empty_arena", energy = 3000, pred_condition = 0, prey_condition = 0),
                  save_folder = "default", 
                  load_folder = "empty_with_prey_pinned")
trainer.test()
trainer.close()

