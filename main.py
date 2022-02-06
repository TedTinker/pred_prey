from utils import parameters as para, change_para
from train import Trainer

# Train!

trainer = Trainer(change_para(flowers = 0), 
                  training_agent = "pred", play_by_hand = False,
                  save_folder = "empty_prey_pinned_no_flower")
trainer.train()
trainer.close()



trainer = Trainer(change_para(pred_start = 0, prey_condition = 1), 
                  training_agent = "prey", play_by_hand = False,
                  done_if =    {},
                  save_folder = "empty_prey_alone")
trainer.train(300)
trainer.close()







# Test!
trainer = Trainer(change_para(arena_name = "empty_arena", energy = 3000, flowers = 2,
                              pred_start = 2, pred_condition = 0, 
                              prey_start = 2, prey_condition = "pin"),
                  save_folder = "default", 
                  pred_load_folder = "empty_prey_pinned_no_flower",
                  prey_load_folder = "empty_prey_alone")
trainer.test()
trainer.close()





trainer = Trainer(change_para(flowers = 0),
                  save_folder = "default", 
                  pred_load_folder = "empty_prey_pinned_no_flower")
trainer.test()
trainer.close()