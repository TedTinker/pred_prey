from utils import parameters as para
from utils import change_para
from train import Trainer

# Train!
trainer = Trainer(para, 
                  training_agent = "pred", play_by_hand = False,
                  save_folder = "empty_with_prey_pinned")
trainer.train()
trainer.close()



trainer = Trainer(change_para(arena_name = "big_arena", energy = 4000, pred_condition = 1, prey_condition = "random"),
                  play_by_hand = False, training_agent = "both",
                  save_folder = "big_with_prey_random", 
                  load_folder = "empty_with_prey_pinned")
trainer.train()
trainer.close()


trainer = Trainer(change_para(arena_name = "big_arena", energy = 4000, pred_condition = 1, prey_condition = 1),
                  play_by_hand = False, training_agent = "prey",
                  restart_if = {"prey" : {500 : .9}},
                  done_if =    {"prey" : {100 : .05}},
                  save_folder = "big", 
                  load_folder = "big_with_prey_random")
trainer.train()
trainer.close()



trainer = Trainer(change_para("final_arena", energy = 4000, pred_condition = 1, prey_condition = 1),
                  play_by_hand = False, training_agent = "both",
                  save_folder = "final", 
                  load_folder = "big",
                  restart_if = {},
                  done_if =    {})
trainer.train(max_epochs = 600, how_often_to_show_and_save = 100)
trainer.close()







# Test!
trainer = Trainer(change_para(arena_name = "empty_arena", energy = 3000, pred_condition = 0, prey_condition = "pin"),
                  save_folder = "default", 
                  load_folder = "empty_with_prey_pinned")
trainer.test()
trainer.close()

trainer = Trainer(change_para(arena_name = "big_arena", energy = 4000, pred_condition = 0, prey_condition = "random"),
                  save_folder = "default", 
                  load_folder = "big_with_prey_random")
trainer.test()
trainer.close()

trainer = Trainer(change_para(arena_name = "big_arena", energy = 4000, pred_condition = 0, prey_condition = 0),
                  save_folder = "default", 
                  load_folder = "big",
                  agent_size = .8)
trainer.test()
trainer.close()

trainer = Trainer(change_para(arena_name = "final_arena", energy = 4000, pred_condition = 0, prey_condition = 0),
                  save_folder = "default", 
                  load_folder = "final",
                  agent_size = .8)
trainer.test()
trainer.close()
