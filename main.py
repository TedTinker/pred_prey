from train import Trainer

# Train!
trainer = Trainer("empty_arena", energy = 3000, pred_condition = 1, prey_condition = "pin",
                  play_by_hand = False,
                  save_folder = "empty_with_prey_pinned", agent_size = .8)
trainer.train()
trainer.close()



trainer = Trainer("big_arena", energy = 4000, pred_condition = 1, prey_condition = "random",
                  play_by_hand = False,
                  save_folder = "big_with_prey_random", 
                  load_folder = "empty_with_prey_pinned",
                  agent_size = .8)
trainer.train()
trainer.close()



trainer = Trainer("final_arena", energy = 4000, pred_condition = 1, prey_condition = 1,
                  play_by_hand = False,
                  save_folder = "final", 
                  load_folder = "big_with_prey_random",
                  agent_size = .8,
                  restart_if = {},
                  done_if =    {})
trainer.train(max_epochs = 500, how_often_to_show_and_save = 100)
trainer.close()



# Test!
trainer = Trainer("empty_arena", energy = 3000, pred_condition = 0, prey_condition = "pin",
                  save_folder = "default", 
                  load_folder = "empty_with_prey_pinned",
                  agent_size = .8)
trainer.test()
trainer.close()

trainer = Trainer("big_arena", energy = 4000, pred_condition = 0, prey_condition = "random",
                  save_folder = "default", 
                  load_folder = "big_with_prey_random",
                  agent_size = .8)
trainer.test()
trainer.close()

trainer = Trainer("final_arena", energy = 4000, pred_condition = 0, prey_condition = 0,
                  save_folder = "default", 
                  load_folder = "final",
                  agent_size = .8)
trainer.test()
trainer.close()