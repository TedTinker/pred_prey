### A few utilities
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # I don't know what this is, but it's necessary. 



import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if(device.type == "cpu"):   print("\n\nAAAAAAAUGH! CPU! >:C\n")
else:                       print("\n\nUsing CUDA! :D\n")



# Track seconds starting right now. 
import datetime
start_time = datetime.datetime.now()
def reset_start_time():
    global start_time
    start_time = datetime.datetime.now()
def duration():
    change_time = datetime.datetime.now() - start_time
    change_time = change_time - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)




# Monitor GPU memory.
def get_free_mem(string = ""):
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print("\n{}: {}.\n".format(string, f))

# Remove from GPU memory.
def delete_these(verbose = False, *args):
    if(verbose): get_free_mem("Before deleting")
    del args
    torch.cuda.empty_cache()
    if(verbose): get_free_mem("After deleting")
  
  
  
# How to get an input from keyboard.
def get_input(string, okay, default = None):
    if(type(okay) == list): okay = {i+1:okay[i] for i in range(len(okay))}
    while(True):
        inp = input("\n{}\n{}\n".format(string, okay))
        if(inp == "" and default != None): 
          if(type(default) == int): 
              print("Default: {}.\n".format(okay[default]))
              return(okay[default])
          else:     
              print("Default: {}.\n".format(default))                
              return(default)
        try: 
            if(inp in okay.values()): return(inp)
            else:                     return(okay[int(inp)])
        except: 
            print("\n'{}' isn't a good answer.".format(inp))
            
            
            
# How to get rolling average.
def get_rolling_average(wins, roll = 100):
    if(len(wins) < roll):
        return(sum(wins)/len(wins))
    return(sum(wins[-roll:])/roll)       
            
            
            
            
            
            
# How to save plots.
import matplotlib.pyplot as plt
import os
import shutil

def remove_folder(folder):
    files = os.listdir("saves")
    if(folder not in files): return
    shutil.rmtree("saves/" + folder)

def make_folder(folder):
    files = os.listdir("saves")
    if(folder in files): return
    os.mkdir("saves/"+folder)
    os.mkdir("saves/"+folder+"/plots")
    os.mkdir("saves/"+folder+"/preds")
    os.mkdir("saves/"+folder+"/preys")

def save_plot(name, folder = "default"):
    make_folder(folder)
    plt.savefig("saves/"+folder+"/plots/"+name+".png")
  
  
  
  
  
  
  
  
# How to plot an episode's rewards.
def plot_rewards(rewards, name = None, folder = "default"):
    total_length = len(rewards)
    x = [i for i in range(1, total_length + 1)]
    plt.plot(x, [r[0] for r in rewards], color = "lightcoral", label = "Predator") # Predator
    plt.plot(x, [r[1] for r in rewards], color = "turquoise", label = "Prey")  # Prey
    plt.legend(loc = 'upper left')
    plt.title("Rewards")
    plt.xlabel("Time")
    plt.ylabel("Reward")
    if(name!=None): save_plot(name, folder)
    plt.show()
    plt.close()
  
  
# How to plot losses.
def plot_losses(pred_losses, prey_losses, too_long = None, name = None, folder = "default"):
  
    pred_length = len(pred_losses)
    pred_x = [i for i in range(1, pred_length + 1)]
    if(too_long != None and pred_length > too_long):
        pred_x = pred_x[-too_long:]; pred_losses = pred_losses[-too_long:]
    pred_actor_x  = [x_ for i, x_ in enumerate(pred_x) if pred_losses[i][0] != None]
    pred_actor_y = [l[0] for l in pred_losses if l[0] != None]
    pred_critic_x = [x_ for i, x_ in enumerate(pred_x) if pred_losses[i][1] != None]
    pred_critic_1_y = [l[1] for l in pred_losses if l[1] != None]
    pred_critic_2_y = [l[2] for l in pred_losses if l[2] != None]
    
    if(len(pred_critic_x) >= 1):
        fig, ax1 = plt.subplots() 
        ax2 = ax1.twinx() 
        ax1.plot(pred_actor_x, pred_actor_y, color = "#ff0000", label = "Actor")
        ax2.plot(pred_critic_x, pred_critic_1_y, color = "lightcoral", linestyle = "--", label = "Critic")
        ax2.plot(pred_critic_x, pred_critic_2_y, color = "lightcoral", linestyle = ":", label = "Critic")
        ax1.legend(loc = 'upper left')
        ax2.legend(loc = 'lower left')
        plt.title("Predator Losses")
        plt.xlabel("Training iterations")
        ax1.set_ylabel("Actor losses")
        ax2.set_ylabel("Critic losses")
        if(name!=None): save_plot("pred"+name, folder)
        plt.show()
        plt.close()
    
    prey_length = len(prey_losses)
    prey_x = [i for i in range(1, prey_length + 1)]
    if(too_long != None and prey_length > too_long):
        prey_x = prey_x[-too_long:]; prey_losses = prey_losses[-too_long:]
    prey_actor_x  = [x_ for i, x_ in enumerate(prey_x) if prey_losses[i][0] != None]
    prey_actor_y = [l[0] for l in prey_losses if l[0] != None]
    prey_critic_x = [x_ for i, x_ in enumerate(prey_x) if prey_losses[i][1] != None]
    prey_critic_1_y = [l[1] for l in prey_losses if l[1] != None]
    prey_critic_2_y = [l[2] for l in prey_losses if l[2] != None]
    
    if(len(prey_critic_x) >= 1):
        fig, ax1 = plt.subplots() 
        ax2 = ax1.twinx() 
        ax1.plot(prey_actor_x, prey_actor_y, color = "#0000ff", label = "Actor")
        ax2.plot(prey_critic_x, prey_critic_1_y, color = "turquoise", linestyle = "--", label = "Critic")
        ax2.plot(prey_critic_x, prey_critic_2_y, color = "turquoise", linestyle = ":", label = "Critic")
        ax1.legend(loc = 'upper left')
        ax2.legend(loc = 'lower left')
        plt.title("Prey Losses")
        plt.xlabel("Training iterations")
        ax1.set_ylabel("Actor losses")
        ax2.set_ylabel("Critic losses")
        if(name!=None): save_plot("prey"+name, folder)
        plt.show()
        plt.close()
  
  
  
  
  
# How to plot predator victory-rates.
def plot_wins(win_easy, win_med, win_hard, max_len = None, name = None, folder = "default"):
    total_length = len(win_easy)
    x = [i for i in range(1, len(win_easy)+1)]
    if(max_len != None and total_length > max_len):
        x = x[-max_len:]
        win_easy = win_easy[-max_len:]
        win_med = win_med[-max_len:]
        win_hard = win_hard[-max_len:]
    plt.plot(x, win_easy, color = "turquoise", label = "Easy for predator")
    plt.plot(x, win_med, color = "gray", label = "Medium")
    plt.plot(x, win_hard, color = "lightcoral", label = "Hard")
    plt.ylim([0, 1])
    plt.legend(loc = 'upper left')
    plt.title("Predator win-rates")
    plt.xlabel("Epochs")
    plt.ylabel("Predator win-rate")
    if(name!=None): save_plot(name, folder)
    plt.show()
    plt.close()
  
  
  

  
  
  
  
  
  
# How to save/load pred/prey

def save_pred_prey(pred, prey, post = "", folder = "default"):
    torch.save(pred.state_dict(), "saves/" + folder + "/preds/pred_{}.pt".format(post))
    torch.save(prey.state_dict(), "saves/" + folder + "/preys/prey_{}.pt".format(post))

def load_pred_prey(pred, prey, post = "last", folder = "default"):
    pred.load_state_dict(torch.load("saves/" + folder + "/preds/pred_{}.pt".format(post)))
    prey.load_state_dict(torch.load("saves/" + folder + "/preys/prey_{}.pt".format(post)))
    return(pred, prey)