### A few utilities
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # I don't know what this is, but it's necessary. 

file_1 = r"C:\Users\tedjt\Desktop\pred_prey"  # When I move here, I cannot restart my kernel? 
file_2 = r"C:\Users\tedjt"                              # When I move here, I CAN restart my kernel? 

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if(device.type == "cpu"):   print("\n\nAAAAAAAUGH! CPU! >:C\n")
else:                       print("\n\nUsing CUDA! :D\n")

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

# Track seconds starting right now. 
import datetime
start_time = datetime.datetime.now()
def duration():
    change_time = datetime.datetime.now() - start_time
    change_time = change_time - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)