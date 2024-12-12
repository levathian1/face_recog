import csv
import pandas as pd
import numpy as np
import torch
 
def load_tensor(name):
    tensor = torch.load('tensors/' + name + ".txt")

    return tensor

def add_to_db(name, tensor):
    print(tensor)
    content = f'{name}  {tensor}'
    torch.save(tensor, 'tensors/' + name + ".txt")
    
    return content

# print(read_db())