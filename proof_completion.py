import torch
from model import *

proof_length = 15
datafile = './raw_data_15.pt'  #'./raw_data_10.pt'
model_path = './final_model_15_512.pt' #'./results_10/final_model_10_512.pt'
d_model = 512
max_len =   13  #8
num_layers = 2
num_tokens =  6163   #1700
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = torch.load(datafile)

tranformer_model = Prooformer(d_model, max_len, num_layers, num_tokens, device).to(device)
tranformer_model.load_state_dict(torch.load(model_path))
tranformer_model.to(device)
                                 
all_rules = 0
completed_proofs = 0
    
for rule, stack_trace in data.items():
    print(rule)
    start = stack_trace[-1]
    #print(start)
    stack = torch.LongTensor(start).unsqueeze(0).to(device)
    for i in range(len(stack_trace)):
        tgt = torch.LongTensor(stack_trace[i]).unsqueeze(0).to(device)
        output = tranformer_model(stack, tgt)
        stack = torch.argmax(output,dim=2).to(device)
        #print(stack)
        #print(tgt)
        if stack.size(1) == 1:
            if stack.item() == start[0]:
                print("******** proof found ********")
                completed_proofs += 1
                break
    all_rules += 1
    
print(all_rules)
print(completed_proofs)
print(completed_proofs/ all_rules)