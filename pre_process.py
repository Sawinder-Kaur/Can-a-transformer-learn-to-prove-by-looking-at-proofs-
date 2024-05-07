# imports
from metamathpy.database import parse
from metamathpy.proof import *
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from time import perf_counter

from utils import *

"""
Load database (a little slow)
"""
db = parse("set.mm")


target_proof_length = 15

"""
print(db.rules["syl"])

rules, steps, stack = generate_proof_stack_trace(db, db.rules['syl'])

for s in stack:
    print(s)

#print(rules)
#print(steps)


#for step in steps:
#    print(step)
#    print(" ".join(flatten_wff(step)))
#"""

keys = db.rules.keys()





proof_stacks = {}
decisions = []
#data = []
proof_lengths = []
max_stack_length = 0
for key in keys:
    #print(key)
    #print(db.rules[key])
    #print(db.rules[key].consequent.proof)
    if len(db.rules[key].consequent.proof) != 0:
        #rules, steps = verify_proof(db, db.rules[key])
        rules, steps, stack = generate_proof_stack_trace(db, db.rules[key])
        #print(len(stack))
        proof_lengths.append(len(stack))
        if(len(stack) > 1 and len(stack) < target_proof_length):
            proof_stacks[key] = stack
            for s in stack:
                if( len(s) > max_stack_length):
                    max_stack_length = len(s)
                
print("max_stack_length : {}".format(max_stack_length))

"""        
length_unique = list(set(proof_lengths))

# This is the corresponding count for each value
counts = [proof_lengths.count(value) for value in length_unique]
print(length_unique)
print(counts)
plt.bar(length_unique,counts, label = length_unique)
#plt.xlim(0, 7471)
plt.xlim(0, 200)
## max stack length = 7471
plt.savefig('proof_lengths_200.pdf')
#"""
        
### uncomment the following lines to print the proof stacks ###        
"""
count = 0
for key, stack in proof_stacks.items():
    print(db.rules[key])
    print(key)
    for s in stack:
        print(s)
    print("-----------------------------------")
    count += 1
    if count == 20: break
exit()
#"""

        
#print(len(decisions))
#print(len(stacks))
tokens = set([])
for key, stack in proof_stacks.items():
    #print(key)
    for s in stack:
        #print(set(s))
        tokens = tokens.union(set(s))

#print(tokens)
alphabet = list(tokens)
print("Number of tokens : {}".format(len(alphabet)))


#exit()

#print(tokens)

samples = {}
for key, stack in proof_stacks.items():
    enum_stack = []
    for s in stack:
        e_s = []
        for term in s:
            e_s.append(alphabet.index(term))
        enum_stack.append(e_s)
    samples[key] = enum_stack

    
torch.save(samples,"raw_data_"+str(target_proof_length)+".pt")
print("{} items saved".format(len(samples.keys())))

#lookup_table = list(alphabet)

#"""
lookup_table = {}
inverse_lookup_table = {}
for index, token in enumerate(alphabet):
    lookup_table[index] = token
    inverse_lookup_table[token] = index


torch.save(lookup_table,"lookup_table_"+str(target_proof_length)+".pt")
print("{} items saved in lookup_table".format(len(lookup_table)))
torch.save(inverse_lookup_table,"inverse_lookup_table_"+str(target_proof_length)+".pt")
print("{} items saved in lookup_table".format(len(inverse_lookup_table)))
