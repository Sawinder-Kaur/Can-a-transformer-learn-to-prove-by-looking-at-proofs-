import torch
import random

from torch.utils.data import DataLoader
import math 



def data_loader(datafile = "./raw_data_10.pt", lookup_file = "./lookup_table_10.pt", inverse_lookup_file = "./inverse_lookup_table_10.pt", batch_size = 1, seed = 10):
    
    
    random.seed(seed)
    data = torch.load(datafile)
    lookup_table = torch.load(lookup_file)
    inverse_lookup_table = torch.load(inverse_lookup_file)
    #print(lookup_table)
    num_rules = len(data.keys())

    samples = []
    targets = []
    count = 0
    for _, stack in data.items():
        sample = []
        target = []
        rule = stack[-1]
        for s in stack:
            #pad_len = 8 - len(s)
            #target.append(s + ([1699] * pad_len))
            target.append(rule + s)
        sample.append(target[-1])
        if len(target) > 1: sample += target[:-1]
        #print(sample)
        #print(target)
        #print("----------------------------------")
        samples += sample
        targets += target

    ### uncomment the following lines to see the tensor data
    """    
    for (x,y) in zip(samples,targets):
        print(x)
        print(y)
        print("------------------------")
        count+=1
        if count == 20: break 
    """

    dataset = []
    for (x,y) in zip(samples,targets):
        dataset.append([x,y])

    dataset_len = len(dataset)
    print("Total dataset length : {}".format(dataset_len))


    ### the test set is not disjoint for now
    test_len = math.floor(0.2 * dataset_len)
    testset = random.sample(dataset, test_len)
    print("Testset length : {}".format(test_len))

    for data in testset:
         dataset.remove(data) 
    print("Trainset length : {}".format(len(dataset)))
    

    #train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    #print("******* dataloaders created ********" )
    
    #return train_dataloader, test_dataloader, lookup_table#, inverse_lookup_table
    return dataset, testset, lookup_table, inverse_lookup_table
