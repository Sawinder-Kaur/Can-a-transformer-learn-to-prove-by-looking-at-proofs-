import torch
from dataloader_appended import *
import matplotlib.pyplot as plt
import numpy as np

from model import *
import statistics
import copy


def test_accuracy(model, loader, lookup_table,device):
    total = 0
    correct = 0
    for (src,tgt) in loader: 
        src, tgt = torch.LongTensor(src).unsqueeze(0).to(device), torch.LongTensor(tgt).unsqueeze(0).to(device)
        output = model(src,tgt)
        #print(output)
        predict = torch.argmax(output,dim=2).to(device)
        correct += torch.prod((predict == tgt)).item()
        total += src.size(0)
    accuracy = correct*100/total
    return accuracy
    
    

    
def train(model, best_model, train_loader,test_loader,lookup_table, device, num_epochs = 150, lr = 0.00005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    
    train_acc = []
    test_acc = []
    best_accuracy = 0
    best_testaccuracy = 0
    for e in range(num_epochs):
        total = 0
        correct = 0
        epoch_loss = 0
        print("********** Epoch : {} **********".format(e))
        for (src,tgt) in train_loader: 
            src, tgt = torch.LongTensor(src).unsqueeze(0).to(device), torch.LongTensor(tgt).unsqueeze(0).to(device)
            #print(tgt.size())
            
            
            
            output = model(src,tgt)
            
            #print(processed_tgt.size())
            #processed_output = process_output(output, lookup_table, device)
            
            loss = criterion(output[0], tgt[0])
            optimizer.zero_grad()
            loss.backward()
            #print(loss)
            optimizer.step()
            #print(processed_output.eq(tgt))
            predict = torch.argmax(output,dim=2).to(device)
            correct += torch.prod((predict == tgt)).item()
            
            total += src.size(0)
            epoch_loss += loss.item()
        losses.append(epoch_loss/len(train_loader))
        #print(correct)
        #print(total)
        accuracy = correct*100/total
        testaccuracy = test_accuracy(model, test_loader,lookup_table, device)
        train_acc.append(accuracy)
        test_acc.append(testaccuracy)
        print("Train Accuracy : {}".format(accuracy))
        print("Test Accuracy : {}".format(testaccuracy))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_testaccuracy = testaccuracy
            best_model = copy.deepcopy(model)
    plt.plot(np.array(losses), 'r')
    plt.savefig('train_loss_15_128_appended.png')
    
    plt.clf()
    plt.plot(np.array(train_acc), 'r')
    plt.plot(np.array(test_acc), 'b')
    plt.savefig('accuracy_15_128_appended.png')
    return best_model, best_accuracy, best_testaccuracy

def main(seed = 10):
    """
    proof_length = 10
    embedding_size = 512
    
    max_len = 9
    num_tokens = 1700#1646 #1647
    
    """
    ### for proofs of length max 15
    proof_length = 15
    embedding_size = 512
    
    max_len = 14
    num_tokens = 6200#1646 #1647
    #"""
    
    
    d_model = embedding_size   ## == embedding size
    num_layers = 2
    batch_size = 1
    file_name = "./raw_data_"+str(proof_length)+".pt"
    
    
    train_loader, test_loader, lookup_table, inverse_lookup_table = data_loader(datafile = file_name, seed = seed, batch_size = 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #tranformer_model = torch.nn.Transformer(d_model = embedding_size)
    #"""
    tranformer_model = Prooformer(d_model, max_len, num_layers, num_tokens, device).to(device)
    best_model = Prooformer(d_model, max_len, num_layers, num_tokens, device).to(device)
    #loss_fn = torch.nn.CrossEntropyLoss()
    #opt = torch.optim.Adam(model.parameters(), lr=0.00005)
    #"""
    tranformer_model.to(device)
    best_model.to(device)
    
    #print("Train Accuracy of the randomly initialized tranformer : {}".format(test_accuracy(tranformer_model, train_loader)))
    print("Test Accuracy of the randomly initialized tranformer : {}".format(test_accuracy(tranformer_model, test_loader,lookup_table, device)))
    
    best_model, train_acc, test_acc = train(tranformer_model, best_model,train_loader,test_loader,lookup_table, device, lr = 0.00005)
    
    print("Test Accuracy of the trained tranformer : {}".format(test_accuracy(best_model, test_loader,lookup_table, device)))
    
    torch.save(best_model.state_dict(), "./final_model_"+str(proof_length)+"_"+str(embedding_size)+"_appended_6200.pt")
    return train_acc, test_acc


if __name__ == "__main__":
    """
    seed = 1013
    trainacc, testacc = main(seed)
    print("Train Accuracy for seeds : {}".format(trainacc))
    print("Test Accuracy for seeds : {}".format(testacc))
    """
    trainaccuracy = []
    testaccuracy = []
    for seed in [1013, 1234, 8478]:
        trainacc, testacc = main(seed)
        trainaccuracy.append(trainacc)
        testaccuracy.append(testacc)
    print("Test Accuracy for 3 seeds : {}".format(testaccuracy))
    print("Mean Train Accuracy for 3 seeds : {}".format(statistics.mean(trainaccuracy)))
    print("Mean Test Accuracy for 3 seeds : {}".format(statistics.mean(testaccuracy)))  