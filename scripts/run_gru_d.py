from sacred import Experiment
import torch
import numpy as np
import pandas as pd
import os
import math
import warnings
import itertools
import numbers
# import torch.utils.data as utils
from torch.utils.data import TensorDataset, DataLoader
#import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score

import sys
sys.path.insert(0, '../src')
# from data_loader_oo import DataContainer
from GRUD_model import grud_model

### NEW IMPORTS FOR MEDGEN ###


ex = Experiment("GRU-D-mean")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#TODO: data gets padded before, does this have any effect if i just take the last step output as output?

def fit(model, criterion, l2_penalty, learning_rate, \
        train_dataloader, val_dataloader, test_dataloader, \
        learning_rate_decay=0, n_epochs=30, checkpoint_path = 'model_checkpoints', sacred_run=None):
    
    test_freq = int(len(train_dataloader) / 4) # TODO: more sensible validation criterion/freq
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    # to check the update 
    old_state_dict = {}
    for key in model.state_dict():
        old_state_dict[key] = model.state_dict()[key].clone()
    

    AU_PRCs = list()
    n_batch = 0
    for epoch in range(n_epochs):
        print("starting Epoch: {}".format(epoch))
        if learning_rate_decay != 0:

            # every [decay_step] epoch reduce the learning rate by half
            if  epoch % learning_rate_decay == 0:
                learning_rate = learning_rate/2
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay= l2_penalty)
                print('at epoch {} learning_rate is updated to {}'.format(epoch, learning_rate))
        
        # train the model
        losses, acc = [], []
        label, pred = [], []
        y_pred_col= []
        
        
        #get a minibatch
        # for train_data, train_label, train_num_obs in train_dataloader:
        for train_data, train_label in train_dataloader:
            model.train()
            #push current sample to GPU or CPU
            # train_data, train_label, train_num_obs = train_data.to(device), train_label.to(device), train_num_obs.to(device)
            train_data, train_label= train_data.to(device), train_label.to(device)


            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass : Compute predicted y by passing train data to the model
            y_pred = model(train_data)[:,-1,:] #last output

            #with pad after:
            # y_pred = model(train_data)
            #print(y_pred.size())
            y_pred = torch.squeeze(y_pred)
            #print(y_pred.size())
            # y_pred = torch.gather(y_pred,1,train_num_obs.long())
            #print(y_pred.size())
            #print(train_num_obs.size())

            
            #print(train_num_obs)


            # Compute loss
            loss = criterion(y_pred, train_label)
            #print(loss)
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    train_label)
            )
            losses.append(loss.item())
            print(str(n_batch)+" loss: "+ str(loss.item()) )
            #print(model.w_dg_x.weight)
            #print(model.w_dg_x.bias)

            # perform a backward pass, and update the weights.
            loss.backward()
            optimizer.step()

            #sacred_run.log_scalar('loss', loss, n_batch)
            #sacred_run.log_scalar('acc', acc, n_batch)
            n_batch = n_batch + 1

            if n_batch % test_freq == 0:
                print("Validate...")
                losses, acc = [], []
                label, pred = np.array([]), np.array([])
                model.eval()

                # for val_data, val_label, dev_num_obs in val_dataloader:
                for val_data, val_label in val_dataloader:

                    # val_data, val_label, dev_num_obs = val_data.to(device), val_label.to(device), dev_num_obs.to(device)
                    val_data, val_label = val_data.to(device), val_label.to(device)

                    print(val_data.size())
                    print(val_label.size())
                    # print(dev_num_obs.size())

                    optimizer.zero_grad()
                    # Forward pass : Compute predicted y by passing train data to the model
    
                    y_pred = model(val_data)[:,-1,:] #last output
                    # y_pred = model(val_data)
                    y_pred = torch.squeeze(y_pred)
                    #print(y_pred.size())
                    # y_pred = torch.gather(y_pred,1,dev_num_obs.long())

                    # Compute loss
                    loss = criterion(y_pred, val_label)
                    acc.append(
                        torch.eq(
                            (torch.sigmoid(y_pred).data > 0.5).float(),
                            val_label)
                    )
                    losses.append(loss.item())

                    label = np.append(label, val_label.detach().cpu().numpy())
                    pred = np.append(pred, y_pred.detach().cpu().numpy())


                val_acc = torch.mean(torch.cat(acc).float())
                val_loss = np.mean(losses)
                
                val_pred_out = pred
                val_label_out = label

                va_auc = roc_auc_score(label, pred)
                va_prc = average_precision_score(label, pred)   
                print("validation auc:{}".format(va_auc))
                print("validation prc:{}".format(va_prc))

                sacred_run.log_scalar('va_auc', va_auc, n_batch)
                sacred_run.log_scalar('va_prc', va_prc, n_batch)
                sacred_run.log_scalar('loss', loss.item(), n_batch) #TODO untested

                filename = checkpoint_path+"/GRU_D_epoch_"+str(epoch)+"_step_"+str(n_batch)+".pth"
                print('trying to save file {}'.format(filename) )
                torch.save(model.state_dict(),filename)

                AU_PRCs.append(va_prc)        
                train_loss = np.mean(losses)
                # print("Epoch: {} Train: {:.4f}/{:.2f}%, Dev: {:.4f}/{:.2f}%, Test: {:.4f}/{:.2f}% AUC: {:.4f}".format(
                #     epoch, train_loss, train_acc*100, val_loss, val_acc*100, test_loss, test_acc*100, auc_score))
                print("Epoch: {} Train loss: {:.4f}, Dev loss: {:.4f}, Val PRC: {:.4f}".format(
                    epoch, train_loss, val_loss, va_prc))

        
        train_acc = torch.mean(torch.cat(acc).float())
        train_loss = np.mean(losses)
        
        train_pred_out = pred
        train_label_out = label
        
        # save new params
        new_state_dict= {}
        for key in model.state_dict():
            new_state_dict[key] = model.state_dict()[key].clone()
            
        # compare params
        for key in old_state_dict:
            if (old_state_dict[key] == new_state_dict[key]).all():
                print('Not updated in {}'.format(key))
   
        """
        # dev loss
        print("Validate...")
        losses, acc = [], []
        label, pred = np.array([]), np.array([])
        model.eval()
        for val_data, val_label, num_obs in dev_dataloader:
            # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
            #val_data = torch.squeeze(val_data)
            #val_label = torch.squeeze(val_label)
            val_data, val_label, = val_data.to(device), val_label.to(device)

            optimizer.zero_grad()
            # Forward pass : Compute predicted y by passing train data to the model
            #print(val_data.size())
            #print(model.w_dg_x.weight)
            #print(model.w_dg_x.bias)
            y_pred = model(val_data)[:,-1,:] #last output
            #print(y_pred.size())
            
            # Save predict and label
            #pred.append(y_pred.item())
            #label.append(val_label.item())

            # Compute loss
            loss = criterion(y_pred, val_label)
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    val_label)
            )
            losses.append(loss.item())

            # Save predict and label
            #print(val_label.detach().numpy().shape)
            #print(y_pred.detach().numpy().shape)
            label = np.append(label, val_label.detach().cpu().numpy())
            pred = np.append(pred, y_pred.detach().cpu().numpy())
            #print(label.shape)

        val_acc = torch.mean(torch.cat(acc).float())
        #print(val_acc)
        #print(label.shape)
        val_loss = np.mean(losses)
        
        dev_pred_out = pred
        dev_label_out = label

        va_auc = roc_auc_score(label, pred)
        va_prc = average_precision_score(label, pred)   
        print("validation auc:{}".format(va_auc))
        print("validation prc:{}".format(va_prc))

        sacred_run.log_scalar('va_auc', va_auc, n_batch)
        sacred_run.log_scalar('va_prc', va_prc, n_batch)

        AU_PRCs[epoch] = va_prc        
        # print("Epoch: {} Train: {:.4f}/{:.2f}%, Dev: {:.4f}/{:.2f}%, Test: {:.4f}/{:.2f}% AUC: {:.4f}".format(
        #     epoch, train_loss, train_acc*100, val_loss, val_acc*100, test_loss, test_acc*100, auc_score))
        print("Epoch: {} Train loss: {:.4f}, Dev loss: {:.4f}, Val PRC: {:.4f}".format(
            epoch, train_loss, val_loss, va_prc))
        """
        
        # save the parameters
        train_log = []
        train_log.append(model.state_dict())
        #torch.save(model.state_dict(), './save/grud_mean_grud_para.pt')
        
        #print(train_log)
    
    return AU_PRCs        

#____________________________________________________________________________________________________________________________________________________

@ex.config
def gru_d_config():
    #data config
    file_path = "/Users/Simon/Documents/Uni/MasterThesis/datasets/mimiciii/15_mins_test"
    data_sources = ['labs','vitals','covs']

    features_path = '/Users/Simon/Documents/Uni/MasterThesis/datasets/mimiciii/15_mins_test/X_6hrs_15min_100_vent_bin.npz'
    labels_path = '/Users/Simon/Documents/Uni/MasterThesis/datasets/mimiciii/15_mins_test/y_6hrs_15min_100_vent_bin.npz'
    

    #model config
    hidden_size = 100
    n_layers = 1

    #training config
    n_epochs = 40
    batch_size = 10
    learning_rate = 0.001   
    learning_rate_decay = 7
    dropout = 0.1
    l2_penalty = 0.001

def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def reshape_for_grud_d(X):
    """
    Args:
        X: dictionary containing features, missingness mask and delta_t arrays
    Retruns:
        train_data, val_data, test_data: [N_patients, [input, m_mask, delta_t], features, time_len]
    """
    train_data = np.stack((X['X_train'], X['m_train'], X['delta_t_train']), axis=1)
    val_data = np.stack((X['X_val'], X['m_val'], X['delta_t_val']), axis=1)
    test_data = np.stack((X['X_test'], X['m_test'], X['delta_t_test']), axis=1)

    return train_data, val_data, test_data

@ex.main
def run_gru_d(_run, file_path, features_path, labels_path, data_sources, batch_size, n_epochs, hidden_size, n_layers, learning_rate, learning_rate_decay, l2_penalty, dropout):
    X = np.load(features_path)
    y = np.load(labels_path)


    #setup data
    # data = DataContainer(file_path,data_sources) #problem: need same padding for train and va data
    # va_data = data.va_data_gru_d()
    # tr_data = data.tr_data_gru_d()
    train_data, val_data, test_data = reshape_for_grud_d(X)

    # x_mean = tr_data[:,0,:,:].mean(axis=0).mean(axis=1)
    x_mean = train_data[:,0,:,:].mean(axis=(0,2))

    # train_dataset = torch.utils.data.TensorDataset(torch.Tensor(tr_data),torch.Tensor(np.expand_dims(data.labels_tr, axis =1)),torch.Tensor(np.expand_dims(data.num_obs_times_tr-1, axis =1)))
    # val_dataset = torch.utils.data.TensorDataset(torch.Tensor(va_data), torch.Tensor(np.expand_dims(data.labels_va, axis =1)),torch.Tensor(np.expand_dims(data.num_obs_times_va-1, axis =1)))

    train_dataset = TensorDataset(torch.Tensor(train_data), torch.Tensor(y['y_train']))
    val_dataset = TensorDataset(torch.Tensor(val_data), torch.Tensor(y['y_val']))
    test_dataset = TensorDataset(torch.Tensor(test_data), torch.Tensor(y['y_test']))
    x_mean = torch.Tensor(x_mean)

    #needs int() bc hyperparameter script passes np.int64 and pytorch doesnt like that
    #TODO add num_obs to dataset, then dataloader 

    train_dataloader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size= int(batch_size), shuffle=True)

    print(train_data.shape)
    print(val_data.shape)
    print('Data loaded!')
    #print(tr_data[1])

    print("Build model with params:")
    print("hidden_size: {}, dropout: {}, num_layers: {}".format(hidden_size,dropout,n_layers))

    #fit the model
    # input_size = 44 # num of variables
    input_size = train_data.shape[2]
    #hidden_size = 100 # same as inputsize
    output_size = 1
    #num_layers = 3 # num of GRU layers (only first layer is GRU-D)
    # seq_len = 1264 # max seq len based on data
    model = grud_model(input_size = input_size, hidden_size= int(hidden_size), output_size=output_size, dropout=dropout, dropout_type='mloss', x_mean=x_mean, num_layers=int(n_layers))
    
  
    use_cuda = torch.cuda.is_available()
    if use_cuda :
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
        print("Running on GPU")

    criterion = torch.nn.BCELoss()

    ### IGNORING WEIGHTED LOSS FOR NOW SINCE WE STRATIFY IN SPLIT ###
    #untested
    #logits=preds, targets=minibatch.O_dupe_onehot, pos_weight=self.class_imb)
    #Get class imbalance (for weighted loss):
    # case_prev = data.labels_tr.sum()/float(len(data.labels_tr)) #get prevalence of cases in train dataset
    # class_imb = torch.tensor(1/case_prev) #class imbalance to use as class weight if losstype='weighted'
    # #TODO then we do not need to apply sigmoid to the model output in the model
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_imb)
     
    

    if len(_run.observers) > 0:
         checkpoint_path = os.path.join(_run.observers[0].dir, 'model_checkpoints')
    else:
        checkpoint_path = 'model_checkpoints'

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    print(model) 


    print('Begin training!')
    AU_PRCs = fit(model, criterion, l2_penalty, learning_rate,\
                   train_dataloader, val_dataloader, val_dataloader,\
                   learning_rate_decay, n_epochs, checkpoint_path, _run)

    
    print(AU_PRCs)
    best_val = np.array(AU_PRCs).max()

    return {'Best Validation AUPRC': best_val}




if __name__ == '__main__':
    ex.run_commandline()