import torch as th
from torch.nn import Linear, ReLU, Sigmoid, LeakyReLU, GELU, Module, init
from torch_geometric.nn import Sequential, ChebConv, GCNConv, GraphConv
from data import SIR_simulation_graph
import numpy as np
import math
import matplotlib.pyplot as plt
from torch import nn
import networkx as nx
import pickle
import time
import copy
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN, GConvGRU, A3TGCN, GConvLSTM
import graph_utils
import argparse
from typing import Tuple, Union
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import ( Adj, OptPairTensor, OptTensor, Size, SparseTensor,)
import json

device = 'cuda'
#train_points = int(100)
#valid_points = int(100 * 1.25)
#test_points = int(100 * 2.25)
M = 10000000000
dt = 1/50 # for the graph simulation

def read_input(seed, dataset):
    """ returns the dataset of snapshots as (nodes, timesteps, features), snapshots derivatives and the torch geometric edge_index and edge weight for the neighbours and neighbours of neighbours """
    # recover input files for this seed
    with open('./'+dataset+'/in/G.gpickle', 'rb') as f:
        G = pickle.load(f)
    max_degree = max([d for n, d in G.degree()])
    iterations = []
    with open(r'./'+dataset+'/in/signal'+str(seed)+'.data', 'rb') as fp:
        data = fp.read()
        iterations = pickle.loads(data)

    edge_index_single = np.array([list(e) for e in G.edges], dtype=np.int64).T # collect edges ends in 2 separate lists
    edge_index = np.array([np.concatenate((edge_index_single[0], edge_index_single[1])), np.concatenate((edge_index_single[1], edge_index_single[0]))]) # add the other end of each edge
    edge_weight = np.ones(edge_index.shape[1])

    iterations_features = []
    current_states = np.zeros((1000, 3))
    for i,iteration in enumerate(iterations):
        for node, state in iteration['status'].items(): 
            current_states[node, :] = np.zeros(3)
            current_states[node, state] = 1
        iterations_features.append(current_states.copy())
    iterations_features = np.array(iterations_features)
    # apply filter to input features in the time dimension
    #iterations_features = np.apply_along_axis(lambda m: np.convolve(m, np.ones(2)/2, 'same'), axis=0, arr=iterations_features)
    # I predict 16 successive time steps
    target = np.roll(iterations_features, -1, axis=0)

    iterations_features = th.tensor(iterations_features, dtype=th.float32, device=device)    
    edge_index = th.tensor(edge_index, dtype=th.int64, device=device)
    edge_weight = th.tensor(edge_weight, dtype=th.float32, device=device)
    target = th.tensor(target, dtype=th.float32, device=device)

    return th.mean(iterations_features, 1), iterations_features, edge_index, edge_weight, target, th.mean(target, 1), max_degree

def early_stopping(model, stop_metric, best_stop_metric, best_model, best_epoch, countdown_epochs, tolerance, epoch):
    #if stop_metric < best_stop_metric : 
    # apply early stopping after waiting some epochs for some improvement(comparison in lexicographic order)
    if (stop_metric[0] < best_stop_metric[0]) or (stop_metric[0] == best_stop_metric[0] and stop_metric[1] < best_stop_metric[1]): 
        best_stop_metric = stop_metric
        countdown_epochs = tolerance
        best_model = copy.deepcopy(model.state_dict())
        best_epoch = epoch
    else: countdown_epochs -= 1
    #if countdown_epochs == tolerance-1:    # save model if it starts to degrade
    #    best_model = copy.deepcopy(model.state_dict())
    #    best_epoch = epoch
    return best_model, best_epoch, best_stop_metric, countdown_epochs

def train_cycle(params, model, get_model_output, get_loss, get_model_constraints, criterion, learning_rate, rand_sampling, init_snap, aggr_val, \
                epoch, best_model, best_loss_val, best_epoch, countdown_epochs, \
                train_evolution, val_evolution, constr_change):
    """Do a training cycle using the constraints or not until the early stopping criterion is met"""

    train_points, valid_points, test_points = params["train_points"], params["valid_points"], params["test_points"]
    constrain, tolerance_epochs = params["constrain"], params["tolerance_epochs"]
    loss_constr, err_train, err_val = M, M, M
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
    if constrain: constr_opt = graph_utils.constrained_opt(get_model_constraints, model.parameters(), rand_sampling[0], learning_rate, params["dual_rate_multiplier"])

    epochs_window = 0
    while countdown_epochs > 0 and epochs_window < 8000: 
        if constrain:
            # apply one step of the constrained optimizer telling which function generates the current loss
            #init_conds = init_cond_opt.opt_step() # get most violating init conditions and use them to constrain the main problem
            constr_opt.opt_step(get_loss, None)
        else:
            # if I don't use constraints use simply adam as optimizer
            loss = get_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calculate validation loss for early stopping
        # validate by integrating the model up until the validation set and calculate the mape
        if epoch%10 == 0 or countdown_epochs==tolerance_epochs:   # only once every 10 epochs as it is expensive
            err_train = get_loss()
            pred = Model_regression.forward_recurrent(get_model_output, init_snap[None,:,:], valid_points)
            err_val = criterion(th.mean(pred, (1,2))[train_points:,:], aggr_val)
            loss_constr = graph_utils.constr_loss(get_model_constraints, rand_sampling[1])
        # to early stop the val error is used, if I constrain the loss constraint is used instead but only after some epochs to avoid the trivial initial model
        #early_stop_metric = err_val if not constrain else (loss_constr if epoch > int(tolerance_epochs/2) else M)
        early_stop_metric = (0,err_val) if not constrain else ((max(loss_constr,5e-4*dt), err_val) if epoch > int(tolerance_epochs/2) else (M,M))
        best_model, best_epoch, best_loss_val, countdown_epochs = early_stopping(model, early_stop_metric, best_loss_val, best_model, best_epoch, countdown_epochs, tolerance_epochs, epoch)

        # record error evolution
        if epoch %1000==0: print("({:d}k, {:1.4f})".format(int(epoch/1000), float(err_val.cpu().detach().numpy())), end="", flush=True)
        train_evolution.append(float(err_train.cpu().detach().numpy()))
        val_evolution.append(float(err_val.cpu().detach().numpy()))
        constr_change.append(float(loss_constr.detach().cpu().numpy()/dt))

        epochs_window += 1
        epoch += 1

    return  best_model, best_loss_val, best_epoch, epoch, countdown_epochs

def train_node_supervision(params, model, snap_train, aggr_val, rand_sampling, edge_index, edge_weight, target, learning_rate, dense_nodes, pretrain, reduced_features, dense_layers):
    """ train a generic recurrent model to predict given one snapshot the next one """
    train_points, valid_points, test_points, dual_rate_multiplier = params["train_points"], params["valid_points"], params["test_points"], params["dual_rate_multiplier"]

    best_stop_metric = (M,M) # best metric starting from a threshold
    model.train()
    cost = th.tensor(0.0)
    epoch = 0
    tolerance = params["tolerance_epochs"]
    countdown_epochs = tolerance
    best_epoch = 0
    best_model = copy.deepcopy(model.state_dict())
    train_evolution, val_evolution, constr_change, phases_changes = [],[],[],[]
    criterion = nn.MSELoss()

    # get the model output for the input at each timestep
    def get_model_output(input):
        y_hat = th.zeros(input.size(), dtype=th.float32, device=device)
        y_hat = model(input, edge_index, edge_weight, None, pretrain, reduced_features, dense_layers)
        return y_hat

    # bypass the aggregation phase of the graph conv and give directly the constraints' measures
    def get_model_constraints(rand_measures):
        y_hat = th.zeros(rand_measures.size(0), rand_measures.size(1), int(rand_measures.size(2)/2), dtype=th.float32, device=device)
        y_hat[:,:,:] = model(rand_measures[:,:,:3], edge_index, edge_weight, rand_measures[:,:,3:], pretrain, False, dense_layers)
        y_hat = (y_hat - rand_measures[:,:,:3]) # difference without dt because it unbalances the objective constrained function 
        return y_hat

    # define how to get the loss for the constr optimizer
    def get_train_loss():
        y_hat = get_model_output(snap_train)
        return criterion(y_hat, target[:train_points,:,:])

    best_model, best_stop_metric, best_epoch, epoch, countdown_epochs = \
                train_cycle(params, model, get_model_output, get_train_loss, get_model_constraints, criterion, learning_rate, rand_sampling, snap_train[0,:,:], aggr_val, \
                epoch, best_model, best_stop_metric, best_epoch, countdown_epochs, \
                train_evolution, val_evolution, constr_change)

    phases_changes.append((best_epoch, "best epoch"))
    phases_changes.append((epoch, "phase end"))  # save phase change position
    # take best model
    model.load_state_dict(copy.deepcopy(best_model))
    print(best_epoch,end="",flush=True)
    loss_constr = graph_utils.constr_loss(get_model_constraints, rand_sampling[1])/dt # calculate constr error (dt was not divided before)

    return model, train_evolution[best_epoch], loss_constr, (train_evolution,phases_changes), (val_evolution,phases_changes), (constr_change, phases_changes)

def train_aggr_supervision(params, model, init_snap, aggr_train, aggr_val, aggr_target, rand_sampling, edge_index, edge_weight, init_epoch, learning_rate, dense_nodes, pretrain, dense_layers, bootstrap_increment):
    """ train without supervision for each node, but only aggregated information obtained from all nodes, return trained model and train loss,validation error evolutions \n
        works in 2 phases: \n
        (1) learning using appoximated teacher forcing every n steps (n gradually increasing) \n
        (2) fine tuning applying thresholding with no teacher forcing"""
    train_points, test_points, tolerance_epochs, bootstrap_increment = params["train_points"], params["test_points"], params["tolerance_epochs"], params["bootstrap_increment"]

    # keep best model for early stopping
    best_stop_metric = (M,M) # best metric starting from a threshold
    countdown_epochs = tolerance_epochs
    best_model = copy.deepcopy(model.state_dict())
    best_epoch, loss_constr = 0,M
    epoch = init_epoch # count epochs effectively done (excluding tolerance epochs done in vain)
    steps_no_forcing = 1 # start from predicting 1 step (complete teacher forcing)
    predictions_accum, target_accum = [],[]
    train_evolution, val_evolution, phases_changes, lambdas_change, constr_change = [],[],[],[],[]
    criterion = nn.MSELoss()

    # initialize the input used for approximated teacher forcing
    forcing = th.zeros((aggr_train.size(0), 1000, aggr_train.size(1)), dtype=th.float32, device=device)
    forcing[:,:,:] = aggr_train[:,None,:]    # give an approximate evolutions by spreading the pandemic equally between nodes

    # bypass the aggregation phase of the graph conv and give directly the constraints' measures
    def get_model_constraints(rand_measures):
        y_hat = th.zeros(rand_measures.size(0), rand_measures.size(1), int(rand_measures.size(2)/2), dtype=th.float32, device=device)
        y_hat[:,:,:] = model(rand_measures[:,:,:3], edge_index, edge_weight, rand_measures[:,:,3:], pretrain, False, dense_layers)
        y_hat = (y_hat - rand_measures[:,:,:3]) # difference without dt because it unbalances the objective constrained function 
        return y_hat

    #phase 1
    while steps_no_forcing <= 60 :# int(train_points) : 
        # the approximated teacher forcing is less and less present, to make it learn its nodes information but avoiding explosions. to do so a forcing step is done 
        # every steps_no_forcing, moreover the training set is shifted many times to augment the data
        #stride = int(math.ceil(steps_no_forcing/4)) # the stride increases as the model is more precise (i augment less and less the data)
        stride = int(train_points/50) if steps_no_forcing>1 else 1  # 50 windows independently of the steps no forcing
        shifts = int((train_points-steps_no_forcing)/stride)
        aggr_shifted = []   # repeat aggr_train shifting according to the stride
        for s in range(shifts): aggr_shifted.append(aggr_target[s*stride:s*stride+steps_no_forcing,:])
        aggr_shifted = th.stack(aggr_shifted, 1)
        lr = learning_rate*20 if steps_no_forcing <=1 else learning_rate

        # get the model output for the input at each timestep
        def get_model_output(input):
            return model(input, edge_index, edge_weight, None, pretrain, False, dense_layers)

        # define how to get the loss for the constr optimizer
        def get_train_loss():
            y_hat = Model_regression.forward_recurrent(get_model_output, forcing[:shifts*stride:stride,:,:], steps_no_forcing)
            preds_aggr = th.mean(y_hat,2)
            loss = criterion(preds_aggr, aggr_shifted)
            #forcing = forcing.detach()  # backpropagate starting from this epoch
            return loss
        
        # train using a smaller dictionary with complete approx forcing, then full, without thresholding
        best_model, best_stop_metric, best_epoch, epoch, countdown_epochs = \
                    train_cycle(params, model, get_model_output, get_train_loss, get_model_constraints, criterion, lr, rand_sampling, init_snap, aggr_val, \
                    epoch, best_model, best_stop_metric, best_epoch, countdown_epochs, \
                    train_evolution, val_evolution, constr_change)

        steps_no_forcing += bootstrap_increment   # increment training set size 
        countdown_epochs = tolerance_epochs     # reset the countdown to start a new phase with larger window
        phases_changes.append((epoch, "approx supervision %i steps"%steps_no_forcing))  # save phase change position
        print("|", end="", flush=True)

    model.load_state_dict(copy.deepcopy(best_model))
    print(best_epoch, end=" \n", flush=True)
    loss_constr = graph_utils.constr_loss(get_model_constraints, rand_sampling[1])/dt # calculate constr error (dt was not divided before)

    return model, train_evolution[best_epoch-init_epoch], loss_constr, train_evolution, val_evolution, constr_change, phases_changes, best_epoch, epoch

# custom implemented because of initialization and constraints
class GraphConv(MessagePassing):
    def __init__( self, in_channels: int, out_channels: int, aggr: str = 'add', bias: bool = True, **kwargs,):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_rel = Linear(in_channels, out_channels, bias=bias)
        self.lin_root = Linear(in_channels, out_channels, bias=bias)
        # weights are initialized to copy the input for node's features and close to 0 for the rest
        init.uniform_(self.lin_rel.weight.data, -0.001,0.001)  
        init.uniform_(self.lin_rel.bias.data, -0.001,0.001)  
        init.uniform_(self.lin_root.bias.data, -0.001,0.001)  
        init.eye_(self.lin_root.weight.data)

    def forward(self, x: Union[th.Tensor, OptPairTensor], edge_index: Adj, measures_neigh, reduced_features, edge_weight: OptTensor = None, size: Size = None) -> th.Tensor:
        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        if isinstance(x, th.Tensor):
            x: OptPairTensor = (x, x)

        # if the neigh measures are given, they are not calculated by aggregation but used directly
        if measures_neigh == None:
            if reduced_features: # train without neighbours features (it is easier)
                measures_neigh = th.zeros(x[1].size(), dtype=th.float32, device=device)
            else:
                measures_neigh = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        
        out = self.lin_rel(measures_neigh)
        x_r = x[1]
        if x_r is not None:
            out = out + self.lin_root(x_r)

        return out

    def message(self, x_j: th.Tensor, edge_weight: OptTensor) -> th.Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptTensor) -> th.Tensor:
        return th.matmul(adj_t, x, reduce=self.aggr)

class Model_regression(Module):
    def __init__(self, nodes):
        super().__init__()
        self.nodes = nodes
        self.conv = GraphConv(3, 3)
        self.act = LeakyReLU()
        self.linear1 = Linear(3, nodes)
        init.eye_(self.linear1.weight.data) # 
        init.uniform_(self.linear1.bias.data, -0.01,0.01)
        self.linear2 = Linear(nodes, nodes)
        init.eye_(self.linear2.weight.data)
        init.uniform_(self.linear2.bias.data, -0.01,0.01)
        self.linear3 = Linear(nodes, nodes)
        init.eye_(self.linear3.weight.data)
        init.uniform_(self.linear3.bias.data, -0.01,0.01)
        self.out = Linear(nodes, 3)
        init.eye_(self.out.weight.data)
        init.uniform_(self.out.bias.data, -0.01,0.01)
    
    # the model predicts the next state starting from the current (not the derivative)
    def forward(self, x, edge_index, edge_weights, measures_neigh, pretrain, reduced_features, layers):
        x = self.conv(x, edge_index, measures_neigh, reduced_features, edge_weights)
        if pretrain==True or layers==0: return x
        else:
            if layers >= 1:
                x = self.act(x)
                x = self.linear1(x)
            if layers >= 2:
                x = self.act(x)
                x = self.linear2(x)
            if layers >= 3:
                x = self.act(x)
                x = self.linear3(x)
            x = self.act(x)
            x = self.out(x)
            return x
    
    def forward_recurrent(forward_fun, initial_snapshots, steps):
        """More than one initial snapshot can be provided, in that case multiple recurrent evolutions are calculated, each starting from a different snapshot.
        The input has shape [n_initial_snaps, nodes, 3], the output has shape [steps, n_initial_snaps, nodes, 3].
        By default the gradient is backpropagated only one step in the past """
        curr_snapshot = initial_snapshots    
        # store a steps long evolution for each initial snapshot
        preds = th.zeros((steps, initial_snapshots.size(0), initial_snapshots.size(1), 3), dtype=th.float32, device=device)
        for t in range(steps):
           curr_snapshot = curr_snapshot.detach()   # gradient backpropagation should start from here
           curr_snapshot = forward_fun(curr_snapshot)
           preds[t,:,:,:] = curr_snapshot[:,:,:]
        
        return preds

    def save_model(path, model):
        """ save model as torch tensor"""
        torch.save(model.state_dict(), path + "_model.data")

    def clone_model(self):
        clone = copy.deepcopy(self.state_dict())
        new = Model_regression(self.nodes).to(device)
        new.load_state_dict(clone)
        return new

    def read_trained_model(folder_path, seed, hyp):
        """Read model and stats for a seed and return train info"""
        with open(folder_path+str(seed)+"_stats.json", "r") as f:
            s = f.read()
            stats_json = json.loads(s)
        
        phases = np.stack([stats_json["phases"], stats_json["phases_names"]]).T
        train_evolution =  (stats_json["train_evolution"] , phases)
        val_evolution = (stats_json["validation_evolution"], phases)
        constr_evolution = (stats_json["constr_evolution"], phases)
        idx = tuple(stats_json["best_idx"])

        model = Model_regression(int(hyp[1][idx])).to(device)
        model.load_state_dict(torch.load(folder_path +str(seed) + "_model.data"))

        return model, stats_json["best_train_err"], stats_json["best_constr_loss"], train_evolution, val_evolution, constr_evolution, [(idx,0)]

def conv_regression_node(params, snap_train, aggr_val, rand_sampling, edge_index, edge_weight, target, learning_rate, dense_nodes, dense_layers):
    """ predict the next step by using a cheb graph convolution """
    node_features = 3
    model = Model_regression(dense_nodes).to(device)

    # pretrain on the conv layer
    # first only node's features are used to help the train process
    #model, pred, train_error, constr_err, train_evolution, val_evolution, constr_change = train_node_supervision(params, model, snap_train, aggr_val, rand_sampling, edge_index, edge_weight, target, learning_rate, dense_nodes, True, True, 0)
    model, train_error, constr_err, train_evolution, val_evolution, constr_change = train_node_supervision(params, model, snap_train, aggr_val, rand_sampling, edge_index, edge_weight, target, learning_rate, dense_nodes, True, False, 0)
    # finetune on the deep layers, the learning rate should be much smaller
    for param in model.conv.parameters():
        param.requires_grad = False
    if dense_layers == 0:
        return model, train_error, constr_err, (train_evolution[0],train_evolution[1]) , (val_evolution[0],val_evolution[1]), constr_change  
    else:
        model2, train_error2,  constr_err_2, train_evolution2, val_evolution2, constr_change2 = train_node_supervision(params, model, snap_train, aggr_val, rand_sampling, edge_index, edge_weight, target, learning_rate*0.1, dense_nodes, False, False, dense_layers)
        return model2, train_error2, constr_err_2, (train_evolution[0]+train_evolution2[0],train_evolution[1]+train_evolution2[1]) , (val_evolution[0]+val_evolution2[0],val_evolution[1]+val_evolution2[1]) , (constr_change[0]+constr_change2[0],constr_change[1]+constr_change2[1]) 

def conv_regression_aggr(params, init_snap, aggr_train, aggr_val, rand_sampling, edge_index, edge_weight, aggr_target, learning_rate, dense_nodes, dense_layers):
    """ predict the next step by using a cheb graph convolution """
    model = Model_regression(dense_nodes).to(device)

    # pretrain on the conv layer
    model, train_error, constr_err, train_evolution, val_evolution, constr_change, phases_changes, best_epoch, epoch = train_aggr_supervision(params, model, init_snap, aggr_train, aggr_val, aggr_target, rand_sampling, edge_index, edge_weight, 0, learning_rate, dense_nodes, True, dense_layers, params["bootstrap_increment"])
    # finetune on the deep layers, the learning rate should be much smaller
    for param in model.conv.parameters():
        param.requires_grad = False
    phases_changes.append((best_epoch, "best epoch GNN"))
    phases_changes.append((epoch, "GNN phase end"))  # save phase change position
    if dense_layers == 0:
        return model, train_error, constr_err, (train_evolution, phases_changes), (val_evolution, phases_changes), (constr_change, phases_changes)
    else:   
        model2, train_error2, constr_err_2, train_evolution2, val_evolution2, constr_change2, phases_changes2, best_epoch,_= train_aggr_supervision(params, model, init_snap, aggr_train, aggr_val, aggr_target, rand_sampling, edge_index, edge_weight, epoch, learning_rate*0.5, dense_nodes, False, dense_layers, params["bootstrap_increment"])
        phases_changes.extend(phases_changes2)
        phases_changes.append((best_epoch, "best epoch MLP"))
        return model2, train_error2, constr_err_2, (train_evolution+train_evolution2,phases_changes) , (val_evolution+val_evolution2,phases_changes) , (constr_change+constr_change2,phases_changes)


def grid_search(input, params, load_model = None):
    """ Do a grid search un the input provided that is a tuple as returned from read_input using the hyper_parameters """
    hyp = params["hyper_parameters"]
    train_points, valid_points, test_points = params["train_points"], params["valid_points"], params["test_points"]
    # learning_rate, dense_nodes, dense_layers
    hyp = np.mgrid[hyp[0][0]:hyp[0][1]:hyp[0][2], hyp[1][0]:hyp[1][1]:hyp[1][2], hyp[2][0]:hyp[2][1]:hyp[2][2]]

    aggregated, snap, edge_index, edge_weight, target, aggr_target, max_degree = input
    snap_train, snap_val, aggr_train, aggr_val = snap[:train_points, :,:],snap[train_points:valid_points, :,:], aggregated[:train_points,:], aggregated[train_points:valid_points,:]
    aggregated = aggregated.cpu().numpy()
    rand_sampling_S, rand_sampling_P, rand_sampling_U = graph_utils.rand_init_conditions(True, 100, max_degree)
    rand_sampling = [(rand_sampling_S[:64,:,:], rand_sampling_P[:64,:,:], rand_sampling_U[:64,:,:]), (rand_sampling_S[:,:,:], rand_sampling_P[:,:,:], rand_sampling_U[:,:,:])]

    start = time.time() # time execution of training for a single seed
    best_train_err = M
    best_val_err = M
    best_test_err_nodes = M
    best_constr_loss = M
    best_idx = 0
    best_model = []
    best_pred = []
    best_pred0 = [] # best prediction viewed on node 0
    snap_np = np.transpose(snap.cpu().numpy(),(1,0,2))   # transpose to do tests and plots
    init_snap = (th.mean(snap[0,:,:],0)[None,:]).repeat(snap.size(1),1) # the init snap is averaged and spread between nodes

    if load_model is not None:
        (dir, seed) = load_model
        model, train_error, constr_loss, train_evolution, val_evolution, constr_evolution, hyp_matrix = Model_regression.read_trained_model(dir, seed, hyp)
    else:
        hyp_matrix = np.ndenumerate(hyp[0])

    # try every combination of hyper parameters and find the best
    it= 0
    for idx, _ in hyp_matrix:
        # train 
        if load_model is None:
            if params["nodes_supervision"]:
                model, train_error, constr_loss, train_evolution, val_evolution, constr_evolution = conv_regression_node(params, snap_train, aggr_val, rand_sampling, edge_index, edge_weight, target, hyp[0][idx], int(hyp[1][idx]),int(hyp[2][idx]))
            else:
                model, train_error, constr_loss, train_evolution, val_evolution, constr_evolution = conv_regression_aggr(params, init_snap, aggr_train, aggr_val, rand_sampling, edge_index, edge_weight, aggr_target, hyp[0][idx], int(hyp[1][idx]), int(hyp[2][idx]))

        # prediction 
        def get_model_output(input):
            return model(input, edge_index, edge_weight, None, False, False, int(hyp[2][idx]))
        pred = Model_regression.forward_recurrent(get_model_output, snap_train[0:1,:,:], test_points)
        pred = pred[:,0,:,:].detach().cpu().numpy()
        pred = np.transpose(pred,(1,0,2))

        # calculate test stats
        test_error_nodes, acc_pred, val_error, test_mse, test_mape, test_forecast, test_forecast_20, constr_err = graph_utils.test_stats(train_points, valid_points, test_points, snap_np, pred, aggregated)
        print(".", end="", flush=True)
        

        if (((val_error < best_val_err and not params["constrain"]) or (constr_err < best_constr_loss and params["constrain"])) and val_error != np.nan and val_error != np.inf) or it==0:
            best_train_err, best_val_err, best_constr_loss, best_test_mse, best_test_mape, best_test_forecast, best_test_forecast_20, best_idx, best_model, best_pred, best_pred0 = train_error, val_error, constr_err, test_mse, test_mape, test_forecast, test_forecast_20, idx, model, acc_pred, pred
            best_train_evolution, best_val_evolution, best_constr_evolution = train_evolution, val_evolution, constr_evolution 
        it += 1

    out_str = ""
    out_str += "\n({:5.4f},{:5.4f},{:5.4f}) ".format(hyp[0][best_idx],hyp[1][best_idx],hyp[2][best_idx])
    #out_str += "train err for seed {}: {:7.5f}, ".format(seed, best_train_err)
    out_str += "constr err: {:10.8f}, ".format(best_constr_loss)
    out_str += "val err: {:7.5f}, ".format(best_val_err)
    out_str += "test mse : {:7.5f} ".format(best_test_mse)
    out_str += "test mape : {:7.5f} ".format(best_test_mape)
    out_str += "test forecast : {:7.5f} ".format(best_test_forecast)
    out_str += "test forecast 20 : {:7.5f} ".format(best_test_forecast_20)
    out_str += "time: {:4.2f}s".format(time.time() - start)

    print(out_str)

    # plots
    if(params["plots"]):
        #graph_utils.plot_evolution(constr_evolution, 'constraints error', True)
        #graph_utils.plot_evolution(train_evolution, 'train error')
        #graph_utils.plot_evolution(val_evolution, 'validation error')
        # plot aggregation
        #plot_sir(best_pred[:,:], aggregated[:,:])
        label_method = "STGNN"
        if params["constrain"]:
            label_method += " with Constrains"
        else:
            label_method += " without Constrains"
        graph_utils.plot_sir(best_pred, aggregated[:test_points,:], label_method, 'Fraction of Population', params)
        for i in range(3): graph_utils.plot_sir(best_pred0[i*2,:test_points,:], snap_np[i*2,:test_points,:], f'{i*2} Node Dynamics '+label_method, 'Compartment Probability', params) # plot node's signal

    return best_model, best_constr_loss, best_train_err, best_test_mse, best_test_mape, best_test_forecast, best_test_forecast_20, best_idx, out_str, best_train_evolution, best_val_evolution, best_constr_evolution

def main():
    parser = argparse.ArgumentParser(prog='Graph Recurrent', description='Find the graph recurrent model from a range of input files and a parameters.json, and output found models and stats on file')
    parser.add_argument('--first_seed', type=int, default=0, help='first seed to test')
    parser.add_argument('--last_seed', type=int, default=10, help='last seed to test')
    parser.add_argument('--dataset', default='dataset1', help='name of the dataset used, should contain an "in" folder with the range of input files and one folder per test to be done')
    parser.add_argument('--test_name', default='recurrent_aggr_no_constr_100', help='folder containing the parameters.json file used to specify test parameters')
    args = parser.parse_args()
    #graph_utils.test_many_seeds(args.first_seed, args.last_seed, args.test_name, args.dataset, grid_search, read_input, Model_regression.save_model)
    graph_utils.many_seeds_stats(args.first_seed, args.last_seed, args.test_name, args.dataset, Model_regression.read_trained_model, grid_search, read_input, None)
    #graph_utils.read_model(0, "./dataset2/test/", "dataset2", grid_search, read_input)

if __name__ == "__main__":
    main()
