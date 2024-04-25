# implementation of sindy on torch, adapted to evolve graph features
import numpy as np
from scipy import signal
import torch as th
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from data import SIR_simulation_graph
from scipy.signal import savgol_filter
import copy
import time
import networkx as nx
import pickle
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader, WikiMathsDatasetLoader
import cooper
import graph_utils
import sindy_graph_SEIR
import argparse
import json
import re
import torch.nn.functional as F

device = "cuda" if th.cuda.is_available() else "cpu"
#device = "cuda"
start = 0
dt = 1/50 # for the graph simulation
# points are overwritten by parameters.json
#train_points = int(100)
#valid_points = int(125)
#test_points = int(225)
#end = dt * test_points

filter_len = 1
M = th.tensor(1000000000, dtype=th.float32, device=device)


class Sindy(nn.Module):
    def __init__(self):
        super(Sindy, self).__init__()
        # features of the node and of neighbours crossed with themselves
        self.poly3_dict = [lambda X,d: X[:,:,0], lambda X,d: X[:,:,1],lambda X,d: X[:,:,2], 
                           lambda X,d: X[:,:,1] * X[:,:,0], lambda X,d: X[:,:,2] * X[:,:,0], lambda X,d: X[:,:,2] * X[:,:,1], lambda X,d: th.ones((X.size(dim=0), X.size(dim=1)), dtype=th.float32, device=d), 
                           ]
        #self.poly3_dict = [lambda X,d: X[:,:,0], lambda X,d: X[:,:,1],lambda X,d: X[:,:,2], 
        #                   lambda X,d: th.ones((X.size(dim=0), X.size(dim=1)), dtype=th.float32, device=d), ]
        # features of the node and of neighbours crossed together
        self.poly3_dict_cross = [
                           lambda X,X_n,d: X[:,:,0] * X_n[:,:,1], lambda X,X_n,d: X[:,:,1] * X_n[:,:,0], lambda X,X_n,d: X[:,:,2] * X_n[:,:,0],
                           lambda X,X_n,d: X[:,:,0] * X_n[:,:,2], lambda X,X_n,d: X[:,:,1] * X_n[:,:,2], lambda X,X_n,d: X[:,:,2] * X_n[:,:,1], ]
        
        
        self.dict_names = ['S', 'I', 'R', 'I*S', 'R*S', 'R*I', '',
                           'S_n', 'I_n', 'R_n', 'I_n*S_n', 'R_n*S_n', 'R_n*I_n', 
                           #'S_n2', 'I_n2', 'R_n2', 'I_n2*S_n2', 'R_n2*S_n2', 'R_n2*I_n2', 
                           'S*I_n', 'I*S_n', 'R*S_n', 'S*R_n', 'I*R_n', 'R*I_n',
                          ]

        self.n_features = len(self.dict_names) 

        # the coefficients are shared among all nodes
        self.coeff = nn.Parameter(th.rand(self.n_features,3, dtype=th.float32, device=device)/1000.0)
        #self.coeff = nn.Parameter(mp)
        # has a 0 if the corresponding coefficint has been thresholded and fixed to 0, else 1
        self.active_coeff = th.ones(self.n_features,3, dtype=th.float32, device=device)

    def forward(self, measures):
        return th.matmul(measures, th.mul(self.coeff, self.active_coeff))

    def save_model(path, model):
        """ save model coeffs and active coeffs as np arrays to be humanly readable"""
        np.savetxt(path + "_model.data", model.coeff.cpu().detach().numpy())
        np.savetxt(path + "_active.data", model.active_coeff.cpu().numpy())

    def read_model(path):
        model = Sindy()
        model.coeff = th.nn.Parameter(th.tensor(np.genfromtxt(path + "_model.data", dtype=np.float32), device=device))
        model.active_coeff = th.tensor(np.genfromtxt(path + "_active.data", dtype=np.float32), device=device)

        return model.to(device)
    
    def read_trained_model(folder_path, seed):
        """Read model and stats for a seed and return train info"""
        model = Sindy.read_model(folder_path +str(seed))
        with open(folder_path+str(seed)+"_stats.json", "r") as f:
            s = f.read()
            stats_json = json.loads(s)
        
        phases = np.stack([stats_json["phases"], stats_json["phases_names"]]).T
        train_evolution =  (stats_json["train_evolution"] , phases)
        val_evolution = (stats_json["validation_evolution"], phases)
        constr_evolution = (stats_json["constr_evolution"], phases)
        idx = tuple(stats_json["best_idx"])

        return model, stats_json["best_train_err"], stats_json["best_constr_loss"], train_evolution, val_evolution, constr_evolution, [(idx,0)]

    def clone_model(self):
        model_clone = copy.deepcopy(self.state_dict())
        model_clone.active_coeff = th.clone(self.active_coeff)
        return model_clone
    
    def load_clone(self,model):
        self.load_state_dict(copy.deepcopy(model))   
        self.active_coeff = th.clone(model.active_coeff)
    
    def toggle_neigh_features(self, active):
        if active:
            self.active_coeff[:,:] = 1
            # reset all neigh features to 0, as they may have been optimized with arbitrary values when 
            # not active
            dict = self.state_dict()
            #dict['coeff'][7:,:] = 0
            self.load_state_dict(dict)
        else: self.active_coeff[7:,:] = 0   
            #self.active_coeff[:,:] = 0
            #self.active_coeff[3,0] = 1
            #self.active_coeff[3,1] = 1
            #self.active_coeff[1,1] = 1
            #self.active_coeff[1,2] = 1

            #self.active_coeff[0,:] = 1
            #self.active_coeff[1,:] = 1
            #self.active_coeff[2,:] = 1

class Graph_message_passer(MessagePassing):
    """ used to represent the graph in torch geometric and use message passing to calculate measures and integrate the learned dynamics """
    # G of type torch_geometric.data.Data
    def __init__(self, edge_index, edge_attr, mean_weight, max_degree, sindy_model):
        # "Add" aggregation because I aggregate the sir features from neighbours
        super().__init__(aggr='add', flow="source_to_target")  
        # main graph that contains the adjacency
        self.G = Data(edge_index=edge_index[0], edge_attr=edge_attr[0], mean_weight=mean_weight[0])
        # graph where each node of the original graph is connected to its neighbours of the neighbours, to propagate info from them
        self.G2 = Data(edge_index=edge_index[1], edge_attr=edge_attr[1], mean_weight=mean_weight[1])
        self.max_degree = max_degree
        self.sindy_model = sindy_model

    def forward(self, snapshots, get_pred):
        # curr_snap has shape [N, time, node_features], I transpose 1st and 2nd dimensions to make S split work on message
        aggr_neigh_out = self.propagate(self.G.edge_index, S=th.transpose(snapshots, 0,1), W = self.G.edge_attr, meanW = self.G.mean_weight) 
        # normalize neighbours aggregation to avoid goig past 1 (this simplifies the constraining)
        aggr_neigh_out /= self.max_degree
        #aggr_neigh2_out = self.propagate(self.G2.edge_index, S=th.transpose(snapshots, 0,1), W = self.G2.edge_attr, meanW = self.G2.mean_weight) # aggregate neighbours of neighbours
        aggr_neigh2_out = aggr_neigh_out*0 # disable the calculation of neighbours of neighbours
        # measure the evolution using aggregated features
        measures = self.measure_aggr(th.transpose(aggr_neigh_out,0,1), th.transpose(aggr_neigh2_out,0,1),snapshots)
        #f = th.max(measures[:,:,:])
        # I only want measures
        if not get_pred: return measures
        else:   # integrate with the previous snapshot provided
            # multiply measures by learned coefficients
            dS_dt = self.sindy_model(measures)
            return dS_dt

    # aggregate the neighbouring features to get the features that the learned sindy model uses (S_,S_n,S_n2)
    def message(self, S_i, S_j, W, meanW):
        # S_i, S_j have shape [time, E, node_features], the first contains destination node's features, the second source neighbour's features
        ## calculate S_, features of the current node weighted by neighbouring arcs then accumulated
        #S_ = th.mul(S_i, self.G.edge_attr) / self.G.mean_weight
        # features of the neighbouring nodes accumulated and weighted by arcs len
        S_n = th.mul(S_j, W) / meanW
        #return th.concat((S_, S_n), dim=2)
        return  S_n

    # the update step is postponed to do it manually
    def update(self, aggr_out):
        return aggr_out

    # get aggregated features and use the model learned in sindy to get the derivatives for each node
    def measure_aggr(self, S_n, S_n2, S):
        f_len = len(self.sindy_model.poly3_dict)

        # measure the state of each node of all snapshots using dictionary's functions 
        measures = th.zeros((S.size(0), S.size(1), self.sindy_model.coeff.size(0)), dtype=th.float32, device=device)
        # node's features
        for i,f in enumerate(self.sindy_model.poly3_dict[:]):
            measures[:,:,i] = f(S, device) 
        # neighbours convolved polynomial features (excluding constant bias)
        for i,f in enumerate(self.sindy_model.poly3_dict[:-1]):
            measures[:,:,i + f_len ] = f(S_n, device)*1
        ## neighbours of neighbours convolved polynomial features (excluding constant bias)
        #for i,f in enumerate(self.sindy_model.poly3_dict[:-1]):
        #    measures[:,:,i + f_len*2 -1] = f(S_n2, device) 
        ## features of the node and neighbours multiplied together
        for i,f in enumerate(self.sindy_model.poly3_dict_cross[:]):
            measures[:,:,i + f_len*2 -1] = f(S, S_n, device) 

        return measures
    
    @th.no_grad()
    def measure(self, snapshots):
        """ Use message passing to calculate sindy's dictionary features """
        return self.forward(snapshots, False)
   
    #@th.no_grad()
    def integrate(self, model, initial_snapshots, steps):
        """ Use message passing to calculate sindy's dictionary features, multiply them by learned coefficients and predict the next snapshot.
        More than one initial snapshot can be provided, in that case multiple integrations are performed, each starting from a different snapshot.
        The input has shape [nodes, n_initial_snaps, 3], the output has shape [nodes, n_initial_snaps, steps, 3].
        By default the gradient is backpropagated only one step in the past """
        self.sindy_model = model
        #curr_snapshot = initial_snapshot[:,None,:]
        curr_snapshot = initial_snapshots    
        # store a steps long evolution for each initial snapshot
        d_preds_t = th.zeros((initial_snapshots.size(0), initial_snapshots.size(1),steps, 3), dtype=th.float32, device=device)
        preds = th.zeros((initial_snapshots.size(0), initial_snapshots.size(1),steps, 3), dtype=th.float32, device=device)
        for t in range(steps):
           curr_snapshot = curr_snapshot.detach()   # gradient backpropagation should start from here
           dS_dt = self.forward(curr_snapshot, True)
           d_preds_t[:,:,t,:] = dS_dt[:,:,:]
           # explicit euler step for integration
           curr_snapshot = curr_snapshot + dS_dt * dt
           preds[:,:,t,:] = curr_snapshot[:,:,:]
        
        return d_preds_t, preds

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

    nedges = G.number_of_edges()
    # get adjacency list [[adjacency_node0], [adj_node1], ..]
    adj_list = []
    for n, nbrdict in G.adjacency():
        adj_list.append(list(nbrdict.keys()))
    # get for each node the neighbours of the neighbours without repetitions nor the first neighbours nor the node
    adj2_list = []
    for n, adj in enumerate(adj_list):
        n_adj2 = []
        for neigh in adj:
            n_adj2.extend(adj_list[neigh])
        adj2_list.append(list(set(n_adj2).difference(set(adj)).difference(set([n]))))

    # get neighbours weights (in this case are all 1)
    adj_weigths = [[1.0 for e in adj] for adj in adj_list]
    adj2_weigths = [[1.0 for e in adj] for adj in adj2_list]
    # get mean neighbours weight for normalizing features
    mean_weight = [sum(n) for n in adj_weigths]
    mean_weight = sum(mean_weight)/len(mean_weight)
    # neighbours of neigh
    mean_weight2 = [sum(n) for n in adj2_weigths]
    mean_weight2 = sum(mean_weight2)/len(mean_weight2)

    # th geometric edges given by nodes indices (in both directions)
    edge_index_0 = [i for i,node in enumerate(adj_list) for j in node]
    edge_index_1 = [j for i,node in enumerate(adj_list) for j in node]
    edge_index = th.tensor([edge_index_0, edge_index_1], dtype=th.long, device=device)
    # edges to neighbours of neighbours
    edge_index_0 = [i for i,node in enumerate(adj2_list) for j in node]
    edge_index_1 = [j for i,node in enumerate(adj2_list) for j in node]
    edge_index2 = th.tensor([edge_index_0, edge_index_1], dtype=th.long, device=device)
    # th geometric arc weights with shape [num_edges, num_edge_features]
    edge_attr = [[j] for node in adj_weigths for j in node]
    edge_attr = th.tensor(edge_attr, dtype=th.float32, device=device)
    # neighbours of neighbours
    edge_attr2 = [[j] for node in adj2_weigths for j in node]
    edge_attr2 = th.tensor(edge_attr2, dtype=th.float32, device=device)

    # convert iterations delta values to dense snapshots with one hot encoding for sir features
    iterations_features = []
    current_states = np.zeros((1000, 3))
    for i,iteration in enumerate(iterations):
        for node, state in iteration['status'].items(): 
            current_states[node, :] = np.zeros(3)
            current_states[node, state] = 1
        iterations_features.append(current_states.copy())
    iterations_features = np.array(iterations_features)
    # get (nodes, timesteps, features) shape
    iterations_features = np.transpose(iterations_features, (1,0,2))
    
    # apply a filter to eliminate noise
    iterations_features = signal.convolve(iterations_features, np.ones(filter_len)[np.newaxis, :, np.newaxis]/filter_len, mode='full')[:,int(filter_len)-1:,:]
    #X = np.stack([savgol_filter(X_noise[:,0], filter_len, 3), savgol_filter(X_noise[:,1], filter_len, 3), savgol_filter(X_noise[:,2], filter_len, 3)], axis=1)
    #snapshots = th.tensor(iterations, dtype=th.float32)
    dX_dt = (iterations_features[:, 1:, :] - iterations_features[:, :-1, :]) / dt
    # aggregate features
    aggregated_features = np.mean(iterations_features, 0)
    # aggregate also the derivatives, in order to do aggregated supervision
    aggregated_derivatives = np.mean(dX_dt, 0)

    # create tensors
    iterations_features = th.tensor(iterations_features, dtype=th.float32, device=device)
    aggregated_features = th.tensor(aggregated_features, dtype=th.float32, device=device)
    dX_dt = th.tensor(dX_dt, dtype=th.float32, device=device)
    aggregated_derivatives = th.tensor(aggregated_derivatives, dtype=th.float32, device=device)

    return aggregated_features, iterations_features, dX_dt, aggregated_derivatives, (edge_index, edge_index2), (edge_attr, edge_attr2), (mean_weight, mean_weight2), adj_list, max_degree

def read_input_chicken():
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()

    iterations_features = np.apply_along_axis(lambda m: np.convolve(m, np.ones(32)/32, 'valid'), axis=0, arr=dataset.features)[20:,:,-1]
    iterations_features = th.tensor(iterations_features, device=device, dtype=th.float32)
    iterations_features = th.transpose(iterations_features, 0, 1)[:,:,None]
    iterations_features = F.pad(iterations_features, (0,2,0,0, 0,0), value=0) # pad to get 3 iterations_features
    dX_dt = (iterations_features[:, 1:, :] - iterations_features[:, :-1, :]) / dt

    edge_index = th.tensor(dataset.edge_index,device = device, dtype=th.long)
    edge_weight = th.tensor(dataset.edge_weight, device = device, dtype=th.float32)[:,None]

    mean_weight = edge_weight.size(0) / iterations_features.size(1)
    aggregated_features = th.sum(iterations_features, 0).cpu().numpy()

    return aggregated_features/20, iterations_features, dX_dt, (edge_index,edge_index), (edge_weight,edge_weight), (mean_weight,mean_weight)

def similar_nodes_stats(measures, pred):
    """ get all the nodes with the same initial conditions and plot their average signal """
    init = measures[:,0,:].cpu().numpy()
    similar_nodes = []
    for i in range(init.shape[0]):
        foo = init[i,0] == 0.0
        
        if init[i,0] == 1.0 and init[i,1] == 0.0 and init[i,2] == 0.0 and init[i,7] > 2.5/5 and init[i,7] < 3.0/5.0:# and init[i,8] > 0.5/5.0 and init[i,8] < 1.0/5.0 :
            similar_nodes.append(i)
        #if init[i,0] == 0.0 and init[i,1] == 1.0 and init[i,2] == 0.0 :
        #    similar_nodes.append(i)

    similar_signals = measures[similar_nodes,:,:3]
    signals_average = th.mean(similar_signals, 0).cpu().numpy()
    similar_pred = pred[similar_nodes,:,:3]
    pred_average = np.mean(similar_pred, 0)
    

    ax = plt.figure().add_subplot(111)
    ax.plot(np.array(pred_average)[:,0], 'r--', label='predicted S')
    ax.plot(np.array(pred_average)[:,1], 'g--', label='predicted I')
    ax.plot(np.array(pred_average)[:,2], 'b--', label='predicted R')
    ax.plot(np.array(signals_average)[:,0], 'r', label='average S')
    ax.plot(np.array(signals_average)[:,1], 'g', label='average I')
    ax.plot(np.array(signals_average)[:,2], 'b', label='average R')
    #ax.axvline(x = params["train_points"], color = 'grey', label='train_set')
    #ax.axvline(x = params["valid_points"], color = 'black', label='validation_set')
    ax.set_xlabel('t')
    ax.set_ylabel('compartment probability')
    ax.legend()
    #ax.set_ylim([-0.1, 1.1])
    plt.show()

def infection_speed_stats(params, adj_list, snap, pred):
    """ calculate infection speed related to the number of neighbours """
    test_points = params["test_points"]
    def calc (ax, adj_list, snap):
        n_neigh = [len(a) for a in adj_list]
        acc_infection_time = np.zeros(20)
        acc_infection_n = np.zeros(20)
        for i,n in enumerate(n_neigh):
            acc_infection_time[n] += 1-np.sum(snap[i,:,0],0)/ test_points
            acc_infection_n[n] += 1
        return acc_infection_time, acc_infection_n

    ax = plt.figure().add_subplot(111)
    acc_infection_time, acc_infection_n = calc(ax, adj_list, snap) 
    ax.plot(np.arange(0,20,1), acc_infection_time/acc_infection_n, label="true")
    acc_infection_time, acc_infection_n = calc(ax, adj_list, pred) 
    ax.plot(np.arange(0,20,1), acc_infection_time/acc_infection_n, label= "predicted")
    ax.legend()
    ax.set_xlabel('number of neighbours')
    ax.set_ylabel('average infection speed')
    plt.show()

def plot_constraints(measures_rand, model):
    """ plot how much the constraints are wrong """
    # plot how much a variable's derivative tends to be negative despite being close to 0, varying the other 2 variables
    extr = 0
    ax = plt.figure().add_subplot(422)
    for v,var in enumerate(["S","I","R"]): 
        measures_near_extremes = th.clone(measures_rand)
        # zero out (or bring to 1) all terms that contain the variable
        for term_with_var in [i for i,x in enumerate(model.dict_names) if var in x and var+'_' not in x]:
            measures_near_extremes[:,:,term_with_var] = extr
        # take the derivative that should be 0 (or 1) and check how much it is wrong
        der = th.clamp(-M*(1-extr), M*extr, model(measures_near_extremes)[:,:,v])
        der = th.abs(der).cpu().detach().numpy() * 100
        ax = plt.subplot(2,3,v+1)
        a = ax.scatter(measures_rand.cpu().detach().numpy()[:,:,1],measures_rand.cpu().detach().numpy()[:,:,2], s=5, c=der)
        plt.colorbar(a)


    # plot how much the sum is not constant varying the 3 variables
    rand_pred = model(measures_rand) * 100
    const_sum_deviation = th.abs((th.sum(rand_pred, 2)))
    a = plt.subplot(2,3,v+2).scatter(measures_rand.cpu().detach().numpy()[:,:,0],measures_rand.cpu().detach().numpy()[:,:,1], s=5, c=const_sum_deviation.cpu().detach().numpy())
    plt.colorbar(a)
    plt.show()


def loss_node_supervision(model, measures, criterion, d_snap_train_t):
    pred = model(measures)
    # calculate loss with mean square error 
    loss = criterion(pred, d_snap_train_t)
    return pred, loss

def loss_aggr_supervision(params, model, criterion, message_passer, d_aggr_train, forcing, steps):
    """ Predict using aggregated supervision in time from every provided forcing step for steps time steps. The loss with respect to the correct aggregated signal is returned.
    If integration_per_step is active, many initial states can be given, every one is integrated forward in time by steps steps,
    in this case aggr_train has shape [n_forced_timesteps, steps, 3], forcing [nodes, n_forced_timesteps, 3].
    Otherwise for every time step the input is provided by forcing, here aggr_train has shape [steps, 3], forcing [nodes, 3] """
    train_points, valid_points, test_points = params["train_points"], params["valid_points"], params["test_points"]
    steps = min(steps, train_points)

    # because I don't have teacher forcing I estimate the correct measures from the previous epoch's prediction
    d_preds_t, _ = message_passer.integrate(model, forcing, steps)

    #d_preds_t = (forcing_1[:,:,1:,:] - forcing_1[:,:,:-1,:])/dt  # calculate predicted derivatives
    preds_aggr = th.mean(d_preds_t,0)
    loss = criterion(preds_aggr, d_aggr_train[:preds_aggr.size(0),:,:])
    forcing = forcing.detach()  # backpropagate starting from this epoch
    return forcing, loss

def early_stopping(model, stop_metric, best_model, best_stop_metric, best_stop_metric_th, best_epoch, best_th, tolerance_epochs, countdown_epochs, countdown_thresholds, tolerance_thresholds, epoch, thresh):
    """ early stopping for the number of epochs and the number of thresholds based on the validation error """
    # apply early stopping after waiting some epochs for some improvement(comparison in lexicographic order)
    if (stop_metric[0] < best_stop_metric[0]) or (stop_metric[0] == best_stop_metric[0] and stop_metric[1] < best_stop_metric[1]): 
        best_stop_metric = stop_metric
        countdown_epochs = tolerance_epochs
    else: countdown_epochs -= 1

    # save model is it's the best found so far
    if (stop_metric[0] < best_stop_metric_th[0]) or (stop_metric[0] == best_stop_metric_th[0] and stop_metric[1] < best_stop_metric_th[1]): 
        best_model = model.clone_model()
        # reset thresholds countdown
        best_stop_metric_th = best_stop_metric
        best_epoch = epoch
        countdown_thresholds = tolerance_thresholds
        best_th = thresh
    
    return best_model, best_stop_metric, best_stop_metric_th, best_epoch, best_th, countdown_epochs, countdown_thresholds

def threshold_fun(model, measures, coeffs_increment, thresh, best_epoch, best_model, tolerance_epochs):
    """ Repeat thresholding until countdown_thresholds becomes 0, doing descent in between """
    model.load_clone(best_model)# take best model 
    print("|", end="", flush=True)
    #ax.axvline(x = int(epoch/10), color = 'grey')
    #tolerance_thresholds += 10
    countdown_epochs = tolerance_epochs
    #epoch = best_epoch

    # fix to 0 all coefficients below the coeffs_increment
    #m = nn.Threshold(coeffs_increment, 0.0)
    #thresholded_coeff = m(model.coeff) - m(-model.coeff)
    #model.active_coeff[thresholded_coeff == 0] = 0.0

    # terms to check are given multiplying measures' magnitudes with model's coefficients
    mean_dict_fun = th.mean(th.abs(measures), (0,1))  # find every measure's mean magnitude 
    terms_magnitude = th.abs(mean_dict_fun[:,None].repeat(1,model.coeff.size(1)) * model.coeff * model.active_coeff) 
    already_zero = th.numel(model.active_coeff) - th.sum(model.active_coeff)

    # put to 0 very small terms
    m = nn.Threshold(0.0001, 0.0)
    thresholded_coeff = m(terms_magnitude) - m(-terms_magnitude)
    model.active_coeff[thresholded_coeff == 0] = 0.0

    # fix to 0 the coefficients corresponding to summing terms in the derivatives with smallest magnitude (coeff*mean(dict_fun))
    # coefficients to 0 will grow linearly at each threshold to give time to recover from previous threshold
    #zero_coeffs = int(coeffs_increment * model.coeff.size(dim=0) * model.coeff.size(dim=1) * (thresh+1))
    zero_coeffs = int(coeffs_increment * th.numel(model.active_coeff) + already_zero)
    _, indices_sort_flattened = th.sort(th.flatten(terms_magnitude) , stable = True)    # then sorted to find smallest ones across all coeffs
    #sorted, indices_sort = th.sort(terms_magnitude, dim=0 , stable = True)    # then sorted to find smallest ones across all coeffs

    # search for couples of coefficients that are correlated and zero one of them
    #correlated = sorted[1:,:] < 1.02 * sorted[:-1,:]
    #flat_corr = F.pad(correlated, (0,0,0,1), value=0)
    #correlated_zero = [indices_sort[flat_corr[:,0],0], indices_sort[flat_corr[:,1],1], indices_sort[flat_corr[:,2],2]]

    indices_zero = indices_sort_flattened[:zero_coeffs]   # then the zero_coeffs smallest are picked
    active_flattened = th.flatten(model.active_coeff)  
    active_flattened[indices_zero] = 0.0    # zero small coefficients
    model.active_coeff = th.reshape(active_flattened, model.active_coeff.size()) # and put to 0
    #model.active_coeff[correlated_zero[0],0] = 0.0 # zero correlated coefficients
    #model.active_coeff[correlated_zero[1],1] = 0.0 # zero correlated coefficients
    #model.active_coeff[correlated_zero[2],2] = 0.0 # zero correlated coefficients


    # eliminate coeffs based on train and validation error
    # cycle through all non zero coefficients and try to zero one of them at a time, then validate to know which is best
    #for c in range(int(coeffs_increment * model.coeff.size(dim=0) * model.coeff.size(dim=1))):
    #    best_term = (0,0)
    #    best_loss = M
    #    old_active_coeff = copy.deepcopy(model.active_coeff)
    #    for c0 in range(model.coeff.size(0)):
    #        for c1 in range(model.coeff.size(1)):
    #            if model.active_coeff[c0,c1] > 0.5:
    #                model.active_coeff = copy.deepcopy(old_active_coeff)
    #                model.active_coeff[c0,c1] = 0
    #                pred_train = model(measures)
    #                pred_val = model(measures_val)
    #                loss_tv = criterion(pred_train, d_snap_train_t) 
    #                loss_tv += criterion(pred_val, d_snap_val_t) 
    #                if loss_tv < best_loss:   
    #                    best_loss = loss_tv
    #                    best_term = (c0,c1)
    #    # eliminate best coeff
    #    model.active_coeff = copy.deepcopy(old_active_coeff)
    #    model.active_coeff[best_term[0], best_term[1]] = 0

    return countdown_epochs

def train_cycle(params, model, message_passer, get_loss, criterion, learning_rate, dual_rate_multiplier, measures, measures_rand, init_snap, aggr_val, \
                epoch, best_model, best_stop_metric, best_stop_metric_th, best_epoch, best_th, countdown_epochs, tolerance_epochs, \
                limit_epochs, thresh, countdown_thresholds, train_evolution, val_evolution, constr_change, phases_changes, threshold):
    """Do a training cycle using the constraints or not until the early stopping criterion is met"""

    train_points, valid_points, test_points = params["train_points"], params["valid_points"], params["test_points"]
    constrain, coeffs_increment, tolerance_thresholds = params["constrain"], params["coeffs_increment"], params["tolerance_thresholds"]
    loss_constr, err_train, err_val = M, M, M
    pred = None
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
    if constrain: 
        if 'SEIR' in params and params['SEIR']==True: constr_opt = sindy_graph_SEIR.constrained_opt(model, model.parameters(), measures_rand, learning_rate, dual_rate_multiplier)
        else: constr_opt = graph_utils.constrained_opt(model, model.parameters(), measures_rand, learning_rate, dual_rate_multiplier)

    epochs_window = 0
    while countdown_epochs > 0 and epochs_window < limit_epochs: 
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
            _, pred = message_passer.integrate(model, init_snap[:,None,:], valid_points)
            err_val = criterion(th.mean(pred, (0,1))[train_points:,:], aggr_val)
            loss_constr = graph_utils.constr_loss(model, measures_rand)
        # to early stop the val error is used, if I constrain the loss constraint is used instead above a certain threshold, after err_val is used
        #early_stop_metric = err_val if not constrain else (loss_constr if epoch > int(tolerance_epochs/2) else M)
        early_stop_metric = (0,err_val) if not constrain else ((max(loss_constr,5e-4), err_val) if epoch > int(tolerance_epochs/2) else (M,M))
        best_model, best_stop_metric, best_stop_metric_th, best_epoch, best_th, countdown_epochs, countdown_thresholds = early_stopping(model, early_stop_metric, best_model, best_stop_metric, best_stop_metric_th, best_epoch, best_th, tolerance_epochs, countdown_epochs, countdown_thresholds, tolerance_thresholds, epoch, thresh)

        # record error evolution
        if epoch %1000==0: print("({:d}k, {:1.5f})".format(int(epoch/1000), float(err_val.cpu().detach().numpy())), end="", flush=True)
        train_evolution.append(float(err_train.cpu().detach().numpy()))
        val_evolution.append(float(err_val.cpu().detach().numpy()))
        constr_change.append(float(loss_constr.detach().cpu().numpy()))

        epochs_window += 1
        epoch += 1

    ## THRESHOLD ##
    # Repeat thresholding until countdown_thresholdsbecomes 0, doing descent in between
    if threshold:
        if countdown_thresholds > 0 and th.sum(model.active_coeff)>0:   # skip last threshold (I should conclude with a training phase)
            countdown_epochs = threshold_fun(model, measures, coeffs_increment, thresh, best_epoch, best_model, tolerance_epochs)
            best_stop_metric = (M,M)
            phases_changes.append((epoch, "threshold %i "%thresh))  # save threshold position
        thresh += 1
        countdown_thresholds -= 1

    return  best_model, best_stop_metric, best_stop_metric_th, best_epoch, epoch, best_th, countdown_epochs, thresh, countdown_thresholds, th.mean(pred,1)

def train_node_supervision(params, model, message_passer, snap, init_snap, d_snap_train_t, aggr_val, measures, measures_rand, learning_rate):
    """Train with supervision for each node for all training steps with thresholding, return trained model and train loss,validation error evolutions"""
    tolerance_thresholds, tolerance_epochs = params["tolerance_thresholds"], params["tolerance_epochs"]
    dual_rate = 0 if not params["constrain"] else params["dual_rate_multiplier"]

    best_stop_metric_th = (M,M) # best validation across thresholds
    best_stop_metric = (M,M) # best validation starting from a threshold
    countdown_epochs, countdown_thresholds = tolerance_epochs,tolerance_thresholds
    best_model = model.clone_model()
    best_epoch, best_th, loss_constr, thresh = 0,0,M,0
    epoch = 0 # count epochs effectively done 
    train_evolution, val_evolution, phases_changes, lambdas_change, constr_change = [],[],[],[],[]
    criterion = nn.MSELoss()

    # define how to get the loss for the constr optimizer
    def get_loss():
        _,loss = loss_node_supervision(model, measures, criterion, d_snap_train_t)
        return loss

    # train model with a smaller dictionary first
    if params["constrain"]:
        #model.toggle_neigh_features(False)
        best_model, best_stop_metric, best_stop_metric_th, best_epoch, epoch, best_th, countdown_epochs, thresh, countdown_thresholds, pred = \
            train_cycle(params, model, message_passer, get_loss, criterion, learning_rate, dual_rate, measures, measures_rand[0], init_snap, aggr_val, epoch, best_model, best_stop_metric, best_stop_metric_th, best_epoch, best_th, countdown_epochs, tolerance_epochs, \
            15000, thresh, countdown_thresholds, train_evolution, val_evolution, constr_change, phases_changes, threshold= False)

    countdown_epochs = tolerance_epochs     # reset the countdown to start a new phase with larger window
    phases_changes.append((epoch, "threshold phase"))  # save phase change position
    model.load_clone(best_model)# take best model 

    # do thresholding until I don't improve for tolerance_threshold times
    #model.toggle_neigh_features(True)
    while countdown_thresholds > 0: 
        best_model, best_stop_metric, best_stop_metric_th, best_epoch, epoch, best_th, countdown_epochs, thresh, countdown_thresholds, pred = \
            train_cycle(params, model, message_passer, get_loss, criterion, learning_rate, dual_rate, measures, measures_rand[0], init_snap, aggr_val, epoch, best_model, best_stop_metric, best_stop_metric_th, best_epoch, best_th, countdown_epochs, tolerance_epochs, \
            5000, thresh, countdown_thresholds, train_evolution, val_evolution, constr_change, phases_changes, threshold= True)


    phases_changes.append((best_epoch, "best epoch"))
    print(best_epoch, end=" \n", flush=True)
    model.load_clone(best_model)
    loss_constr = graph_utils.constr_loss(model, measures_rand[1])

    return model, train_evolution[best_epoch], loss_constr, (train_evolution, phases_changes), (val_evolution, phases_changes), (constr_change, phases_changes)

def train_aggr_supervision(params, model, message_passer, init_snap, aggr_train, d_aggr_train_t, aggr_val, n_nodes, measures, measures_rand, learning_rate):
    """ train without supervision for each node, but only aggregated information obtained from all nodes, return trained model and train loss,validation error evolutions \n
        works in 2 phases: \n
        (1) learning using appoximated teacher forcing every n steps (n gradually increasing) \n
        (2) fine tuning applying thresholding with no teacher forcing"""
    train_points, tolerance_thresholds, tolerance_epochs, bootstrap_increment = params["train_points"], params["tolerance_thresholds"], params["tolerance_epochs"], params["bootstrap_increment"]
    dual_rate = params["dual_rate_multiplier"]

    # keep best model for early stopping
    best_stop_metric_th = (M,M) # best metric across thresholds
    best_stop_metric = (M,M) # best metric starting from a threshold
    countdown_epochs, countdown_thresholds = tolerance_epochs,tolerance_thresholds
    best_model = model.clone_model()
    best_epoch, best_th, loss_constr, thresh = 0,0,M,0
    epoch = 0 # count epochs effectively done (excluding tolerance epochs done in vain)
    steps_no_forcing = 1 # start from predicting 1 step (complete teacher forcing)
    train_evolution, val_evolution, phases_changes, lambdas_change, constr_change = [],[],[],[],[]
    criterion = nn.MSELoss()

    # initialize the input used for approximated teacher forcing
    forcing = th.zeros((n_nodes, aggr_train.size(0), aggr_train.size(1)), dtype=th.float32, device=device)
    forcing[:,:,:] = aggr_train[None,:,:]    # give an approximate evolutions by spreading the pandemic equally between nodes

    #phase 1
    while steps_no_forcing <= train_points: 
        # the approximated teacher forcing is less and less present, to make it learn its nodes information but avoiding explosions. to do so a forcing step is done 
        # every steps_no_forcing, moreover the training set is shifted many times to augment the data
        #stride = int(math.ceil(steps_no_forcing/4)) # the stride increases as the model is more precise (i augment less and less the data)
        stride = int(train_points/20) if steps_no_forcing>1 else 1  # 50 windows independently of the steps no forcing
        shifts = int((train_points-steps_no_forcing)/stride)
        d_aggr_shifted = []   # repeat aggr_train shifting according to the stride
        for s in range(shifts): d_aggr_shifted.append(d_aggr_train_t[s*stride:s*stride+steps_no_forcing,:])
        d_aggr_shifted = th.stack(d_aggr_shifted, 0)
        def get_loss():
            _,loss = loss_aggr_supervision(params, model, criterion, message_passer, d_aggr_shifted, forcing[:,:shifts*stride:stride,:], steps_no_forcing)
            return loss
        
        # train using a smaller dictionary with complete approx forcing, then full, without thresholding
        phases_changes.append((epoch, "approx supervision %i steps"%steps_no_forcing))  # save phase change position
        best_model, best_stop_metric, best_stop_metric_th, best_epoch, epoch, best_th, countdown_epochs, thresh, countdown_thresholds, pred = \
            train_cycle(params, model, message_passer, get_loss, criterion, learning_rate, dual_rate, measures, measures_rand[0], init_snap, aggr_val, epoch, best_model, best_stop_metric, best_stop_metric_th, best_epoch, best_th, countdown_epochs, tolerance_epochs, \
        3000, thresh, countdown_thresholds, train_evolution, val_evolution, constr_change, phases_changes, threshold= False)

        #forcing = pred[:,:train_points,:]  # update the forcing using the last prediction to make it more precise
        steps_no_forcing += bootstrap_increment   # increment training set size 
        countdown_epochs = tolerance_epochs     # reset the countdown to start a new phase with larger window
        print("|", end="", flush=True)

    model.load_clone(best_model)# take best model 
    #best_stop_metric, best_stop_metric_th = M,M # the next best epoch should be after thresholding
    phases_changes.append((epoch, "threshold phase"))  # save phase change position

    # phase 2: here no forcing and no overlapping windows are used
    def get_loss():
        _,loss = loss_aggr_supervision(params, model, criterion, message_passer, d_aggr_train_t[None,:,:], forcing[:,0:1,:], train_points)
        return loss
    while countdown_thresholds > 0: 
        #def get_loss():
        #    _,loss = loss_aggr_supervision(params, model, criterion, message_passer, d_aggr_shifted, forcing[:,:shifts*stride:stride,:], steps_no_forcing-bootstrap_increment)
        #    return loss
        best_model, best_stop_metric, best_stop_metric_th, best_epoch, epoch, best_th, countdown_epochs, thresh, countdown_thresholds, pred = \
            train_cycle(params, model, message_passer, get_loss, criterion, learning_rate, dual_rate, measures, measures_rand[0], init_snap, aggr_val, epoch, best_model, best_stop_metric, best_stop_metric_th, best_epoch, best_th, countdown_epochs, tolerance_epochs*0.5, \
            1000, thresh, countdown_thresholds, train_evolution, val_evolution, constr_change, phases_changes, threshold= True)
        countdown_epochs = tolerance_epochs     
        #forcing = pred[:,:train_points,:]  # update the forcing using the last prediction to make it more precise

    phases_changes.append((best_epoch, "best epoch"))
    print(best_epoch, end=" \n", flush=True)
    model.load_clone(best_model)
    loss_constr = graph_utils.constr_loss(model, measures_rand[1])

    return model, train_evolution[best_epoch], loss_constr, (train_evolution, phases_changes), (val_evolution, phases_changes), (constr_change, phases_changes)

# get a string for each equation found symbolically
def symbolic_form(model):
    coeff = model.coeff.cpu().detach().numpy()
    dict_names = model.dict_names
    active_coeff = model.active_coeff.cpu().numpy()
    equations = ""
    for e in range(coeff.shape[1]): # s,i,r
        dedt = "dX{}/dt = ".format(e)
        for i, f in enumerate(dict_names): # features
            c = coeff[i,e]
            # only if the coefficient wasn't thresholded, show it
            if active_coeff[i,e]>0: dedt += "+ {} {} ".format(str(c), f)
        #for ii, f in enumerate(dict_names_crossed): # features crossed
        #    c = coeff[i+ii,e]
        #    if active_coeff[i+ii,e]>0: dedt += "+ {} {} ".format(str(c), f)
        equations += dedt + "\n"
    return equations

def symbolic_frequency(models):
    """ Take the models trained from different datasets instances, find the most recurring terms per diff. eq. and calculate their average and variance """
    dict_names = Sindy().dict_names
    terms = 8
    active = th.stack([c.active_coeff for c in models],2)
    coeffs = th.stack([th.mul(c.coeff ,c.active_coeff) for c in models],2)
    outStr = ""
    for e in range(3):  # for s,i,r
        # find the most used terms
        terms_freq = th.sum(active[:,e,:], 1)
        _, indices_sort = th.sort(terms_freq, stable = True, descending=True)    

        # take the best 5 coefficients
        mean = th.mean(coeffs[indices_sort[:terms], e, :], 1)
        variance = th.mean(th.abs(coeffs[indices_sort[:terms], e, :] - mean[:,None]),1)

        # write mean, variance and symbol for each diff eq.
        dedt = '$\\frac{d}{dt}$ & $'
        for i, f in enumerate(indices_sort[:terms]): # features
            mean_ = mean[i].detach().cpu().numpy()
            var_perc = variance[i].detach().cpu().numpy()# * 100 / np.abs(mean_)
            if var_perc > 0:
                dedt += "+{:6.2f}(\pm{:5.2f}) {}".format(mean_, var_perc, dict_names[f.detach().cpu().numpy()])
        outStr += dedt + "$\\\\ \\hline \n"
        print(dedt)
    return outStr

def plot_sir(predictions_accum, target_accum):
    """Plot the real graph signal together with the predicted one"""
    ax = plt.figure().add_subplot(111)
    ax.plot(np.array(predictions_accum)[:,0], 'r--', label='Sindy S')
    ax.plot(np.array(predictions_accum)[:,1], 'g--', label='Sindy I')
    ax.plot(np.array(predictions_accum)[:,2], 'b--', label='Sindy R')
    ax.plot(np.array(target_accum)[:,0], 'r', label='S')
    ax.plot(np.array(target_accum)[:,1], 'g', label='I')
    ax.plot(np.array(target_accum)[:,2], 'b', label='R')
    ax.axvline(x = 100, color = 'grey', label='train_set')
    ax.axvline(x = 125, color = 'black', label='validation_set')
    ax.set_xlabel('t')
    ax.set_ylabel('n of people')
    ax.legend()
    ax.set_ylim([-0.1, 1.1])
    plt.show()

def grid_search(input, params, load_model = None):
    """ Do a grid search un the input provided that is a tuple as returned from read_input using the hyper_parameters """
    hyp = params["hyper_parameters"]
    train_points, valid_points, test_points = params["train_points"], params["valid_points"], params["test_points"]
    #hyp[1,:,:] = 10**hyp[1,:,:] # exponentially spaced learning rate
    hyp = np.mgrid[hyp[0][0]:hyp[0][1]:hyp[0][2], hyp[1][0]:hyp[1][1]:hyp[1][2]]
    #hyp[1,:,:] = 10**hyp[1,:,:] # exponentially spaced learning rate
    # check if the dictionary corresponds with the one given as parameter
    if params["dictionary"] != Sindy().dict_names: raise Exception("The dictionary in parameters.json does not correspond with the one used by sindy")

    aggregated, snap, d_snap_t, d_aggr_t, edge_index, edge_attr, mean_weight, adj_list, max_degree = input
    #aggregated, snap, d_snap_t, d_aggr_t = aggregated/max_degree, snap/max_degree, d_snap_t/max_degree, d_aggr_t/max_degree  # normalize to eliminate bound constraint issues
    snap_train, snap_val, aggr_train, aggr_val = snap[:,:train_points,:],snap[:,train_points:valid_points,:], aggregated[:train_points,:], aggregated[train_points:valid_points,:]
    d_snap_train_t, d_snap_val_t, d_aggr_train_t, d_aggr_val_t = d_snap_t[:,:train_points,:], d_snap_t[:,train_points:valid_points,:], d_aggr_t[:train_points,:], d_aggr_t[train_points:valid_points,:]
    aggregated = aggregated.cpu().numpy()
    # measure input
    if 'SEIR' in params and params['SEIR']==True:
        message_passer = sindy_graph_SEIR.Graph_message_passer(edge_index, edge_attr, mean_weight, max_degree, sindy_graph_SEIR.Sindy_SEIR())
    else:
        message_passer = Graph_message_passer(edge_index, edge_attr, mean_weight, max_degree, Sindy())
    measures = message_passer.measure(snap_train)

    # generate random states to enforce the constraints
    if 'SEIR' in params and params['SEIR']==True:
        rand_sampling_S, rand_sampling_P, rand_sampling_U = sindy_graph_SEIR.rand_init_conditions(True, 100, max_degree)
        measures_rand_S = message_passer.measure_aggr(rand_sampling_S[:,:,4:], None, rand_sampling_S[:,:,:4])
        measures_rand_P = message_passer.measure_aggr(rand_sampling_P[:,:,4:], None, rand_sampling_P[:,:,:4])
        measures_rand_U = message_passer.measure_aggr(rand_sampling_U[:,:,4:], None, rand_sampling_U[:,:,:4])
    else:
        rand_sampling_S, rand_sampling_P, rand_sampling_U = graph_utils.rand_init_conditions(True, 100, max_degree)
        measures_rand_S = message_passer.measure_aggr(rand_sampling_S[:,:,3:], None, rand_sampling_S[:,:,:3])
        measures_rand_P = message_passer.measure_aggr(rand_sampling_P[:,:,3:], None, rand_sampling_P[:,:,:3])
        measures_rand_U = message_passer.measure_aggr(rand_sampling_U[:,:,3:], None, rand_sampling_U[:,:,:3])
    measures_rand = [(measures_rand_S[:64,:,:], measures_rand_P[:64,:,:], measures_rand_U[:64,:,:]),(measures_rand_S[:,:,:], measures_rand_P[:,:,:], measures_rand_U[:,:,:])]

    start = time.time() # time execution of training for a single seed
    best_train_err, best_val_err, best_test_err_nodes, best_constr_loss = M, M, M, M
    best_idx = (0,0)
    best_model = []
    best_pred = []
    best_pred0 = [] # best prediction viewed on node 0
    snap_np = snap.cpu().numpy()   # transpose to do tests and plots
    if params["nodes_supervision"]:
        init_snap = snap[:,0,:]
    else: init_snap = (th.mean(snap[:,0,:],0)[None,:]).repeat(snap.size(0),1) # the init snap is averaged and spread between nodes
    
    # either compute the models or load one from file
    if load_model is not None:
        (dir, seed) = load_model
        if 'SEIR' in params and params['SEIR']==True:
            model, train_error, constr_loss, train_evolution, val_evolution, constr_evolution, hyp_matrix = sindy_graph_SEIR.Sindy_SEIR.read_trained_model(dir, seed)
        else: model, train_error, constr_loss, train_evolution, val_evolution, constr_evolution, hyp_matrix = Sindy.read_trained_model(dir, seed)
    else:
        hyp_matrix = np.ndenumerate(hyp[0])

    # try every combination of hyper parameters and find the best
    #graph_utils.plot_sir(aggregated[:225,:], aggregated[:225,:], 'Sindy', 'people in a compartment', params)
    it= 0
    for idx, _ in hyp_matrix:
        # train and get symbolic equations
        if load_model is None:
            if 'SEIR' in params and params['SEIR']==True:
                model = sindy_graph_SEIR.Sindy_SEIR().to(device)
            else: model = Sindy().to(device)
            if params["nodes_supervision"]:
                model, train_error, constr_loss, train_evolution, val_evolution, constr_evolution = train_node_supervision(params, model, message_passer, snap, init_snap, d_snap_train_t, aggr_val, measures, measures_rand, hyp[1][idx])
            else: model, train_error, constr_loss, train_evolution, val_evolution, constr_evolution = train_aggr_supervision(params, model, message_passer, init_snap, aggr_train, d_aggr_train_t, aggr_val, snap.size(0), measures, measures_rand, hyp[1][idx])

        # prediction using integration on graph
        with th.no_grad(): _, pred = message_passer.integrate(model, init_snap[:,None,:], test_points)
        pred = th.squeeze(pred).cpu().detach().numpy()
        # calculate test stats
        test_error_nodes, acc_pred, val_error, test_mse, test_mape, test_forecast, test_forecast_20, constr_err = graph_utils.test_stats(train_points, valid_points, test_points, snap_np, pred, aggregated)
        # recalculate constr loss
        #constr_loss = graph_utils.constr_loss(model, measures_rand[1])

        # record best model and errors so far
        if (((val_error < best_val_err and not params["constrain"]) or (constr_err < best_constr_loss and params["constrain"])) and val_error != np.nan and val_error != np.inf) or it==0:
            best_train_err, best_val_err, best_constr_loss, best_test_mse, best_test_mape, best_test_forecast, best_test_forecast_20, best_idx, best_model, best_acc_pred, best_pred = train_error, val_error, constr_err, test_mse, test_mape, test_forecast, test_forecast_20, idx, model, acc_pred, pred 
            best_train_evolution, best_val_evolution, best_constr_evolution = train_evolution, val_evolution, constr_evolution 
        it += 1

    out_str = ""
    # symbolic form
    out_str += "\n"+symbolic_form(best_model)+"\n"
    #similar_nodes_stats(message_passer.measure(snap), best_pred0)
    #infection_speed_stats(params, adj_list, snap.cpu().numpy(), best_pred)
    out_str += "({:4.3f},{:4.3f})".format(hyp[0][best_idx], hyp[1][best_idx])
    out_str += " non zero coeff:{:3.0f}".format(np.sum(best_model.active_coeff.cpu().numpy()))+"\n\n"
    #out_str += "train err for seed {}: {:7.5f}, ".format(seed, best_train_err)
    out_str += "constr err: {:10.8f}, ".format(best_constr_loss)
    out_str += "val err: {:7.5f}, ".format(best_val_err)
    out_str += "test mse : {:7.5f} ".format(best_test_mse)
    out_str += "test mape : {:7.5f} ".format(best_test_mape)
    out_str += "test forecast : {:7.5f} ".format(best_test_forecast)
    out_str += "test forecast : {:7.5f} ".format(best_test_forecast_20)
    out_str += "time: {:4.2f}s".format(time.time() - start)

    print(out_str)

    # plots
    if(params["plots"]):
        if 'SEIR' in params and params['SEIR']==True: sindy_graph_SEIR.plot_seir(message_passer, model, init_snap, aggregated[:test_points,:], params)
        #similar_nodes_stats(measures, pred[:,:100,:])
        #graph_utils.plot_evolution(best_constr_evolution, 'constraints error', True)
        #graph_utils.plot_evolution(best_train_evolution, 'train error')
        #graph_utils.plot_evolution(best_val_evolution, 'validation error')
        #plot_sir(best_pred, aggregated)
        label_method = "SINDy-Graph"
        if params["constrain"]:
            label_method += " with Constrains"
        else:
            label_method += " without Constrains"
        graph_utils.plot_sir(best_acc_pred[:test_points,:], aggregated[:test_points,:], label_method, 'Fraction of Population', params)
        #for i in range(3): graph_utils.plot_sir(best_pred[i*7,:test_points,:], snap_np[i*7,:test_points,:], 'sindy node', 'compartment probability', params) # plot node's signal
        #plot_constraints(measures_rand, best_model)

    return best_model, best_constr_loss, best_train_err, best_test_mse, best_test_mape, best_test_forecast, best_test_forecast_20, best_idx, out_str, best_train_evolution, best_val_evolution, best_constr_evolution

def main():
    parser = argparse.ArgumentParser(prog='Sindy Graph', description='Find the sindy graph model from a range of input files and a parameters.json, and output found models and stats on file')
    parser.add_argument('--first_seed', type=int, default=0, help='first seed to test')
    parser.add_argument('--last_seed', type=int, default=19, help='last seed to test')
    parser.add_argument('--dataset', default='dataset1', help='name of the dataset used, should contain an "in" folder with the range of input files and one folder per test to be done')
    parser.add_argument('--test_name', default='sindy_SEIR_aggr_100', help='folder containing the parameters.json file used to specify test parameters')
    args = parser.parse_args()
    #graph_utils.test_many_seeds(args.first_seed, args.last_seed, args.test_name, args.dataset, grid_search, read_input, Sindy.save_model)
    graph_utils.many_seeds_stats(args.first_seed, args.last_seed, args.test_name, args.dataset, sindy_graph_SEIR.Sindy_SEIR.read_model, grid_search, read_input, symbolic_frequency)
    #graph_utils.read_model(0, "./dataset1/sindy_node_100/", "dataset1", grid_search, read_input)

if __name__ == "__main__":
    main()

