import numpy as np
from scipy import signal
import torch as th
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import math
from data import SIR_simulation_graph
import copy
import networkx as nx
import graph_utils
import json
import torch.nn.functional as F
import cooper
import matplotlib.pyplot as plt

device = "cuda" if th.cuda.is_available() else "cpu"
dt = 1/50 # for the graph simulation

class Sindy_SEIR(nn.Module):
    def __init__(self):
        super(Sindy_SEIR, self).__init__()
        # features of the node and of neighbours crossed with themselves
        self.poly4_dict = [lambda X,d: X[:,:,0], lambda X,d: X[:,:,1],lambda X,d: X[:,:,2], lambda X,d: X[:,:,3], 
                           lambda X,d: X[:,:,0] * X[:,:,1], lambda X,d: X[:,:,0] * X[:,:,2], lambda X,d: X[:,:,0] * X[:,:,3], lambda X,d: X[:,:,1] * X[:,:,2], lambda X,d: X[:,:,1] * X[:,:,3], lambda X,d: X[:,:,2] * X[:,:,3], 
                           lambda X,d: th.ones((X.size(dim=0), X.size(dim=1)), dtype=th.float32, device=d), 
                           ]
        #self.poly4_dict = [lambda X,d: X[:,:,0], lambda X,d: X[:,:,1],lambda X,d: X[:,:,2], 
        #                   lambda X,d: th.ones((X.size(dim=0), X.size(dim=1)), dtype=th.float42, device=d), ]
        # features of the node and of neighbours crossed together
        self.poly4_dict_cross = [
                           lambda X,X_n,d: X[:,:,0] * X_n[:,:,1], lambda X,X_n,d: X[:,:,0] * X_n[:,:,2], lambda X,X_n,d: X[:,:,0] * X_n[:,:,3], lambda X,X_n,d: X[:,:,1] * X_n[:,:,2], lambda X,X_n,d: X[:,:,1] * X_n[:,:,3], lambda X,X_n,d: X[:,:,2] * X_n[:,:,3], 
                           lambda X,X_n,d: X[:,:,1] * X_n[:,:,0], lambda X,X_n,d: X[:,:,2] * X_n[:,:,0], lambda X,X_n,d: X[:,:,3] * X_n[:,:,0], lambda X,X_n,d: X[:,:,2] * X_n[:,:,1], lambda X,X_n,d: X[:,:,3] * X_n[:,:,1], lambda X,X_n,d: X[:,:,3] * X_n[:,:,2],] 
        
        
        self.dict_names = ['S', 'E', 'I', 'R', 'S*E', 'S*I', 'S*R', 'E*I', 'E*R', 'I*R', '',
                           'S_n', 'E_n', 'I_n', 'R_n', 'S_n*E_n', 'S_n*I_n', 'S_n*R_n', 'E_n*I_n', 'E_n*R_n', 'I_n*R_n',
                           'S*E_n', 'S*I_n', 'S*R_n', 'E*I_n', 'E*R_n', 'I*R_n',
                           'S_n*E', 'S_n*I', 'S_n*R', 'E_n*I', 'E_n*R', 'I_n*R',
                          ]

        self.n_features = len(self.dict_names) 

        # the coefficients are shared among all nodes
        self.coeff = nn.Parameter(th.rand(self.n_features,4, dtype=th.float32, device=device)/1000.0)
        #self.coeff = nn.Parameter(mp)
        # has a 0 if the corresponding coefficint has been thresholded and fixed to 0, else 1
        self.active_coeff = th.ones(self.n_features,4, dtype=th.float32, device=device)

    def forward(self, measures):
        # using the hidden variable E and I', the output I = E+I'

        return th.matmul(measures, th.mul(self.coeff, self.active_coeff))

    def save_model(path, model):
        """ save model coeffs and active coeffs as np arrays to be humanly readable"""
        np.savetxt(path + "_model.data", model.coeff.cpu().detach().numpy())
        np.savetxt(path + "_active.data", model.active_coeff.cpu().numpy())

    def read_model(path):
        model = Sindy_SEIR()
        model.coeff = th.nn.Parameter(th.tensor(np.genfromtxt(path + "_model.data", dtype=np.float32), device=device))
        model.active_coeff = th.tensor(np.genfromtxt(path + "_active.data", dtype=np.float32), device=device)

        return model.to(device)
    
    def read_trained_model(folder_path, seed):
        """Read model and stats for a seed and return train info"""
        model = Sindy_SEIR.read_model(folder_path +str(seed))
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
        # features of the neighbouring nodes accumulated and weighted by arcs len
        S_n = th.mul(S_j, W) / meanW
        #return th.concat((S_, S_n), dim=2)
        return  S_n

    # the update step is postponed to do it manually
    def update(self, aggr_out):
        return aggr_out

    # get aggregated features and use the model learned in sindy to get the derivatives for each node
    def measure_aggr(self, S_n, S_n2, S):
        f_len = len(self.sindy_model.poly4_dict)

        # measure the state of each node of all snapshots using dictionary's functions 
        measures = th.zeros((S.size(0), S.size(1), self.sindy_model.coeff.size(0)), dtype=th.float32, device=device)
        # node's features
        for i,f in enumerate(self.sindy_model.poly4_dict[:]):
            measures[:,:,i] = f(S, device) 
        # neighbours convolved polynomial features (excluding constant bias)
        for i,f in enumerate(self.sindy_model.poly4_dict[:-1]):
            measures[:,:,i + f_len ] = f(S_n, device)
        ## features of the node and neighbours multiplied together
        for i,f in enumerate(self.sindy_model.poly4_dict_cross[:]):
            measures[:,:,i + f_len*2 -1] = f(S, S_n, device) 

        return measures
    
    #def measure_aggr_split(self, S_n, S_n2, S):
    #    # split aggregated features before measuring
    #    S1 = th.zeros((S.size(0), S.size(1), 4), dtype=th.float32, device=device)
    #    S1[:,:,0], S1[:,:,1], S1[:,:,2], S1[:,:,3] = S[:,:,0], S[:,:,1], S[:,:,1], S[:,:,2]
    #    S_n1 = th.zeros((S_n.size(0), S_n.size(1), 4), dtype=th.float32, device=device)
    #    S_n1[:,:,0], S_n1[:,:,1], S_n1[:,:,2], S_n1[:,:,3] = S_n[:,:,0], S_n[:,:,1], S_n[:,:,1], S_n[:,:,2]

    #    return self.measure_aggr(S_n1, None, S1)
        
    @th.no_grad()
    def measure(self, snapshots):
        """ Use message passing to calculate sindy's dictionary features """
        # split I into 2 (I and E) to be able to calculate measures
        snaps_split = th.zeros((snapshots.size(0), snapshots.size(1), 4), dtype=th.float32, device=device)
        snaps_split[:,:,0], snaps_split[:,:,1], snaps_split[:,:,2], snaps_split[:,:,3] = snapshots[:,:,0]/2, snapshots[:,:,0]/2, snapshots[:,:,1], snapshots[:,:,2]
        return self.forward(snaps_split, False)
   
    #@th.no_grad()
    def integrate(self, model, initial_snapshots, steps, separate_seir=False):
        """ Use message passing to calculate sindy's dictionary features, multiply them by learned coefficients and predict the next snapshot.
        More than one initial snapshot can be provided, in that case multiple integrations are performed, each starting from a different snapshot.
        The input has shape [nodes, n_initial_snaps, 4], the output has shape [nodes, n_initial_snaps, steps, 4].
        By default the gradient is backpropagated only one step in the past """
        self.sindy_model = model
        curr_snapshot = th.zeros((initial_snapshots.size(0), initial_snapshots.size(1), 4), dtype=th.float32, device=device)
        # split I and E
        curr_snapshot[:,:,0], curr_snapshot[:,:,1], curr_snapshot[:,:,2], curr_snapshot[:,:,3] = initial_snapshots[:,:,0]/2, initial_snapshots[:,:,0]/2, initial_snapshots[:,:,1], initial_snapshots[:,:,2]
        # store a steps long evolution for each initial snapshot
        d_preds_t = th.zeros((initial_snapshots.size(0), initial_snapshots.size(1),steps, 4), dtype=th.float32, device=device)
        preds = th.zeros((initial_snapshots.size(0), initial_snapshots.size(1),steps, 4), dtype=th.float32, device=device)
        for t in range(steps):
           curr_snapshot = curr_snapshot.detach()   # gradient backpropagation should start from here
           dS_dt = self.forward(curr_snapshot, True)
           d_preds_t[:,:,t,:] = dS_dt[:,:,:]
           # explicit euler step for integration
           curr_snapshot = curr_snapshot + dS_dt * dt
           preds[:,:,t,:] = curr_snapshot[:,:,:]
        
        if not separate_seir:
            # fuse E and I features together
            d_preds_t[:,:,:,0] = d_preds_t[:,:,:,0] + d_preds_t[:,:,:,1] 
            preds[:,:,:,0] = preds[:,:,:,0] + preds[:,:,:,1] 
            return d_preds_t[:,:,:,[0,2,3]], preds[:,:,:,[0,2,3]]
        else: return d_preds_t[:,:,:,:], preds[:,:,:,:]



def rand_init_conditions(also_rand_sampling, samples_per_node, neigh_upper_bound):
    """rand_sampling S samples uniformly the feasible initial conditions for the sir model
       rand_sampling_P samples from the same set but putting to 0 a variable at a time for all samples
       if also_rand_sampling, a random sampling is added to the extremes sampling"""
    rand_sampling_S = th.rand(samples_per_node,1,8, dtype=th.float32, device=device) 

    rand_sampling_P = th.rand(samples_per_node,4,8, dtype=th.float32, device=device) 
    rand_sampling_P[:,:,:4] /= th.sum(rand_sampling_P[:,:,:4],2)[:,:,None]   # make the sum 1
    rand_sampling_P[:,:,4:] /= th.sum(rand_sampling_P[:,:,4:],2)[:,:,None]   # make the sum 1
    rand_sampling_P[:,0,0], rand_sampling_P[:,1,1], rand_sampling_P[:,2,2], rand_sampling_P[:,3,3] = 0,0,0,0
    rand_sampling_U = th.rand(samples_per_node,4,8, dtype=th.float32, device=device) 
    rand_sampling_U[:,:,:4] /= th.sum(rand_sampling_U[:,:,:4],2)[:,:,None]   # make the sum 1
    rand_sampling_U[:,:,4:] /= th.sum(rand_sampling_U[:,:,4:],2)[:,:,None]   # make the sum 1
    rand_sampling_U[:,0,0], rand_sampling_U[:,1,1], rand_sampling_U[:,2,2], rand_sampling_P[:,3,3]  = 1,1,1,1

    hyper_cube = [[0],[1]]
    for _ in range(7): hyper_cube = [[0]+e for e in hyper_cube] + [[1]+e for e in hyper_cube] 
    cube_sampling = th.tensor(hyper_cube, device=device, dtype=th.float32)
    cube_sampling = cube_sampling[:,None,:]

    cube_sampling_S = th.clone(cube_sampling) 

    cube_sampling_P = th.clone(cube_sampling).repeat(1,4,1) 
    cube_sampling_P[:,0,0], cube_sampling_P[:,1,1], cube_sampling_P[:,2,2], cube_sampling_P[:,3,3] = 0,0,0,0
    # search for cube indexes with sum at 1 for node's and features' states
    valid_ind = th.sum(cube_sampling_P[:,:,:4], 2) == 1
    cube_sampling_P = th.stack([cube_sampling_P[:,0,:][valid_ind[:,0],:], cube_sampling_P[:,1,:][valid_ind[:,1],:], cube_sampling_P[:,2,:][valid_ind[:,2],:], cube_sampling_P[:,3,:][valid_ind[:,3],:]],1)

    cube_sampling_U = th.clone(cube_sampling).repeat(1,4,1)  
    cube_sampling_U[:,0,0], cube_sampling_U[:,1,1], cube_sampling_U[:,2,2], cube_sampling_U[:,3,3]  = 1,1,1,1
    # search for cube indexes with sum at 1 for node's and features' states
    valid_ind = th.sum(cube_sampling_U[:,:,:4], 2) == 1
    cube_sampling_U = th.stack([cube_sampling_U[:,0,:][valid_ind[:,0],:], cube_sampling_U[:,1,:][valid_ind[:,1],:], cube_sampling_U[:,2,:][valid_ind[:,2],:], cube_sampling_U[:,3,:][valid_ind[:,3],:]],1)

    if also_rand_sampling:
        return th.cat([cube_sampling_S, rand_sampling_S],0), th.cat([cube_sampling_P, rand_sampling_P],0), th.cat([cube_sampling_U, rand_sampling_U],0)
    else: return cube_sampling_S, cube_sampling_P, cube_sampling_U

def constr_sum_loss(get_model_constraints, measures_rand_S, separate):
    """ get the deviation of having constant sum. If separate=false, the constraints are averaged together """

    # add to loss a term to avoid that the sum of variables changes
    rand_pred = get_model_constraints(measures_rand_S) #prediction on a random set of initial conditions
    const_sum_deviation = th.sum(rand_pred, 2) # dS + dI + dR = 0
    if separate:
        # divide by the number of elements because the optimizer then sums everything
        constr_loss = const_sum_deviation / (const_sum_deviation.size(0))#*const_sum_deviation.size(1))
    else: constr_loss = th.mean(th.abs(const_sum_deviation))

    return constr_loss

def constr_bound_loss(get_model_constraints, measures_rand_B, separate, upper):
    """ get the deviation of having negative derivative if a variable is 1. If separate=false, the constraints are averaged together
        upper = True if <=1 bound else >=0  """
    sign = 1 if upper else -1 # the sign of the loss depends on the inequality type
    defect = get_model_constraints(measures_rand_B)
    ineq_defect = th.stack([sign*defect[:,0,0],sign*defect[:,1,1],sign*defect[:,2,2],sign*defect[:,3,3]],1)
    #upp_loss = th.cat(losses,1) 
    if separate: ineq_defect = ineq_defect / (ineq_defect.size(0))# * ineq_defect.size(1)) 
    else: ineq_defect = th.mean(F.relu(ineq_defect))

    return ineq_defect

def constr_loss(get_model_constraints, measures_rand):
    """Get aggregated constr loss of sum and bounds"""
    sum_loss = constr_sum_loss(get_model_constraints, measures_rand[0], separate=False)
    positivity_loss = constr_bound_loss(get_model_constraints, measures_rand[1], separate=False, upper=False)
    upper_loss = constr_bound_loss(get_model_constraints, measures_rand[2], separate=False, upper=True)
    return (sum_loss + positivity_loss + upper_loss*1)/4

class constrained_opt(cooper.ConstrainedMinimizationProblem):
    def __init__(self, get_model_constraints, model_parameters, measures_rand, learning_rate, dual_rate_multiplier):
        super().__init__(is_constrained=True)
        self.get_model_constraints = get_model_constraints 
        self.measures_rand_S = measures_rand[0]
        self.measures_rand_P = measures_rand[1]
        self.measures_rand_U = measures_rand[2]
        self.formulation = cooper.LagrangianFormulation(self)
        self.learning_rate = learning_rate

        #primal_optimizer = th.optim.Adam(model_params, lr=learning_rate) 
        # for extrasgd the learning rate must be lower than adam
        primal_optimizer = cooper.optim.ExtraSGD(model_parameters, lr=learning_rate) 
        # Define the dual optimizer. Note that this optimizer has NOT been fully instantiated yet. Cooper takes care of this, once it has initialized the formulation state.
        #dual_optimizer = cooper.optim.partial_optimizer(th.optim.Adam, lr=learning_rate)
        dual_optimizer = cooper.optim.partial_optimizer(cooper.optim.ExtraSGD, lr=learning_rate*dual_rate_multiplier)
        self.constr_optimizer = cooper.ConstrainedOptimizer(self.formulation, primal_optimizer, dual_optimizer, dual_restarts=False, alternating=False)
        self.sum_defect, self.positivity_defect = 0,0    # save defects for later access

    # return a CMPstate to guide each step of the optimizer
    def closure(self, get_loss, measures):
        loss = get_loss()
        # inequality is given as constr_loss(x) <= 0
        # the lambdas are initialized small and grow later, otherwise the optimization cannot cross unfeasible reagions to reach the minima
        if measures == None:    # use precalculated measures
            self.sum_defect = constr_sum_loss(self.get_model_constraints, self.measures_rand_S, separate=True)
            self.ineq_defect_0 = constr_bound_loss(self.get_model_constraints, self.measures_rand_P, separate=True, upper=False)
            self.ineq_defect_1 = constr_bound_loss(self.get_model_constraints, self.measures_rand_U, separate=True, upper=True)
            self.ineq_defect = th.cat([self.ineq_defect_0, self.ineq_defect_1],0)
            return cooper.CMPState(loss=loss, ineq_defect=self.ineq_defect, eq_defect=self.sum_defect) 
        else: # use given measures to calculate the defects
            defect = self.get_model_constraints(measures)
            ineq_defect = th.stack([-defect[:,0,0],-defect[:,1,1],-defect[:,2,2],-defect[:,3,3], defect[:,3,0],defect[:,4,1],defect[:,5,2],defect[:,6,3]])
            eq_defect = th.stack([th.sum(defect[:,7,:],1), th.sum(defect[:,8,:],1)])
            return cooper.CMPState(loss=loss, ineq_defect=ineq_defect/self.measures.size(0), eq_defect=eq_defect/self.measures.size(0)) 

    # do a step of the constrained optimizer
    def opt_step(self, get_loss, measures):
        # optimize the main objective making the most violating initial condition not violate
        self.constr_optimizer.zero_grad()
        lagrangian = self.formulation.composite_objective(self.closure, get_loss, measures)
        self.formulation.custom_backward(lagrangian)
        self.constr_optimizer.step(self.closure, get_loss, measures)

def plot_seir(message_passer, model, init_snap, target_accum, params):
    """Plot the real graph signal together with the predicted one after a simulation on the s,e,i,r variables has been done"""
    with th.no_grad(): _, pred = message_passer.integrate(model, init_snap[:,None,:], params["test_points"], separate_seir=True)
    predictions_accum = th.mean(pred, (0,1))
    predictions_accum = predictions_accum.detach().cpu().numpy()

    ax = plt.figure().add_subplot(111)
    ax.plot(np.array(predictions_accum)[:,0], 'r--', label='modeled S')
    ax.plot(np.array(predictions_accum)[:,1], 'y--', label='modeled E')
    ax.plot(np.array(predictions_accum)[:,2], 'g--', label='modeled I')
    ax.plot(np.array(predictions_accum)[:,3], 'b--', label='modeled R')
    ax.plot(np.array(target_accum)[:,0], 'r', label='real S')
    ax.plot(np.array(target_accum)[:,1], 'g', label='real I')
    ax.plot(np.array(target_accum)[:,2], 'b', label='real R')
    ax.axvline(x = params["train_points"], color = '#ffcc00', label='train set')
    ax.axvline(x = params["valid_points"], color = '#ff9900', label='validation set')
    ax.axhline(y = 0.0, color = 'black', linewidth=.3)
    ax.axhline(y = 1.0, color = 'black', linewidth=.3)
    ax.set_xlabel('t')
    ax.set_ylabel('seir')
    ax.legend()
    #ax.set_ylim([-0.1, 1.1])
    plt.show()
