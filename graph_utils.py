import numpy as np
import math
import json
import tkinter as tk
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})
import torch as th
import cooper
import torch.nn.functional as F
from torch import nn
import copy
import sindy_graph
from torch_geometric import utils

device = "cuda"
M = th.tensor(1000000000, dtype=th.float32, device=device)
# Import current time and date
from datetime import datetime
now = datetime.now()
# String of Date and time
dt_string = now.strftime("%d-%m-%Y_%H%M%S")
# Constraints

def rand_init_conditions(also_rand_sampling, samples_per_node, neigh_upper_bound):
    """rand_sampling S samples uniformly the feasible initial conditions for the sir model
       rand_sampling_P samples from the same set but putting to 0 a variable at a time for all samples
       if also_rand_sampling, a random sampling is added to the extremes sampling"""
    rand_sampling_S = th.rand(samples_per_node,1,6, dtype=th.float32, device=device) 
    rand_sampling_S[:,:,3:] *= 1 #neigh_upper_bound

    rand_sampling_P = th.rand(samples_per_node,3,6, dtype=th.float32, device=device) 
    rand_sampling_P[:,:,:3] /= th.sum(rand_sampling_P[:,:,:3],2)[:,:,None]   # make the sum 1
    rand_sampling_P[:,:,3:] /= th.sum(rand_sampling_P[:,:,3:],2)[:,:,None]   # make the sum 1
    rand_sampling_P[:,:,3:] *= 1 #neigh_upper_bound
    rand_sampling_P[:,0,0], rand_sampling_P[:,1,1], rand_sampling_P[:,2,2] = 0,0,0
    rand_sampling_U = th.rand(samples_per_node,3,6, dtype=th.float32, device=device) 
    rand_sampling_U[:,:,:3] /= th.sum(rand_sampling_U[:,:,:3],2)[:,:,None]   # make the sum 1
    rand_sampling_U[:,:,3:] /= th.sum(rand_sampling_U[:,:,3:],2)[:,:,None]   # make the sum 1
    rand_sampling_U[:,:,3:] *= 1 #neigh_upper_bound
    rand_sampling_U[:,0,0], rand_sampling_U[:,1,1], rand_sampling_U[:,2,2] = 1,1,1

    hyper_cube = [[0],[1]]
    for _ in range(5): hyper_cube = [[0]+e for e in hyper_cube] + [[1]+e for e in hyper_cube] 
    cube_sampling = th.tensor(hyper_cube, device=device, dtype=th.float32)
    cube_sampling = cube_sampling[:,None,:]

    cube_sampling_S = th.clone(cube_sampling) 
    cube_sampling_S[:,:,3:] *= 1# neigh_upper_bound/5

    cube_sampling_P = th.clone(cube_sampling).repeat(1,3,1) 
    cube_sampling_P[:,0,0], cube_sampling_P[:,1,1], cube_sampling_P[:,2,2] = 0,0,0
    # search for cube indexes with sum at 1 for node's and features' states
    #valid_ind = th.logical_and(th.sum(cube_sampling_P[:,:,:3], 2) == 1, th.sum(cube_sampling_P[:,:,3:], 2) == 1)
    valid_ind = th.sum(cube_sampling_P[:,:,:3], 2) == 1
    cube_sampling_P = th.stack([cube_sampling_P[:,0,:][valid_ind[:,0],:], cube_sampling_P[:,1,:][valid_ind[:,1],:], cube_sampling_P[:,2,:][valid_ind[:,2],:]],1)
    cube_sampling_P[:,:,3:] *= 1# neigh_upper_bound/5

    cube_sampling_U = th.clone(cube_sampling).repeat(1,3,1)  
    cube_sampling_U[:,0,0], cube_sampling_U[:,1,1], cube_sampling_U[:,2,2] = 1,1,1
    # search for cube indexes with sum at 1 for node's and features' states
    #valid_ind = th.logical_and(th.sum(cube_sampling_U[:,:,:3], 2) == 1, th.sum(cube_sampling_U[:,:,3:], 2) == 1)
    valid_ind = th.sum(cube_sampling_U[:,:,:3], 2) == 1
    cube_sampling_U = th.stack([cube_sampling_U[:,0,:][valid_ind[:,0],:], cube_sampling_U[:,1,:][valid_ind[:,1],:], cube_sampling_U[:,2,:][valid_ind[:,2],:]],1)
    cube_sampling_U[:,:,3:] *= 1# neigh_upper_bound/5

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
    bound = 1 if upper else 0
    sign = 1 if upper else -1 # the sign of the loss depends on the inequality type
    #measures_noise = th.rand(measures_rand_B.size(), dtype=th.float32, device=device)*0.04 -0.02

    #losses = []
    #for v,var in enumerate(["S","I","R"]): 
    #    measures_near_extremes = th.clone(measures_rand_B)
    #    # zero out (or bring to 1) all terms that contain the variable
    #    for term_with_var in [i for i,x in enumerate(get_model_constraints.dict_names) if var in x and var+'_' not in x]:
    #        measures_near_extremes[:,:,term_with_var] = bound
    #    rand_pred = get_model_constraints(measures_near_extremes) 
    #    losses.append(sign*(rand_pred)[:,:,v] + measures_noise[:,:,v])

    # add noise to measures to cover more phase space of initial conditions
    #measures_noise = th.rand(measures_S.size(), dtype=th.float32, device="cuda")*0.04 -0.02
    #measures_S_1 = measures_S + measures_noise
    #measures_I_1 = measures_I + measures_noise
    #measures_R_1 = measures_R + measures_noise

    defect = get_model_constraints(measures_rand_B)
    ineq_defect = th.stack([sign*defect[:,0,0],sign*defect[:,1,1],sign*defect[:,2,2]],1)
    #upp_loss = th.cat(losses,1) 
    if separate: ineq_defect = ineq_defect / (ineq_defect.size(0))# * ineq_defect.size(1)) 
    else: ineq_defect = th.mean(F.relu(ineq_defect))

    return ineq_defect

def constr_loss(get_model_constraints, measures_rand):
    """Get aggregated constr loss of sum and bounds"""
    sum_loss = constr_sum_loss(get_model_constraints, measures_rand[0], separate=False)
    positivity_loss = constr_bound_loss(get_model_constraints, measures_rand[1], separate=False, upper=False)
    upper_loss = constr_bound_loss(get_model_constraints, measures_rand[2], separate=False, upper=True)
    return (sum_loss + positivity_loss*1 + upper_loss*1)/3

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
            return cooper.CMPState(loss=loss, ineq_defect=self.ineq_defect*1, eq_defect=self.sum_defect*1) 
        else: # use given measures to calculate the defects
            defect = self.get_model_constraints(measures)
            ineq_defect = th.stack([-defect[:,0,0],-defect[:,1,1],-defect[:,2,2], defect[:,3,0],defect[:,4,1],defect[:,5,2]])#,-th.sum(defect[:,6,:],1), th.sum(defect[:,7,:],1)])
            eq_defect = th.stack([th.sum(defect[:,6,:],1), th.sum(defect[:,7,:],1)])
            return cooper.CMPState(loss=loss, ineq_defect=ineq_defect/self.measures.size(0), eq_defect=eq_defect/self.measures.size(0)) 

    # do a step of the constrained optimizer
    def opt_step(self, get_loss, measures):
        # optimize the main objective making the most violating initial condition not violate
        self.constr_optimizer.zero_grad()
        lagrangian = self.formulation.composite_objective(self.closure, get_loss, measures)
        self.formulation.custom_backward(lagrangian)
        self.constr_optimizer.step(self.closure, get_loss, measures)

class Init_cond(nn.Module):
    """ To find the most violating initial conditions the Init_cond model is optimized, respecting initial conditions feasibility"""
    def __init__(self, measure_aggr_fun, n_measures, neigh_upper_bound):
        super(Init_cond, self).__init__()
        self.measure_aggr_fun = measure_aggr_fun
        self.neigh_upper_bound = neigh_upper_bound
        self.multi_restart = 64
        # parameters for the least satisfying initial conditions (multi_restart, 3(variables s,i,r)*2(pos and upper)+2(for constant sum, as is expressed as 2 inequalities instead of 1 equality)=8, 
        #  node + neigh features = 6)
        self.init_cond = th.rand(self.multi_restart,8,6, dtype=th.float32, device=device)
        self.init_cond[:,:,3:]*= neigh_upper_bound/4  # initial conditions are initialized on the feasible set
        self.init_cond = nn.Parameter(self.init_cond)
        self.init_cond_0 = th.ones(self.init_cond.size(), dtype=th.float32, device=device)   # for forcing a variable above 0
        self.init_cond_1 = th.zeros(self.init_cond.size(), dtype=th.float32, device=device)   # for forcing a variable below 1
        self.init_cond_0[:,0,0], self.init_cond_0[:,1,1], self.init_cond_0[:,2,2] = 0,0,0
        self.init_cond_0[:,3,0], self.init_cond_0[:,4,1], self.init_cond_0[:,5,2] = 0,0,0
        self.init_cond_1[:,3,0], self.init_cond_1[:,4,1], self.init_cond_1[:,5,2] = 1,1,1
        self.measures = th.zeros((self.multi_restart,8,n_measures), dtype=th.float32, device=device) # the optimized measures

    # return the current conditions violations
    def forward(self, model):
        # force respectively S,I,R to 0 and measure with respect the the current init conditions (for only 1 node)
        init_cond = self.init_cond * self.init_cond_0 + self.init_cond_1
        #measures_pos = self.measure_aggr_fun(init_cond[:,:3,3:], None, init_cond[:,:3,:3])
        #measures_upp = self.measure_aggr_fun(init_cond[:,3:6,3:], None, init_cond[:,3:6,:3])
        #measures_sum_0 = self.measure_aggr_fun(init_cond[:,6:7,3:], None, init_cond[:,6:7,:3]) # sum >=0
        #measures_sum_1 = self.measure_aggr_fun(init_cond[:,7:8,3:], None, init_cond[:,7:8,:3]) # sum <=0
        #measures = th.cat([measures_pos, measures_upp, measures_sum_0, measures_sum_1],1)
        measures = self.measure_aggr_fun(init_cond[:,:,3:], None, init_cond[:,:,:3])
        # save found measures that are then used for constraining the other objective
        self.measures = measures.detach()
        # calculate how positive is the derivative with respect to the current init_conditions with
        # the goal is to get the lowest (most violating) value
        defect = model(measures)
        self.violation = th.stack([defect[:,0,0], defect[:,1,1], defect[:,2,2], - defect[:,3,0], - defect[:,4,1], - defect[:,5,2], + th.sum(defect[:,6,:],1), - th.sum(defect[:,7,:],1)]) 
        return th.mean(defect[:,0,0] + defect[:,1,1] + defect[:,2,2] - defect[:,3,0] - defect[:,4,1] - defect[:,5,2] + th.sum(defect[:,6,:],1) - th.sum(defect[:,7,:],1))

    def constrain(self):
        # I constrain the initial conditions to have correct values by clamping
        self.init_cond.clamp_(0,self.neigh_upper_bound)
        self.init_cond[:,:,:3] = th.clamp(self.init_cond[:,:,:3],0,1)

class init_conditions_opt():
    def __init__(self, model, message_passer, measures_rand_S, measures_rand_P, measures_rand_U, learning_rate):
        self.model = model
        max_degree = th.max(utils.degree(message_passer.G.edge_index[0,:]))
        self.init_cond = Init_cond(message_passer.measure_aggr, model.n_features, max_degree)
        self.optimizer = th.optim.Adam(self.init_cond.parameters(), lr=learning_rate*0)
        # get an upper bound for the neigh features

    # do a step of the constrained optimizer
    def opt_step(self):
        # find the most violating initial condition
        for param in self.init_cond.parameters(): param.requires_grad = True
        for param in self.model.parameters(): param.requires_grad = False

        self.loss = self.init_cond(self.model)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        for param in self.init_cond.parameters(): param.requires_grad = False
        for param in self.model.parameters(): param.requires_grad = True

        self.init_cond.constrain()

        # return the evaluations of the most violating initial conditions
        return self.init_cond.measures
        
# Stats and plots
def test_stats(train_points, valid_points, test_points, snap, pred, aggregated):
    """ calculate test stats for a single prediction pred"""
    # test error calculated on single nodes and averaged
    test_nodes = snap[:,valid_points:test_points,:]
    test_error_nodes = np.mean((pred[:,valid_points:test_points,:] - test_nodes)**2)
    # accumulate nodes' features in order to have the usual sir
    acc_pred = np.mean(pred, 0)
    # calculate mse with respect to the validation set
    val_error = np.mean(((acc_pred[train_points:valid_points] - aggregated[train_points:valid_points]))**2)
    # calculate mse for the test set
    test_mse = np.mean((acc_pred[valid_points:test_points] - aggregated[valid_points:test_points])**2)
    test_actual = aggregated[valid_points:test_points]
    test_mape = np.mean(np.abs((acc_pred[valid_points:test_points] - test_actual)/test_actual))*100    # mape error
    #  timesteps after the validation set until the error is > 10% and > 20%
    test_forecast = np.abs((acc_pred[valid_points:test_points] - test_actual)/test_actual)*100
    test_forecast_10 = np.concatenate((np.argwhere(test_forecast >= 10), [[test_points-valid_points,0]]))[0,0]
    test_forecast_20 = np.concatenate((np.argwhere(test_forecast >= 20), [[test_points-valid_points,0]]))[0,0]
    # constr error on test set
    test_pred = pred[:,valid_points:test_points,:]
    sum_err = np.mean(np.abs(np.sum(test_pred,2)-1))
    pos_err = -np.mean(test_pred[test_pred<0]) 
    pos_err = 0 if np.isnan(pos_err)  else pos_err
    constr_err = (sum_err + pos_err)/2

    return test_error_nodes, acc_pred, val_error, test_mse, test_mape, test_forecast_10, test_forecast_20, constr_err

def plot_sir(predictions_accum, target_accum, label_method, label_y, params):
    """Plot the real graph signal together with the predicted one"""
    ax = plt.figure().add_subplot(111)
    ax.plot(np.array(predictions_accum)[:,0], 'r', label='Predicted S')
    ax.plot(np.array(predictions_accum)[:,1], 'g', label='Predicted I')
    ax.plot(np.array(predictions_accum)[:,2], 'b', label='Predicted R')
    ax.plot(np.array(target_accum)[:,0], 'r--', label='Simulated S')
    ax.plot(np.array(target_accum)[:,1], 'g--', label='Simulated I')
    ax.plot(np.array(target_accum)[:,2], 'b--', label='Simulated R')
    # Annotate the Train set, the points from 0 to train_points, with a <-> arrow and train set label
    ax.annotate('', xy=(0.0, -0.03), xytext=(100, -0.03), arrowprops=dict(arrowstyle="<->", color='black', shrinkA=0, shrinkB=0))
    ax.text(params["train_points"]-params["train_points"]/2, -0.1, 'Train\nSet', ha='center', va='center', color='black', fontsize='x-small', bbox=dict(boxstyle="round", fc="white", ec="white", pad=0.1))

    ax.annotate('', xy=(100, -0.03), xytext=(125, -0.03), arrowprops=dict(arrowstyle="<->", color='black', shrinkA=0, shrinkB=0))
    ax.text(112.5, -0.1, 'Valid.\nSet', ha='center', va='center', color='black', fontsize='x-small', bbox=dict(boxstyle="round", fc="white", ec="white", pad=0.1))

    ax.annotate('', xy=(125, -0.03), xytext=(225, -0.03), arrowprops=dict(arrowstyle="<->", color='black', shrinkA=0, shrinkB=0))
    ax.text(175, -0.1, 'Test\nSet', ha='center', va='center', color='black', fontsize='x-small', bbox=dict(boxstyle="round", fc="white", ec="white", pad=0.1))
    ax.axvline(x = params["train_points"], color = 'black', lw=0.5, ls='-.', zorder=0)
    ax.axvline(x = params["valid_points"], color = 'black', lw=0.5, ls='-.', zorder=0)
    ax.axhline(y = 0.0, color = 'black', linewidth=.3, zorder=0)
    ax.axhline(y = 1.0, color = 'black', linewidth=.3, zorder=0)
    ax.set_xlabel('Time Step $t$')
    ax.set_ylabel(label_y)
    # Set the y ticks and labels only at [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    #ax.legend()#
    # Set the lower limit of the y axis to -0.2 and leave the upper limit unchanged
    lower, upper = ax.get_ylim()
    ax.set_ylim([-0.16, upper])
    # Set title
    ax.set_title(label_method)
    # Set the legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    #ax.set_ylim([-0.2, 1.1])
    plt.savefig(f'./plots/{label_method}_{dt_string}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_evolution(evolution, label, constr_target=False):
    """Plot the train/evaluation error with respect to the epochs along with the best epoch chosen and subdivided into the different phases.
    evolution is a tuple containing the evolution of the validation error and the different phases position in the epochs count, is an output of train"""
    evolution, phases_changes = evolution
    evolution = np.clip(evolution, 0, 10)
    ax = plt.figure().add_subplot(111)
    ax.plot(evolution, linewidth=1)
    # draw vertical lines to delimit phases and the best epoch
    for i,change in enumerate(phases_changes):
        if 'best epoch' in change[1]:
            ax.axvline(x = int(change[0]), label=change[1], color='purple')
        elif change[1] == 'threshold phase':
            ax.axvline(x = int(change[0]), label=change[1], color='orange')
        elif change[1] != 'approx supervision 101 steps': ax.axvline(x = int(change[0]), label=change[1], color=(i/len(phases_changes),1,0))
    if constr_target: ax.axhline(y = 5e-4, color = 'black', linewidth=.5, label='constraint target')
    ax.set_xlabel('epochs')
    ax.set_ylabel(label)
    ax.set_yscale('log')
    ax.legend()
    plt.show()

def test_many_seeds(start_seed, last_seed, out_dir, dataset, grid_search_fun, read_input_fun, save_model_fun):
    """ do a grid search for each seed using parameters got from file and then save stats"""
    # gen all inputs and write them to file
    #gen_inputs(seeds)

    parameters = {}
    with open("./%s/%s/parameters.json"%(dataset,out_dir), "r") as f:
        s = f.read()
        parameters = json.loads(s)

    # hyper params definition
    for seed in range(start_seed, last_seed+1):
        #aggregated, snap, d_snap_t, edge_index, edge_attr, mean_weight = read_input_chicken()
        best_model, best_constr_loss, best_train_err, best_test_mse, best_test_mape, best_test_forecast, best_test_forecast_20, best_idx, out_str, train_evolution, val_evolution, constr_evolution= grid_search_fun(read_input_fun(seed, dataset), parameters)
        stats = {}
        stats["best_constr_loss"] = float(best_constr_loss)
        stats["best_train_err"] = float(best_train_err)
        stats["best_test_mse"] = float(best_test_mse)
        stats["best_test_mape"] = float(best_test_mape)
        stats["best_test_forecast"] = float(best_test_forecast)
        stats["best_test_forecast_20"] = float(best_test_forecast_20)
        stats["best_idx"] = list(best_idx)
        stats["train_evolution"] = train_evolution[0]
        stats["validation_evolution"] = val_evolution[0]
        stats["constr_evolution"] = constr_evolution[0]
        stats["phases"] = [e[0] for e in train_evolution[1]]
        stats["phases_names"] = [e[1] for e in train_evolution[1]]
        # save best model for seed 
        save_model_fun("./%s/%s/%i"%(dataset,out_dir,seed), best_model)
        with open("./%s/%s/%i_out_str.data"%(dataset,out_dir,seed), "w") as f: f.write(out_str)
        with open("./%s/%s/%i_stats.json"%(dataset,out_dir,seed), "w") as f: f.write(json.dumps(stats))
    
def many_seeds_stats(start_seed, last_seed, out_dir, dataset, load_model_fun, grid_search_fun, read_input_fun, symbolic_frequency_fun=None):
    """ get specified seeds' saved info and compute mean stats"""
    mean_constr_err, mean_train_err, mean_test_mse, mean_test_mape, mean_test_forecast, mean_test_forecast_20= 0,0,0,0,0,0
    nan_vals = 0
    best_models = []
    for seed in range(start_seed, last_seed+1):
        #best_model = np.genfromtxt("./%s/%s/%i_model.data"%(dataset,out_dir,seed), dtype=np.float32)
        stats = {}
        with open("./%s/%s/%i_stats.json"%(dataset,out_dir,seed), "r") as f:
            s = f.read()
            stats = json.loads(s)

        if not math.isnan(stats["best_test_mse"]) and not math.isnan(stats["best_test_mape"]):
            #if not math.isnan(stats["best_constr_loss"]) and not math.isnan(stats["best_train_err"]) and not math.isnan(stats["best_test_mse"]) and not math.isnan(stats["best_test_mape"]) and not math.isnan(stats["best_test_forecast"]):
            #mean_constr_err += stats["best_constr_loss"]
            mean_train_err += stats["best_train_err"]
            mean_test_mse += stats["best_test_mse"]
            mean_test_mape += stats["best_test_mape"]
            mean_test_forecast += stats["best_test_forecast"]
            mean_test_forecast_20 += stats["best_test_forecast_20"]
            # recalculate constr_err
            _, best_constr_loss, _, _, _, _, _, _, _, _, _, _ = read_model(seed, "./"+dataset+"/"+out_dir+"/", dataset, grid_search_fun, read_input_fun)
            mean_constr_err += best_constr_loss
        else: nan_vals +=1

    stats = ""
    if symbolic_frequency_fun is not None: 
        for seed in range(start_seed, last_seed+1):
            best_model = load_model_fun("./%s/%s/%i"%(dataset,out_dir,seed))
            best_models.append(best_model)
        stats += symbolic_frequency_fun(best_models)

    seeds = last_seed - start_seed
    stats += "mean constr error is: {}\n".format(mean_constr_err/seeds)
    stats += "mean train error is: {}\n".format(mean_train_err/seeds)
    stats += "mean test mse is: {}\n".format(mean_test_mse/seeds )
    stats += "mean test mape is: {}\n".format(mean_test_mape/seeds)
    stats += "mean test forecast is: {}\n".format(mean_test_forecast/seeds)
    stats += "mean test forecast 20 is: {}\n".format(mean_test_forecast_20/seeds)
    stats += "nan values: {}\n".format(nan_vals)
    print(stats)

    with open("./%s/%s/stats.data"%(dataset,out_dir), "w") as f: f.write(stats)

def read_model(seed, dir, dataset, grid_search_fun, read_input_fun):
    """read model from file to enable doing tests and plots differently without retraining"""
    parameters = {}
    with open(dir + "parameters.json", "r") as f:
        s = f.read()
        parameters = json.loads(s)

    return grid_search_fun(read_input_fun(seed, dataset), parameters, (dir, seed))