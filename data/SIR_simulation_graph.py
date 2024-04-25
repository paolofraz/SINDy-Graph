import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
import pickle

def barabasi_albert_graph(nodes, save_location):
    G = nx.barabasi_albert_graph(nodes,5, 0)
    #nx.draw(G)
    #plt.show()
    nx.write_edgelist(G, save_location)

def simulate(b, g, points, seed):
    # importing graphs
    G1 = nx.read_edgelist('./data/net_barabasi.txt')
    G1 = nx.convert_node_labels_to_integers(G1,label_attribute='original index')
    G1.name="Network1"

    # Model Selection
    model = ep.SIRModel(G1, seed = seed)

    # Model Configuration
    config = mc.Configuration()
    config.add_model_parameter('beta', b)
    config.add_model_parameter('gamma', g)
    config.add_model_parameter("fraction_infected", 0.1)
    model.set_initial_status(config)

    # Simulation
    iterations = model.iteration_bunch(points)
    trends = model.build_trends(iterations)

    # plot trends
    viz = DiffusionTrend(model, trends)
    #p = viz.plot()

    S = viz.trends[0]['trends']['node_count'][0]
    I = viz.trends[0]['trends']['node_count'][1]
    R = viz.trends[0]['trends']['node_count'][2]
    #plt.plot(np.linspace(0, 1, len(S)), S)
    #plt.plot(np.linspace(0, 1, len(S)), I)
    #plt.plot(np.linspace(0, 1, len(S)), R)
    #plt.show()

    return np.stack((S, I, R), axis=-1) / 1000

#simulate(0.01,0.02, 200, 0)

# return all the intermediate states (delta)
def simulate_all_iterations(b, g, points, seed):
    # importing graphs
    G1 = nx.read_edgelist('./data/net_barabasi.txt')
    G1 = nx.convert_node_labels_to_integers(G1,label_attribute='original index')
    G1.name="Network1"

    # Model Selection
    model = ep.SIRModel(G1, seed = seed)

    # Model Configuration
    config = mc.Configuration()
    config.add_model_parameter('beta', b)
    config.add_model_parameter('gamma', g)
    config.add_model_parameter("fraction_infected", 0.03)
    model.set_initial_status(config)

    # Simulation
    iterations = model.iteration_bunch(points)

    return iterations, G1

def gen_inputs(seeds, params):
    """ generate seeds number of simulation signals on graph and save them to file"""
    test_points = params["test_points"]
    for seed in range(seeds):
        # get a simulation of the pandemic spread with features from every node (the numeric class s,i,r)
        iterations, G = simulate_all_iterations(0.008,0.02, test_points, seed)
        # save graph and signal to file so that they are always the same 
        if seed == 0:
            nx.write_gpickle(G,'./dataset1/G.gpickle') # save only the first as they are all equal
        # save signal on graph
        with open(r'./dataset2/in/signal'+str(seed)+'.data', 'wb') as fp:
            pickle.dump(iterations, fp)

#barabasi_albert_graph(1000, "./data/net_barabasi.txt")
#simulate_all_iterations(0.03,0.01, 200, 0)
#gen_inputs(20, {"test_points":275})