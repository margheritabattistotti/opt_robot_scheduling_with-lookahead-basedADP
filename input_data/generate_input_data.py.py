#%%
import numpy  as np
import pickle as pkl
import os
#%%
dir_path    = os.path.dirname(os.path.realpath(__file__))
graph0_info = pkl.load(open(dir_path+"//graph0_info.p","rb"))
tr_dict     = pkl.load(open(dir_path+"//time_and_risk_dict.p","rb"))
# %%

nodes_picking  = graph0_info["nodes_action"]["nodes_picking"]
nodes_throwing = graph0_info["nodes_action"]["nodes_throwing"]
nodes_coord    = graph0_info["nodes"]

nodes_list_p = ["np"+str(c) for c in range(len(nodes_picking))]
nodes_list_t = ["nt"+str(c) for c in range(len(nodes_throwing))]
nodes_list = nodes_list_p + nodes_list_t
nodes_transcod0,nodes_transcod1 = {},{}

for i in range(len(nodes_picking)):
    nodes_transcod0[nodes_picking[i]] = nodes_list_p[i]
    nodes_transcod1[nodes_list_p[i]]  = nodes_picking[i]
for i in range(len(nodes_throwing)):
    nodes_transcod0[nodes_throwing[i]] = nodes_list_t[i]
    nodes_transcod1[nodes_list_t[i]]  = nodes_throwing[i]

nodes_coordinates= {}
for n in nodes_list:
    nodes_coordinates[n] = nodes_coord[nodes_transcod1[n]]

time_risk_dict = {}
for key in tr_dict.keys():
    n0,n1 = key
    path0 = tr_dict[key]["path"]
    path1 = []
    for n in path0:
        path1 += [nodes_coord[n]]
    time_risk_dict[(nodes_transcod0[n0],nodes_transcod0[n1])] = {"time":tr_dict[key]["time"],"risk":tr_dict[key]["risk"],"path":path1}

# %%
target_trays = ["tray0","tray1"]
trays_coordinates = {"tray0":(147.5, 152.0),"tray1":(168.5, 152.0)}

# %%
objects = ["objectA","objectB","objectC","objectD","objectE"]
objects_pick_dict = {"objectA":"np0","objectB":"np1","objectC":"np2","objectD":"np3","objectE":"np4"}
# %%

input_dict = {"nodes":nodes_list,
              "nodes_coordinates":nodes_coordinates,
              "objects":objects,
              "objects_pick_nodes":objects_pick_dict,
              "trays":target_trays,
              "trays_coordinates":trays_coordinates,
              "nodes_connections":time_risk_dict}

pkl.dump(input_dict,open(dir_path+"\\input_dict.p","wb"))