import pickle as pkl
import os
import math
import random
import timeit
import time

import numpy as np
from numpy import infty
import PoissonProcess as Poiss


# ############################################# INPUT DATA EXTRACTION ############################################# #

dir_path = os.path.dirname(os.path.realpath(__file__))
input_data = pkl.load(open(dir_path + "//input_data//input_dict.p", "rb"))

nodes              = input_data["nodes"][:4]            # list of nodes where actions can be performed
nodes.extend(input_data["nodes"][5:7])                  # 4 boxes, 2 throwing nodes
# nodes              = input_data["nodes"][:8]            # list of nodes where actions can be performed
nodes_coordinates  = input_data["nodes_coordinates"]    # dictionary n:c, where n-> node, c-> (x,y) coordinates
objects            = input_data["objects"][:4]              # list of objects that can be picked
O                  = len(objects)                       # total number of object types
objects_pick_nodes = input_data["objects_pick_nodes"]   # dictionary o:n, where o-> object, n-> picking node
trays              = input_data["trays"]                # list of target trays where objects can be place
# trays.append('tray2')                                   # for LARGE instance, need a third tray
T                  = len(trays)                         # total number of trays
trays_coordinates  = input_data["trays_coordinates"]    # dictionary t:c, where t-> tray, c-> (x,y) coordinates
# trays_coordinates['tray2'] = (131.75, 148.0)            # for LARGE instance, need a third tray
nodes_connections  = input_data["nodes_connections"]    # dictionary (n0,n1):(t,r,p), where (n0,n1)-> tuple of nodes,
# t-> time to reach n1 from n0, r-> navigation risk to reach n1 from n0, p-> path to reach n1 from n0


# ######################################## ADDITIONAL PARAMETERS DEFINITION ######################################## #

Thorizon = 300  # (s) horizon of the optimization
throwing_nodes = nodes[4:6]
# throwing_nodes = nodes[5:8]
tpick = 7  # (s) time elapsed to perform picking action
tthrow = 5  # (s) time elapsed to perform throwing action
maxPortableObjs = 4
rewards = {'move': 0, 'pick': 10, 'throw': 12}
fail_rewards = {'move': -2, 'pick': 10, 'throw': 0}
evaluation_rewards = {'move': 0, 'pick': 20, 'throw': 25}

rg_move_stream = random.Random(2404)
rg_throw_stream = random.Random(610)


# ##########################################  ORDERS LIST DEFINITION ########################################### #

all_orders = [{"objectA": 1, "objectB": 1, "objectC": 1, "objectD": 0},
              {"objectA": 3, "objectB": 0, "objectC": 0, "objectD": 1},
              {"objectA": 0, "objectB": 3, "objectC": 0, "objectD": 0},
              {"objectA": 2, "objectB": 0, "objectC": 1, "objectD": 1}]

# all_orders = [{"objectA": 1, "objectB": 2, "objectC": 1, "objectD": 0, "objectE": 0},
#               {"objectA": 3, "objectB": 0, "objectC": 0, "objectD": 1, "objectE": 2},
#               {"objectA": 0, "objectB": 3, "objectC": 0, "objectD": 0, "objectE": 2},
#               {"objectA": 0, "objectB": 0, "objectC": 3, "objectD": 3, "objectE": 0},
#               {"objectA": 2, "objectB": 0, "objectC": 0, "objectD": 2, "objectE": 1},
#               {"objectA": 2, "objectB": 0, "objectC": 1, "objectD": 1, "objectE": 1},
#               {"objectA": 0, "objectB": 1, "objectC": 2, "objectD": 1, "objectE": 2},
#               {"objectA": 4, "objectB": 0, "objectC": 1, "objectD": 0, "objectE": 0}]

orders_list = all_orders.copy()

priorities = [1, 2]
Twait = Thorizon/len(all_orders)  # (s) waiting time after which an order's priority status is upgraded


# ############################################# MISSION DEFINITION ############################################# #

# Also needed for state space computation
def objects4mission_dict(mission_dict):
    # INPUT
    # mission_dict: mission nested dictionary as defined in main
    # OUTPUT
    # tot_object: dictionary {objectType: total amount to be placed during mission}
    obj_keys_list = list(list(mission_dict.values())[0].keys())
    tot_object = {}
    for key in obj_keys_list:
        tot_object[key] = 0
    for l in range(len(mission_dict)):
        for key in obj_keys_list:
            tot_object[key] += list(mission_dict.values())[l][key]
    return tot_object


def set_mission(initial=1, tray_index=None, new_order=None, entering_time=None):
    # INPUT:
    # initial:
    # if = 2, it assigns fixed priorities and arrival times to the orders and defines a fixed initial mission
    # if = 1, it randomly assigns priorities and arrival times to the orders and defines an initial mission
    # else, it defines a new mission by substituting just one tray/order - it needs the tray_index and the new_order
    # OUTPUT (global):
    # mission
    # obj4trays_dict: output of tot_objects4mission()
    # tot_obj4mission: a list where each entry t specifies how many items must be placed in tray t during mission

    # Global initializations
    global mission
    global obj4trays_dict
    global obj4mission_dict
    global trays_entering_times
    global orders_entering_ranking
    global tray_has_order
    global all_arrival_times

    if initial == 2:
        # Fixed sequence - Always same mission - Needed to compare all methods
        mission = {}
        obj4trays_dict = {}
        trays_entering_times = []
        orders_entering_ranking = {}
        tray_has_order = {}
        # Assume number of orders to be always greater than the number of available trays
        for tray in trays:
            mission[tray] = orders_list[0]
            tray_has_order[tray] = orders_list[0]
            orders_entering_ranking[all_orders.index(tray_has_order[tray])] = '0s'
            obj4trays_dict[tray] = sum(list(mission[tray].values()))
            trays_entering_times.append(0)
            orders_list.remove(orders_list[0])

        obj4mission_dict = objects4mission_dict(mission)
        arrival_gap = Thorizon/(len(orders_list)+1)
        # arrival_gap = Thorizon / 6
        arrival_time = 0
        all_arrival_times = [arrival_time] * T
        random.seed(169)
        priority_list = [random.randint(1, 2) for _ in range(len(orders_list))]
        random.seed()
        for order in range(len(orders_list)):
            arrival_time += arrival_gap
            all_arrival_times.append(arrival_time)
            orders_list[order] = [orders_list[order], arrival_time, priority_list[order]]
            arrival_time += len(orders_list)/Thorizon

    elif initial == 1:
        # Create initial mission
        mission = {}
        obj4trays_dict = {}
        trays_entering_times = []
        orders_entering_ranking = {}
        tray_has_order = {}
        # Assume number of orders to be always greater than the number of available trays
        for tray in trays:
            mission[tray] = orders_list[0]
            tray_has_order[tray] = orders_list[0]
            orders_entering_ranking[all_orders.index(tray_has_order[tray])] = '0s'
            obj4trays_dict[tray] = sum(list(mission[tray].values()))
            trays_entering_times.append(0)
            orders_list.remove(orders_list[0])

        obj4mission_dict = objects4mission_dict(mission)
        # Appending of arrival times and priorities
        events = len(orders_list)
        lambda_ = events/Thorizon  # 1 / 150  # Expected value: one arrival every 2/300 seconds
        arrival_times = Poiss.PoissonProcess_simulation(lambda_, events)
        all_arrival_times = np.concatenate((np.zeros((T,)), arrival_times))

        # Modifying orders_list attaching arrival times and priority to each order
        # Result: array of dimension (len(orders_list), 3)
        for order in range(len(orders_list)):
            orders_list[order] = [orders_list[order], arrival_times[order], random.choice(priorities)]

    else:
        # Update mission and associated information
        mission[trays[tray_index]] = new_order[0]
        tray_has_order[trays[tray_index]] = new_order[0]
        obj4trays_dict[trays[tray_index]] = sum(list(mission[trays[tray_index]].values()))
        orders_list.remove(new_order)
        obj4mission_dict = objects4mission_dict(mission)
        trays_entering_times[tray_index] = entering_time
        orders_entering_ranking[all_orders.index(new_order[0])] = str(entering_time)+'s'


def UpdatePriorities(queue, current_t):
    # Every Twait time units, priority levels of writing orders increase
    updated_queue = queue
    temp_index = 0
    for order in queue:
        if current_t - order[1] >= 2 * Twait and order[2] > 2:
            updated_queue[temp_index][2] -= 2
        elif current_t - order[1] >= Twait and order[2] > 1:
            updated_queue[temp_index][2] -= 1
        temp_index += 1
    return updated_queue


def NextOrder(queue):
    # Entered whenever a tray is full check and there are orders waiting
    # Selects the next order based on its priority and arrival time
    if queue:
        highest_priority = queue[0][2]
        prioritized_orders = [order for order in queue if order[2] == highest_priority]
        # prioritized_orders.sort(key=lambda x: x[1])  # It is already ordered by arrival time
        next_order = [order for order in prioritized_orders if order[1] == prioritized_orders[0][1]][0]
        return next_order


def NextOrder2(queue, current_t):
    # Entered whenever a tray is full check and there are orders waiting
    # Selects the next order based on its priority and arrival time
    if queue:
        higher_priority = 0
        for order in queue:
            currentOrder_priority = 1 / order[2] * (current_t - order[1])
            if currentOrder_priority > higher_priority:
                higher_priority = currentOrder_priority
                next_order = order
        return next_order


# ################################### DEFINITION OF THROWING SUCCESS PROBABILITY ################################### #

def throwing_success(c0, c1):
    # Computes the probability of throwing success given the coordinates of the starting and the destination nodes
    # INPUT
    # c0: coordinates of node from where throwing action is performed
    # c1: coordinates of destination (must be tray)
    # OUTPUT
    # p: probability of success
    dst = math.sqrt((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2)
    if dst >= 80:
        p = 0
    else:
        p = -1 / 72 * (dst - 80)
    return p


# ############################################# STATE SPACE DEFINITION ############################################# #

def tot_objects_placed(s):
    # Given the state s of the system, returns a list where each entry j specifies how many objects of type j
    # have already been placed
    placed = [0] * O
    for j in range(len(placed)):
        for t in range(T):
            placed[j] += s[3 + (T+1) * j + t]
    return placed


def trays_level(s):
    # Given the state s of the system, returns a list where each entry t specifies how many objects
    # have already been placed in tray t
    placed = [0] * T
    for t in range(T):
        for j in range(O):
            placed[t] += s[3 + (T+1) * j + t]
    return placed


def objects_collected(s):
    # Yields the number of objects collected but not yet placed (inferred from the state entries)
    # INPUT
    # s: current state of the system
    # OUTPUT
    # obj: the number of objects collected but not yet placed in the given state
    objs = 0
    placed = tot_objects_placed(s)
    for j in range(O):
        objs += s[2 + (T+1) * j] - placed[j]  # picked - placed
    return objs


def state_space_brute_force(tot_obj_dict, maxObj):
    # Generates all possible final states in terms of picked and placed object types, without accounting for
    # final time and node position. Tailored for 5 object types and 2 trays.
    # INPUT
    # tot_obj_dict: {object: total amount to be placed during mission}, output of objects4mission_dict(mission)
    # maxObj: maximum number of objects to be carried at a time by the robot
    # OUTPUT
    # all_final_state: list of all possible final state configurations

    # 1) Definition of all possible states such that picked[i]>=placed[i] for all objects i
    all_states = [[a1, a2, a3, b1, b2, b3, c1, c2, c3, d1, d2, d3]
                  for a1 in range(list(tot_obj_dict.values())[0] + 1)
                  for a2 in range(0, list(list(mission.values())[0].values())[0] + 1)
                  for a3 in range(0, list(list(mission.values())[1].values())[0] + 1)
                  for b1 in range(list(tot_obj_dict.values())[1] + 1)
                  for b2 in range(0, list(list(mission.values())[0].values())[1] + 1)
                  for b3 in range(0, list(list(mission.values())[1].values())[1] + 1)
                  for c1 in range(list(tot_obj_dict.values())[2] + 1)
                  for c2 in range(0, list(list(mission.values())[0].values())[2] + 1)
                  for c3 in range(0, list(list(mission.values())[1].values())[2] + 1)
                  for d1 in range(list(tot_obj_dict.values())[3] + 1)
                  for d2 in range(0, list(list(mission.values())[0].values())[3] + 1)
                  for d3 in range(0, list(list(mission.values())[1].values())[3] + 1)]

    # # LARGE INSTANCE
    # all_states = [[a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, d3, d4, e1, e2, e3, e4]
    #               for a1 in range(list(tot_obj_dict.values())[0] + 1)
    #               for a2 in range(0, list(list(mission.values())[0].values())[0] + 1)
    #               for a3 in range(0, list(list(mission.values())[1].values())[0] + 1)
    #               for a4 in range(0, list(list(mission.values())[2].values())[0] + 1)
    #               for b1 in range(list(tot_obj_dict.values())[1] + 1)
    #               for b2 in range(0, list(list(mission.values())[0].values())[1] + 1)
    #               for b3 in range(0, list(list(mission.values())[1].values())[1] + 1)
    #               for b4 in range(0, list(list(mission.values())[2].values())[1] + 1)
    #               for c1 in range(list(tot_obj_dict.values())[2] + 1)
    #               for c2 in range(0, list(list(mission.values())[0].values())[2] + 1)
    #               for c3 in range(0, list(list(mission.values())[1].values())[2] + 1)
    #               for c4 in range(0, list(list(mission.values())[2].values())[2] + 1)
    #               for d1 in range(list(tot_obj_dict.values())[3] + 1)
    #               for d2 in range(0, list(list(mission.values())[0].values())[3] + 1)
    #               for d3 in range(0, list(list(mission.values())[1].values())[3] + 1)
    #               for d4 in range(0, list(list(mission.values())[2].values())[3] + 1)
    #               for e1 in range(list(tot_obj_dict.values())[4] + 1)
    #               for e2 in range(0, list(list(mission.values())[0].values())[4] + 1)
    #               for e3 in range(0, list(list(mission.values())[1].values())[4] + 1)
    #               for e4 in range(0, list(list(mission.values())[2].values())[4] + 1)]

    # 2) Filtering the states such that the robot does not carry more than maxObj objects
    all_states_b = []
    for s in all_states:
        carriedObj = objects_collected([0, 0] + s)
        if 0 <= carriedObj <= maxObj:
            all_states_b.append(s)

    # 3) Filtering the states such that picked >= tot_placed
    all_states_c = []
    for s in all_states_b:
        placed = tot_objects_placed([0, 0] + s)
        check = 0
        for j in range(O):
            if s[(T+1) * j] >= placed[j]:
                check += 1
        if check == O:
            all_states_c.append(s)

    return all_states_c


# ############################################# ACTION SPACE DEFINITION ############################################# #

# Moving is allowed from each node to all others. Info on time and risk in nodes_connections.
def moving_actions(s):
    # Returns all possible moving actions given the current state s
    # OUTPUT
    # mov_actions: list containing all possible moving actions, identified by the destination node
    move_actions = []
    all_destinations = nodes.copy()
    all_destinations.remove(s[1])
    for k in all_destinations:
        if s[0] + nodes_connections[(s[1], k)]['time'] <= Thorizon:
            move_actions.append(k)

    if len(move_actions) == 0:
        # No moving action allowed: time horizon reached
        move_actions = None

    return move_actions


# Picking is only allowed in picking nodes, and the robot can only pick the object in the respective box
def picking_actions(s):
    # Returns the possible picking action given the current state s
    # OUTPUT
    # pick_action: picking action identified by the type of object to pick (A=0, B=1, C=2, etc.)
    pick_action = None
    if s[0] + tpick <= Thorizon:
        for j in range(len(objects_pick_nodes)):
            if list(objects_pick_nodes.values())[j] == s[1] and s[2 + (T+1) * j] < list(obj4mission_dict.values())[j] \
                    and objects_collected(s) < maxPortableObjs:
                pick_action = j

    return pick_action


# Throwing is allowed only in placing nodes: 'nt0' and 'nt1', that are nodes[5] and nodes[6] in this instance.
def throwing_actions(s):
    # Returns all possible throwing actions given the current state s
    # OUTPUT
    # throw_actions: list containing all possible throwing actions, identified by the object type to place and the tray
    # where to throw it (A=0, B=1, etc. and trays as tray0=0, tray1=1. etc.)
    throw_actions = []
    if s[1] not in throwing_nodes:
        # Robot is not in a throwing location
        return None
    if s[0] + tthrow > Thorizon:
        return None
    for t in range(T):
        for j in range(O):
            placed = tot_objects_placed(s)[j]
            # picked > placed & mission not yet completed
            if s[2 + (T+1) * j] > placed and s[3 + (T+1) * j + t] < list(mission[trays[t]].values())[j]:
                throw_actions.append([j, t])

    if len(throw_actions) == 0:
        # No throwing action allowed: time horizon reached
        throw_actions = None

    return throw_actions


# ######################################## STATE TRANSITIONS DEFINITION ######################################## #

def new_state_moving(destination, s):
    # Returns the new state after moving action to destination, given current state s and exploiting information on
    # their connection {'time': , 'risk': , 'path': }
    new_s = s.copy()
    new_s[0] += nodes_connections[(s[1], destination)]['time']
    new_s[1] = destination
    return new_s


def new_state_picking(a, s):
    # Returns the new state after picking action a (= index of object type to pick : A=0, B=1, etc.),
    # given current state s
    new_s = s.copy()
    new_s[0] += tpick
    new_s[2 + (T+1) * a] += 1
    return new_s


def new_state_throwing(a, s):
    # Returns the new state after throwing action a (= [index of object type to place: A=0, B=1, etc., index of tray]),
    # given current state s
    new_s = s.copy()
    new_s[0] += tthrow
    new_s[3 + (T+1) * a[0] + a[1]] += 1
    return new_s


# ######################################## ADMISSIBLE ACTIONS DEFINITION ######################################## #

def transition_matrix_Q(s):
    # Returns the transition matrix Q of the system in the current state s as a nested dictionary
    # The main key is the type of action: 'move', 'pick', 'throw'
    # The secondary keys are the specific actions specifying where to move or what to pick and/or throw and where
    # To each secondary key it is specified the state that is reached if the respective action is performed
    # Example: Q = {'move': {'npo': s'}, 'pick': {2: s''}, 'throw': {[1, 0]: s'''}}

    Q_dict = {}

    if s[0] >= Thorizon:
        # Time horizon reached
        return None
    if tot_objects_placed(s) == list(obj4mission_dict.values()):
        # Mission accomplished
        if len(orders_list) != 0:
            return infty
        else:
            return None

    move = moving_actions(s)
    pick = picking_actions(s)
    throw = throwing_actions(s)

    if move is None and pick is None and throw is None:
        # No action can be performed: time horizon may be exceeded
        return None

    move_states = {}
    if move is not None:
        move = list(filter(lambda x: x is not None, move))
        for k in move:
            move_states[k] = new_state_moving(k, s)
        Q_dict['move'] = move_states

    pick_state = {}
    if pick is not None:
        pick_state[pick] = new_state_picking(pick, s)
        Q_dict['pick'] = pick_state

    throw_states = {}
    if throw is not None:
        throw = list(filter(lambda x: x is not None, throw))
        for k in throw:
            throw_states[tuple(k)] = new_state_throwing(k, s)
        Q_dict['throw'] = throw_states

    return Q_dict


# ####################################### TERMINAL VALUE FUNCTION DEFINITION ####################################### #

# Terminal value function I
def terminal_value(s):
    # Value function for a terminal state s
    V = Thorizon - s[0]
    placed = tot_objects_placed(s)
    for j in range(O):
        V -= list(obj4mission_dict.values())[j] - placed[j]
        V += s[2 + (T+1) * j]
    return V


# Evaluation metric when we have priorities
def terminal_state_evaluation(completion_rank):
    # Value function for a terminal state s in terms of mean time for the robot to fulfill an order
    # The lower the value the better
    Va = 0
    Ve = 0
    for order in completion_rank.keys():
        Va += float(completion_rank[order][:-1]) - all_arrival_times[order]
        Ve += float(completion_rank[order][:-1]) - float(orders_entering_ranking[order][:-1])
    Va *= 1 / len(completion_rank)
    Ve *= 1 / len(completion_rank)
    return Va, Ve


def completion_sequence_evaluation(completion_rank, sup=0):
    # Associates a value to the sorted sequence of completed orders by comparing it to the sorted
    # sequence of arrival times
    if sup == 1:
        # MAX SHIFT
        completion_list = list(completion_rank.keys())
        max_shift = 0
        for order in completion_list:
            idx_c = completion_list.index(order)
            if idx_c < T:
                idx_c = 0
            else:
                idx_c = idx_c - T + 1
            idx_a = list(orders_entering_ranking.keys()).index(order)
            if idx_a < T:
                idx_a = 0
            else:
                idx_a = idx_a - T + 1
            shift = abs(idx_c-idx_a)
            if shift > max_shift:
                max_shift = shift
        return max_shift

    else:
        # OVERALL SHIFT
        completion_list = list(completion_rank.keys())
        V = len(all_orders)
        for order in completion_list:
            idx_c = completion_list.index(order)
            if idx_c < T:
                idx_c = 0
            else:
                idx_c = idx_c - T + 1
            idx_a = list(orders_entering_ranking.keys()).index(order)
            if idx_a < T:
                idx_a = 0
            else:
                idx_a = idx_a - T + 1
            shift = idx_c - idx_a
            if shift <= 0:
                # only penalizing late completions
                shift = 0
            elif shift == 1:
                shift = 0.5
            else:
                shift -= 1
            V -= shift
        return V


# ################################################# BACKWARD PASS ################################################# #

def backward_pass(st_list, discount_factor=0.99):
    # INPUT
    # st_list: list of all states
    # OUTPUT
    # V: dictionary with a state as key and as value the respective value function
    V = {}  # Dictionary that maps each state to its value function
    for t in range(Thorizon, max(trays_entering_times) - 1, -1):
        for n in nodes:
            for z in st_list:
                s = [t, n] + z
                Q = transition_matrix_Q(s)
                if Q is None or Q is infty:
                    V[tuple(s)] = terminal_value(s)
                else:
                    V_temp = []
                    for action_type in Q.keys():
                        if action_type == 'pick':
                            immediate_contribution = rewards[action_type] * (Thorizon * 2 - s[0]) / Thorizon
                        else:
                            immediate_contribution = rewards[action_type]
                        for k in Q[action_type].keys():
                            v = immediate_contribution + discount_factor * V[tuple(Q[action_type][k])]
                            if action_type == 'throw':
                                if (s[0] - min(trays_entering_times) + 1) == 0:
                                    prova = 1
                                immediate_contribution = rewards[action_type] * (Thorizon * 2 - s[0]) / Thorizon \
                                              - trays_entering_times[k[1]] / (max(trays_entering_times) + 1) \
                                          + (s[0] - trays_entering_times[k[1]]) / (s[0] - min(trays_entering_times) + 1)
                                                    # rewards[action_type] / Thorizon * \
                                                    #  (Thorizon * 2 - s[0] - trays_entering_times[k[1]])  - \
                                                    #  3 * s[0]/Thorizon * ((max(trays_entering_times) + 1)
                                                    # / (trays_entering_times[k[1]] + max(trays_entering_times) + 1))

                                p_success = throwing_success(nodes_coordinates[s[1]], trays_coordinates[trays[k[1]]])
                                failed_s_t = s.copy()
                                failed_s_t[0] += tthrow
                                failed_s_t[2 + (T+1) * k[0]] -= 1  # Throwing failed: object lost
                                v = immediate_contribution + discount_factor * V[tuple(Q[action_type][k])]
                                v = v * p_success + (1 - p_success) * (discount_factor * V[tuple(failed_s_t)])
                            elif action_type == 'move':
                                # NB risk is a percentage
                                p_risk = nodes_connections[(s[1], Q[action_type][k][1])]['risk'] / 100
                                failed_s_m = Q[action_type][k].copy()
                                if failed_s_m[0] > Thorizon - 5:
                                    failed_s_m[0] = Thorizon
                                else:
                                    failed_s_m[0] += 5  # Reach next state but with 5 seconds penalty
                                v = v * (1 - p_risk) + p_risk * (-2 + discount_factor * V[tuple(failed_s_m)])
                            V_temp.append(v)

                    V[tuple(s)] = max(V_temp)
    return V


# ################################################# FORWARD PASS ################################################# #

def forward_pass(initial_state, values):
    # Defines the optimal strategy to complete the mission starting from 'initial_state',
    # given the value functions for each state in 'values'
    # OUTPUT
    # action_sequence: sequence of optimal actions
    # states_sequence: optimal state sequence
    # objective: terminal state evaluation
    # idle_time: time the robot spent waiting for more orders to arrive
    s = initial_state
    idle_time = 0
    action_sequence = []
    state_sequence = [s]
    completion_rank = {}
    Q = transition_matrix_Q(s)
    best_action = None
    best_new_state = None
    cnt = [0] * T
    while Q is not None:  # Time horizon not reached or mission not completed yet
        if Q == infty:
            # All trays are full. Just wait until the next order arrival
            action_sequence.append("Waiting for next order.")
            state_sequence.append("Been waiting for next order")
            next_arrival_time = math.ceil(orders_list[0][1])
            if next_arrival_time > Thorizon:
                # objective += terminal_state_evaluation(s, idle_time)
                objective_a, objective_e = terminal_state_evaluation(completion_rank)
                return action_sequence, state_sequence, objective_a, objective_e, idle_time, completion_rank
            idle_time = next_arrival_time - s[0]
            s[0] = next_arrival_time
        else:
            max_value = -100000
            for action_type in Q.keys():
                if action_type == 'pick':
                    immediate_contribution = rewards[action_type] * (Thorizon * 2 - s[0]) / Thorizon
                for k in Q[action_type].keys():
                    candidate_state = Q[action_type][k]
                    if action_type == 'move':
                        p_risk = nodes_connections[(s[1], Q[action_type][k][1])]['risk'] / 100
                        immediate_contribution = rewards[action_type] * (1 - p_risk) + p_risk * -2
                    if action_type == 'throw':
                        p_success = throwing_success(nodes_coordinates[s[1]], trays_coordinates[trays[k[1]]])
                        immediate_contribution = p_success * rewards[action_type] * (Thorizon * 2 - s[0]) / Thorizon \
                                              - trays_entering_times[k[1]] / (max(trays_entering_times) + 1) \
                                          + (s[0] - trays_entering_times[k[1]]) / (s[0] - min(trays_entering_times) + 1)

                    candidate_value = immediate_contribution + values[tuple(candidate_state)]
                    if candidate_value > max_value:
                        max_value = candidate_value
                        best_new_state = candidate_state
                        best_action = action_type + ' ' + str(k)
                        if action_type == 'throw' and \
                                rg_throw_stream.random() > p_success:
                            best_new_state = s.copy()
                            best_new_state[0] += tthrow
                            best_new_state[2 + (T+1) * k[0]] -= 1  # Throwing failed: object lost
                            best_action = best_action + ' ' + '(FAILED)'
                        elif action_type == 'move' and \
                                rg_move_stream.random() < p_risk:
                            if best_new_state[0] > Thorizon - 5:
                                best_new_state[0] = Thorizon
                            else:
                                best_new_state[0] += 5  # Reach next state but with 5 second penalty
                            best_action = best_action + ' ' + '(COLLISION)'
            action_sequence.append(best_action)
            state_sequence.append(best_new_state)
            s = best_new_state

        check_full_tray = trays_level(s)
        # If a tray is full and another order is in line, change the mission by emptying such tray and assigning to it
        # the new order
        for t in range(T):
            if check_full_tray[t] == obj4trays_dict[trays[t]]:
                if cnt[t] == 0:
                    action_sequence.append(trays[t] + ' ' + 'is full!')
                    completion_rank[all_orders.index(tray_has_order[trays[t]])] = str(s[0]) + 's'
                    cnt[t] = 1
                # if len(orders_list) != 0:
                orders_queue = [order for order in orders_list if order[1] < s[0]]
                orders_queue = UpdatePriorities(orders_queue, s[0])
                orders_queue.sort(key=lambda x: x[2])
                if len(orders_queue) != 0:
                    action_sequence.append("Let's change " + trays[t])
                    cnt[t] = 0
                    next_order = NextOrder(orders_queue)
                    # next_order = NextOrder2(orders_queue, s[0])
                    # Updating mission and associated information
                    set_mission(initial=0, tray_index=t, new_order=next_order, entering_time=s[0])
                    action_sequence.append('Now' + ' ' + trays[t] + ' ' + 'contains a new order'
                                           + ' ' + str(mission[trays[t]]))
                    updated_s = s.copy()
                    # Updating current state
                    for j in range(O):
                        # Removing placed objects in tray t from picked entry
                        updated_s[2 + (T + 1) * j] -= updated_s[3 + (T + 1) * j + t]
                        # Resetting to zero the number of placed objects in tray t
                        updated_s[3 + (T + 1) * j + t] = 0
                    state_sequence.append('Changing ' + trays[t])
                    state_sequence.append(updated_s)
                    state_sequence.append(trays[t] + ' changed')
                    s = updated_s

                    global states_list
                    # Compute a new state space and repeat backward pass for the new configuration
                    states_list = state_space_brute_force(obj4mission_dict, maxPortableObjs)
                    values = backward_pass(states_list, discount_factor=1)

        Q = transition_matrix_Q(s)

    check_full_tray = trays_level(s)
    for t in range(T):
        if check_full_tray[t] == obj4trays_dict[trays[t]] and \
                completion_rank.get(all_orders.index(tray_has_order[trays[t]])) is None:
            completion_rank[all_orders.index(tray_has_order[trays[t]])] = str(s[0]) + 's'

    # objective += terminal_state_evaluation(s, idle_time)
    objective_a, objective_e = terminal_state_evaluation(completion_rank)

    return action_sequence, state_sequence, objective_a, objective_e, idle_time, completion_rank


# ################################################# MAIN EXECTUTION ################################################# #

start = timeit.default_timer()
time.sleep(1)

set_mission(initial=2)

states_list = state_space_brute_force(obj4mission_dict, maxPortableObjs)
valueF = backward_pass(states_list, discount_factor=1)

objective_values_a = []
objective_values_e = []
sequence_values_max = []
sequence_values_overall = []

for iters in range(15):

    init = [0] * (O * (T + 1) + 2)
    # init[1] = random.choice(nodes)
    init[1] = 'np0'

    actions, states, objective_value_a, objective_value_e, idle_period, orders_completion_ranking \
        = forward_pass(init, valueF)

    objective_values_a.append(objective_value_a)
    objective_values_e.append(objective_value_e)
    sequence_value = completion_sequence_evaluation(orders_completion_ranking, sup=1)
    sequence_values_max.append(sequence_value)
    sequence_value2 = completion_sequence_evaluation(orders_completion_ranking, sup=0)
    sequence_values_overall.append(sequence_value2)

    print(actions)
    print(states)
    print('Robot was idle for ' + str(idle_period) + ' seconds.')
    print('Terminal state evaluation (wrt Arrival): ' + str(objective_value_a))
    print('Terminal state evaluation (wrt Entrance): ' + str(objective_value_e))
    print('Orders entering rank: ' + str(orders_entering_ranking))
    print('Orders completion rank: ' + str(orders_completion_ranking))
    print('Completion sequence evaluation (maximum shift):' + str(sequence_value))
    print('Completion sequence evaluation (overall shift):' + str(sequence_value2))

    # Resetting initial mission and orders for the next iteration
    orders_list = all_orders.copy()
    set_mission(initial=2)
    # Must compute state space again and repeat backpropagation because a new mission starts
    states_list = state_space_brute_force(obj4mission_dict, maxPortableObjs)
    valueF = backward_pass(states_list, discount_factor=1)

end = timeit.default_timer()
print(f"Mean terminal state value (wrt Arrival) is {sum(objective_values_a)/len(objective_values_a)}")
print(f"Mean terminal state value (wrt Entrance) is {sum(objective_values_e)/len(objective_values_e)}")
print(f"Time taken is {(end - start)/15}s")
print('Mean completion sequence evaluation (maximum shift):' + str(sum(sequence_values_max)/len(sequence_values_max)))
print('Mean completion sequence evaluation (overall shift):'
       + str(sum(sequence_values_overall)/len(sequence_values_overall)))
