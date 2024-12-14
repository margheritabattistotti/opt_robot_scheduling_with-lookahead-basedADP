import math
import pickle as pkl
import os
import random
import timeit
import time
import numpy as np
from numpy import infty
import PoissonProcess as Poiss
import OrdersCreation as OC


# BEFORE RUNNING EXPERIMENTS ALWAYS CHECK THE TERMINAL STATE EVALUATION FUNCTION AND THE PRIORITY RULE USED (NEXTORDER)

# ############################################# INPUT DATA EXTRACTION ############################################# #

dir_path = os.path.dirname(os.path.realpath(__file__))
input_data = pkl.load(open(dir_path + "//input_data//input_dict.p", "rb"))

nodes              = input_data["nodes"][:8]            # list of nodes where actions can be performed
# Nodes for smaller instance to compare with DP
# nodes              = input_data["nodes"][:4]            # list of nodes where actions can be performed
# nodes.extend(input_data["nodes"][5:7])                  # 4 boxes, 2 throwing nodes
nodes_coordinates = input_data["nodes_coordinates"]     # dictionary n:c, where n-> node, c-> (x,y) coordinates
# Only 4 objects for smaller instance
objects = input_data["objects"]                         # list of objects that can be picked
O = len(objects)                                        # total number of object types
objects_pick_nodes = input_data["objects_pick_nodes"]   # dictionary o:n, where o-> object, n-> picking node
trays = input_data["trays"]                             # list of target trays where objects can be place
trays.append('tray2')                                   # for LARGE instance, need a third tray
T = len(trays)                                          # total number of trays
trays_coordinates = input_data["trays_coordinates"]     # dictionary t:c, where t-> tray, c-> (x,y) coordinates
trays_coordinates['tray2'] = (131.75, 148.0)            # for LARGE instance, need a third tray
nodes_connections = input_data["nodes_connections"]     # dictionary (n0,n1):(t,r,p), where (n0,n1)-> tuple of nodes,
# t-> time to reach n1 from n0, r-> navigation risk to reach n1 from n0, p-> path to reach n1 from n0


# ######################################## ADDITIONAL PARAMETERS DEFINITION ######################################## #

Thorizon = 12500  # (s) horizon of the optimization
throwing_nodes = nodes[5:8]
# For smaller instance to compare with DP
# Thorizon = 300  # (s) horizon of the optimization
# throwing_nodes = nodes[4:6]
tpick = 7  # (s) time elapsed to perform picking action
tthrow = 5  # (s) time elapsed to perform throwing action
maxPortableObjs = 4
rewards = {'move': 0, 'pick': 10, 'throw': 12}
fail_rewards = {'move': -2, 'pick': 10, 'throw': 0}
evaluation_rewards = {'move': 0, 'pick': 20, 'throw': 25}

rg_move_stream = random.Random(2404)
rg_throw_stream = random.Random(610)


# ##########################################  ORDERS LIST DEFINITION ########################################### #

all_orders = OC.create_all_orders(100)

orders_list = all_orders.copy()
all_orders_get_idx = all_orders.copy()
Twait = Thorizon/len(all_orders)


# ############################################# MISSION DEFINITION ############################################# #

def tot_objects4mission():
    # Returns a list where each entry j specifies how many objects of type j must be placed during mission
    tot_object = [0] * O
    for j in range(O):
        for t in range(T):
            tot_object[j] += list(mission.values())[t][objects[j]]

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

    # Global initialization
    global mission
    global obj4trays_dict
    global tot_obj4mission
    global trays_entering_times
    global orders_entering_ranking
    global tray_has_order
    global all_arrival_times
    global order_idx_cnt  # needed to assign indexes to orders when they enter the system

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
            idx = all_orders_get_idx.index(orders_list[0])
            tray_has_order[tray] = [orders_list[0], idx]
            orders_entering_ranking[idx] = '0s'
            all_orders_get_idx[idx] = 'accessed'
            obj4trays_dict[tray] = sum(list(mission[tray].values()))
            trays_entering_times.append(0)
            orders_list.remove(orders_list[0])

        tot_obj4mission = tot_objects4mission()
        arrival_gap = Thorizon/(len(orders_list)+1)
        # arrival_gap = Thorizon / 60
        arrival_time = 0
        all_arrival_times = [arrival_time] * T
        random.seed(169)
        priority_list = [random.randint(1, 2) for _ in range(len(orders_list))]
        random.seed()
        for order in range(len(orders_list)):
            arrival_time += arrival_gap
            all_arrival_times.append(arrival_time)
            orders_list[order] = [orders_list[order], arrival_time, priority_list[order]]
            arrival_time += len(orders_list)/Thorizon  # ????

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
            idx = all_orders_get_idx.index(orders_list[0])
            tray_has_order[tray] = [orders_list[0], idx]
            orders_entering_ranking[idx] = '0s'
            all_orders_get_idx[idx] = 'accessed'
            obj4trays_dict[tray] = sum(list(mission[tray].values()))
            trays_entering_times.append(0)
            orders_list.remove(orders_list[0])

        tot_obj4mission = tot_objects4mission()
        # Appending of arrival times and priorities
        events = len(orders_list)
        lambda_ = events / Thorizon  # 1 / 150  # Expected value: one arrival every 75 seconds
        arrival_times = Poiss.PoissonProcess_simulation(lambda_, events)
        all_arrival_times = np.concatenate((np.zeros((T,)), arrival_times))
        priorities = [1, 2]

        # Modifying orders_list attaching arrival times and priority to each order
        # Result: array of dimension (len(orders_list), 3)
        for order in range(len(orders_list)):
            orders_list[order] = [orders_list[order], arrival_times[order], random.choice(priorities)]

    else:
        # Update mission and associated information
        mission[trays[tray_index]] = new_order[0]
        idx = all_orders_get_idx.index(new_order[0])
        tray_has_order[trays[tray_index]] = [new_order[0], idx]
        all_orders_get_idx[idx] = 'accessed'
        obj4trays_dict[trays[tray_index]] = sum(list(mission[trays[tray_index]].values()))
        orders_list.remove(new_order)
        tot_obj4mission = tot_objects4mission()
        trays_entering_times[tray_index] = entering_time
        orders_entering_ranking[idx] = str(entering_time)+'s'


def UpdatePriorities(queue, current_t):
    # Every Twait time units, priority levels of writing orders increase. Unless they are maximum
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
    # Entered whenever a tray is full and there are orders waiting
    # Selects the next order based on its priority and arrival time
    if queue:
        highest_priority = queue[0][2]
        prioritized_orders = [order for order in queue if order[2] == highest_priority]
        # prioritized_orders.sort(key=lambda x: x[1])  # It is already ordered by arrival time
        next_order = [order for order in prioritized_orders if order[1] == prioritized_orders[0][1]][0]
        return next_order


# def NextOrder2(queue, current_t):
#     # Entered whenever a tray is full and there are orders waiting
#     # Selects the next order based on its priority and arrival time
#     if queue:
#         higher_priority = 0
#         for order in queue:
#             currentOrder_priority = 1 / order[2] * (current_t - order[1])
#             if currentOrder_priority > higher_priority:
#                 higher_priority = currentOrder_priority
#                 next_order = order
#         return next_order


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


# ############################################### ROLLOUT DEFINITION ############################################### #

gamma = 0.95   # discount factor
M = 10  # number of steps of myopic lookahead


def myopic_rollout(s, m=-1):
    # Recursively executes a rollout from state s, where m is the steps in the future counter
    best_action_type, best_action, best_contribution = myopic_decision(s)  # Call myopic decision
    if best_contribution is not None:
        m += 1
        if m < M:
            next_s, real_contribution, _ = simulation(s, best_action_type, best_action)
            if is_terminal(next_s) is False:
                V = myopic_rollout(next_s, m)
                v = real_contribution + gamma * V
            else:
                v = terminal_value(next_s)
        else:
            v = best_contribution
        return v  # Value of the state
    else:
        # v = 0
        return terminal_value(s)


def myopic_decision(s):
    # Calculates admissible actions given the state and returns the best myopic decision
    a = all_admissible_actions(s)  # Dictionary
    if len(a) == 0:
        return None, None, None
    C = 0
    best_action_type = 'move'
    for action_type in a.keys():
        if rewards[action_type] > C:
            best_action_type = action_type
            C = rewards[action_type] * (Thorizon * 2 - s[0]) / Thorizon
    best_a = random.choice(list(a[best_action_type]))
    return best_action_type, best_a, C


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
        # No moving actions allowed: time horizon reached
        move_actions = None

    return move_actions


# Picking is only allowed in picking nodes, and the robot can only pick the object in the respective box
def picking_actions(s):
    # Returns the possible picking action given the current state s
    # OUTPUT
    # pick_action: picking action identified by the type of object to pick (A=0, B=1, C=2, etc.)
    pick_action = []
    if s[0] + tpick <= Thorizon:
        for j in range(O):
            if list(objects_pick_nodes.values())[j] == s[1] \
                    and s[2 + (T + 1) * j] < tot_obj4mission[j] \
                    and objects_collected(s) < maxPortableObjs:
                pick_action.append(j)

    if len(pick_action) == 0:
        pick_action = None

    return pick_action


# Throwing is allowed only in placing nodes
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
        # No time to perform action
        return None
    for t in range(T):
        for j in range(O):
            placed = tot_objects_placed(s)[j]
            # picked > placed & mission not yet completed
            if s[2 + (T + 1) * j] > placed \
                    and s[3 + (T + 1) * j + t] < list(mission[trays[t]].values())[j]:
                throw_actions.append([j, t])

    if len(throw_actions) == 0:
        # No throwing actions allowed: time horizon reached
        throw_actions = None

    return throw_actions


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

    if tot_objects_placed(s) == tot_obj4mission:
        # Full trays
        if len(orders_list) != 0:
            return infty  # Arbitrary value assigned to differentiate the situations
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
        p = pick[0]
        pick_state[p] = new_state_picking(p, s)
        Q_dict['pick'] = pick_state

    throw_states = {}
    if throw is not None:
        throw = list(filter(lambda x: x is not None, throw))
        for k in throw:
            throw_states[tuple(k)] = new_state_throwing(k, s)
        Q_dict['throw'] = throw_states

    return Q_dict


def all_admissible_actions(s):
    # Returns all admissible actions given the current state
    actions = {}

    move = moving_actions(s)
    pick = picking_actions(s)
    throw = throwing_actions(s)

    if move is not None:
        actions['move'] = move

    if pick is not None:
        actions['pick'] = pick

    if throw is not None:
        actions['throw'] = throw

    return actions


# ######################################## STATE TRANSITIONS DEFINITION ######################################## #

def new_state_moving(a, s):
    # Returns the new state after moving action to destination a, given current state s and exploiting information
    # on their connection {'time': , 'risk': , 'path': }
    next_s = s.copy()
    next_s[0] += nodes_connections[(s[1], a)]['time']
    next_s[1] = a

    return next_s


def failed_new_state_moving(a, s):
    # In case of collision, returns the new state after moving action to destination a, given current state s and
    # exploiting information on their connection {'time': , 'risk': , 'path': }
    next_s = s.copy()
    next_s[0] += nodes_connections[(s[1], a)]['time']
    if next_s[0] + 5 <= Thorizon:
        next_s[0] += 5
    else:
        next_s[0] = Thorizon
    next_s[1] = a

    return next_s


def new_state_picking(a, s):
    # Returns the new state after picking action a (= index of object type to pick : A=0, B=1, etc.),
    # given current state s
    next_s = s.copy()
    next_s[0] += tpick
    next_s[2 + (T + 1) * a] += 1

    return next_s


def new_state_throwing(a, s):
    # Returns the new state after throwing action a (= [index of object type to place: A=0, B=1, etc., index of tray]),
    # given current state s
    next_s = s.copy()
    next_s[0] += tthrow
    next_s[3 + (T + 1) * a[0] + a[1]] += 1

    return next_s


def failed_new_state_throwing(a, s):
    # In case of failure, returns the new state after throwing action a = [index of object to throw, index of tray
    # where to throw it], given current state s
    next_s = s.copy()
    next_s[0] += tthrow
    next_s[2 + (T + 1) * a[0]] -= 1

    return next_s


# ########################################### SIMULATION STEP DEFINITION ########################################### #

def simulation(s, action_type, a):
    # Given an action, simulates the future state of the system give the current one (s)
    reward = rewards[action_type]
    success = 1
    if action_type == 'move':
        p_risk = nodes_connections[(s[1], a)]['risk'] / 100
        if random.random() < p_risk:
            success = 0
            next_state = failed_new_state_moving(a, s)
            reward = fail_rewards['move']
        else:
            next_state = new_state_moving(a, s)
    elif action_type == 'throw':
        reward *= (Thorizon * 2 - s[0]) / Thorizon \
                                        - trays_entering_times[a[1]] / (max(trays_entering_times) + 1) \
                                        + (s[0] - trays_entering_times[a[1]]) / (s[0] - min(trays_entering_times) + 1)
        p_success = throwing_success(nodes_coordinates[s[1]],
                                     trays_coordinates[trays[a[1]]])
        if random.random() > p_success:
            success = 0
            next_state = failed_new_state_throwing(a, s)
            reward = fail_rewards['throw']
        else:
            next_state = new_state_throwing(a, s)
    elif action_type == 'pick':
        reward *= (Thorizon * 2 - s[0]) / Thorizon
        next_state = new_state_picking(a, s)

    return next_state, reward, success


# ########################################### ADDITIONAL NEEDED FUNCTIONS ########################################### #

def is_terminal(s):
    # Checks if the current state s is a terminal state
    terminal = False
    # s is terminal if time horizon is reached, mission is completed or there are no more admissible actions
    if s[0] >= Thorizon or \
            tot_objects_placed(s) == tot_obj4mission or \
            len(all_admissible_actions(s)) == 0:
        terminal = True

    return terminal


def tot_objects_placed(s):
    # Given the state s of the system, returns a list where each entry j specifies how many objects of type j
    # have already been placed
    placed = [0] * O
    for j in range(O):
        for t in range(T):
            placed[j] += s[3 + (T + 1) * j + t]

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
        objs += s[2 + (T + 1) * j] - placed[j]  # picked - placed

    return objs


def trays_level(s):
    # Given the state s of the system, returns a list where each entry t specifies how many objects
    # have already been placed in tray t
    placed = [0] * T
    for t in range(T):
        for j in range(O):
            placed[t] += s[3 + (T+1) * j + t]
    return placed


# ####################################### TERMINAL VALUE FUNCTION DEFINITION ####################################### #

# Terminal value function IV
def terminal_value(s):
    # Value function for a terminal state s
    V = (Thorizon - s[0])
    placed = tot_objects_placed(s)
    for j in range(O):
        V -= tot_obj4mission[j] - placed[j]
        V += s[2 + (T + 1) * j]
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


# ################################################## FORWARD PASS ################################################## #

def forward_pass(initial_s):
    # Defines the optimal strategy to complete the mission following a Myopic Rollout starting from 'initial_s',
    # OUTPUT
    # action_sequence: sequence of optimal actions
    # states_sequence: optimal state sequence
    # objective: terminal state evaluation
    # idle_time: time the robot spent waiting for more orders to arrive
    s = initial_s
    action_sequence = []
    state_sequence = [s]
    completion_rank = {}
    Q = transition_matrix_Q(s)
    best_action = None
    best_new_state = None
    idle_time = 0
    cnt = [0] * len(trays)
    while Q is not None:  # Time horizon not reached or mission not completed yet
        if Q == infty:
            # All trays are full. Just wait until the next order arrival
            action_sequence.append("Waiting for next order.")
            state_sequence.append("Waiting for next order")
            next_arrival_time = math.ceil(orders_list[0][1])
            if next_arrival_time > Thorizon:
                # objective += terminal_state_evaluation(s, idle_time)
                objective_a, objective_e = terminal_state_evaluation(completion_rank)
                return action_sequence, state_sequence, objective_a, objective_e, idle_time, completion_rank
            idle_time += next_arrival_time - s[0] + 1
            s[0] = next_arrival_time
        else:
            for action_type in Q.keys():
                max_V = -1000
                for k in Q[action_type].keys():
                    # For every admissible future state compute a rollout policy
                    # V = Q(s, a) for a that brings the system to "state"
                    state = Q[action_type][k]
                    if action_type == 'throw':
                        p_success = throwing_success(nodes_coordinates[s[1]], trays_coordinates[trays[k[1]]])
                        immediate_contribution = rewards[action_type] * (Thorizon * 2 - s[0]) / Thorizon \
                                              - trays_entering_times[k[1]] / (max(trays_entering_times) + 1) \
                                          + (s[0] - trays_entering_times[k[1]]) / (s[0] - min(trays_entering_times) + 1)
                        V = p_success * (immediate_contribution + myopic_rollout(state, -1)) \
                            + (1-p_success) * (myopic_rollout(failed_new_state_throwing(k, s)))
                    elif action_type == 'move':
                        p_risk = nodes_connections[(s[1], k)]['risk'] / 100
                        V = (1 - p_risk) * myopic_rollout(state, -1) \
                            + p_risk * (-2 + myopic_rollout(failed_new_state_moving(k, s)))
                    else:
                        immediate_contribution = rewards[action_type] * (Thorizon * 2 - state[0]) / Thorizon
                        V = immediate_contribution + myopic_rollout(state, -1)

                    if V > max_V:
                        max_V = V
                        best_new_state = state.copy()
                        best_action_type = action_type
                        best_a = [a for a in Q[action_type] if Q[action_type][a] == best_new_state]
                        best_a = best_a[0]
                        best_action = best_action_type + ' ' + str(best_a)
                    if best_action_type == 'throw':
                        p_success = throwing_success(nodes_coordinates[s[1]], trays_coordinates[trays[best_a[1]]])
                        if rg_throw_stream.random() > p_success:
                            best_new_state = failed_new_state_throwing(best_a, s)
                            best_action = best_action + ' ' + '(FAILURE)'
                    if best_action_type == 'move':
                        p_risk = nodes_connections[(s[1], best_a)]['risk'] / 100
                        if rg_move_stream.random() < p_risk:
                            best_new_state = failed_new_state_moving(best_a, s)
                            best_action = best_action + ' ' + '(COLLISION)'
            state_sequence.append(best_new_state)
            action_sequence.append(best_action)
            s = best_new_state.copy()

        check_full_tray = trays_level(s)
        # If a tray is full and another order is in line, change the mission by emptying such tray and assigning to
        # it the new order
        for t in range(T):
            if check_full_tray[t] == obj4trays_dict[trays[t]]:
                if cnt[t] == 0:
                    action_sequence.append(trays[t] + ' ' + 'is full!')
                    completion_rank[tray_has_order[trays[t]][1]] = str(s[0]) + 's'
                    cnt[t] = 1
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
                    # Updating current state
                    updated_s = s.copy()
                    for j in range(O):
                        # Removing placed objects in tray t from picked entry
                        updated_s[2 + (T + 1) * j] -= updated_s[3 + (T + 1) * j + t]
                        # Resetting to zero the number of placed objects in tray t
                        updated_s[3 + (T + 1) * j + t] = 0
                    state_sequence.append('Changing ' + trays[t])
                    state_sequence.append(updated_s)
                    state_sequence.append(trays[t] + ' changed')
                    s = updated_s

        Q = transition_matrix_Q(s)

    # objective += terminal_state_evaluation(s, idle_time)
    objective_a, objective_e = terminal_state_evaluation(completion_rank)

    return action_sequence, state_sequence, objective_a, objective_e, idle_time, completion_rank


# # ################################################# MAIN EXECTUTION ################################################# #
#
# start = timeit.default_timer()
# time.sleep(1)
#
# set_mission(initial=2)
# objective_values_a = []
# objective_values_e = []
# sequence_values_max = []
# sequence_values_overall = []
# for iters in range(30):
#
#     initial_state = [0] * (O * (T + 1) + 2)
#     initial_state[1] = 'np0'
#
#     actions_seq, states_seq, objective_value_a, objective_value_e, idle_period, orders_completion_ranking \
#         = forward_pass(initial_state)
#
#     sequence_value = completion_sequence_evaluation(orders_completion_ranking, sup=1)
#     sequence_values_max.append(sequence_value)
#     sequence_value2 = completion_sequence_evaluation(orders_completion_ranking, sup=0)
#     sequence_values_overall.append(sequence_value2)
#     objective_values_a.append(objective_value_a)
#     objective_values_e.append(objective_value_e)
#     print(actions_seq)
#     print(states_seq)
#     print('Robot was idle for ' + str(idle_period) + ' seconds.')
#     print('Terminal state evaluation (wrt Arrival): ' + str(objective_value_a))
#     print('Terminal state evaluation (wrt Entrance): ' + str(objective_value_e))
#     print('Orders entering rank: ' + str(orders_entering_ranking))
#     print('Orders completion rank: ' + str(orders_completion_ranking))
#     print('Completion sequence evaluation (maximum shift):' + str(sequence_value))
#     print('Completion sequence evaluation (overall shift):' + str(sequence_value2))
#
# #    print(f"Time taken is {end - start}s")
#
#     # Resetting initial mission and orders for the next iteration
#     orders_list = all_orders.copy()
#     all_orders_get_idx = all_orders.copy()
#     set_mission(initial=2)
# #     tot_obj4mission = tot_objects4mission()
#
# end = timeit.default_timer()
# print('Mean terminal state value (wrt Arrival): ' + str(sum(objective_values_a)/len(objective_values_a)))
# print('Mean terminal state value (wrt Entrance): ' + str(sum(objective_values_e)/len(objective_values_e)))
# print('Mean completion sequence evaluation (maximum shift):' + str(sum(sequence_values_max)/len(sequence_values_max)))
# print('Mean completion sequence evaluation (overall shift):'
#       + str(sum(sequence_values_overall)/len(sequence_values_overall)))
#
# print(f"Mean execution time is {(end - start)/30}s")
