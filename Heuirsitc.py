import pickle as pkl
import os
import random
import timeit
import time
import math
import OrdersCreation as OC
import numpy as np
import PoissonProcess as Poiss

# ADD

# ############################################# INPUT DATA EXTRACTION ############################################# #

dir_path = os.path.dirname(os.path.realpath(__file__))
input_data = pkl.load(open(dir_path + "//input_data//input_dict.p", "rb"))

nodes              = input_data["nodes"][:6]            # list of nodes where actions can be performed
nodes_coordinates = input_data["nodes_coordinates"]     # dictionary n:c, where n-> node, c-> (x,y) coordinates
objects = input_data["objects"]                         # list of objects that can be picked
O = len(objects)                                        # total number of object types
objects_pick_nodes = input_data["objects_pick_nodes"]   # dictionary o:n, where o-> object, n-> picking node
trays = input_data["trays"]                             # list of target trays where objects can be place
T = 1                                        # total number of trays
trays_coordinates = input_data["trays_coordinates"]     # dictionary t:c, where t-> tray, c-> (x,y) coordinates
nodes_connections = input_data["nodes_connections"]     # dictionary (n0,n1):(t,r,p), where (n0,n1)-> tuple of nodes,
# t-> time to reach n1 from n0, r-> navigation risk to reach n1 from n0, p-> path to reach n1 from n0


# ######################################## ADDITIONAL PARAMETERS DEFINITION ######################################## #

Thorizon = 12500  # (s) horizon of the optimization
tpick = 7  # (s) time elapsed to perform picking action
tthrow = 5  # (s) time elapsed to perform throwing action

rg_move_stream = random.Random(2404)
rg_throw_stream = random.Random(610)

# ############################################# MISSION DEFINITION ############################################# #


def set_mission(poiss=1):

    global orders
    global Twait
    global initial_len_orders
    global mission
    global entering_times
    global completion_times
    global all_arrival_times

    orders = OC.create_all_orders(100)
    Twait = Thorizon/len(orders)
    initial_len_orders = len(orders)
    mission = {}
    entering_times = []
    completion_times = []
    next_order = orders[0]
    mission['tray0'] = next_order
    entering_times.append(0)
    orders.remove(next_order)

    if poiss == 1:
        events = len(orders)
        # lambda_ = events / Thorizon
        lambda_ = 0.01
        arrival_times = Poiss.PoissonProcess_simulation(lambda_, events)
        all_arrival_times = np.concatenate((np.zeros((T,)), arrival_times))
    else:
        # arrival_gap = Thorizon / (len(orders) + 1)
        arrival_gap = 100
        arrival_times = [0] * len(orders)
        for i in range(len(arrival_times)):
            arrival_times[i] = arrival_gap * (i+1)
        all_arrival_times = np.concatenate((np.zeros((T,)), arrival_times))

    priorities = [1, 2]

    for order in range(len(orders)):
        orders[order] = [orders[order], arrival_times[order], random.choice(priorities)]


def NextOrder(queue):
    # Entered whenever a tray is full and there are orders waiting
    # Selects the next order based on its priority and arrival time
    if queue:
        highest_priority = queue[0][2]
        prioritized_orders = [order for order in queue if order[2] == highest_priority]
        # prioritized_orders.sort(key=lambda x: x[1])  # It is already ordered by arrival time
        next_order = [order for order in prioritized_orders if order[1] == prioritized_orders[0][1]][0]
        return next_order


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


# Defined by me
maxPortableObjs = 4
rewards = {'move': 0, 'pick': 10, 'place': 12}

# I can only place or throw from placing nodes (to any tray)
placing_nodes = nodes[-1]


# ################################### DEFINITION OF THROWING SUCCESS PROBABILITY ################################### #
# Average throwing success probability from any throwing location to any tray
throwing_success = 0.8510161256706374


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
    # Returns the new state after placing action a (= [index of object type to place: A=0, B=1, etc., index of tray]),
    # given current state s
    new_s = s.copy()
    new_s[0] += tthrow
    new_s[3 + (T+1) * a[0] + a[1]] += 1
    return new_s


# ########################################### ADDITIONAL NEEDED FUNCTIONS ########################################### #

def tot_objects_placed(s):
    # Given the state s of the system, returns a list where each entry j specifies how many objects of type j
    # have already been placed
    placed = [0] * O
    for j in range(len(placed)):
        for t in range(T):
            placed[j] += s[3 + (T+1) * j + t]
    return placed


def objects_collected(s):
    # Yields the number of objects collected but not yet placed (inferred from the state values)
    # INPUT
    # s: current state of the system
    # OUTPUT
    # obj: the number of objects collected but not yet placed in the given state
    objs = 0
    placed = tot_objects_placed(s)
    for j in range(O):
        objs += s[2 + (T+1) * j] - placed[j]  # picked - placed
    return objs


# ################################################# MAIN EXECTUTION ################################################# #

start = timeit.default_timer()
time.sleep(1)

set_mission(poiss=1)
objective_values_a = []
objective_values_e = []
orders_completed = []

for iters in range(30):
    idle_time = 0
    for o in range(initial_len_orders):
        actions = []
        states = []
        if o == 0:
            # First order
            init = [0] * (O * (T + 1) + 2)
            # init[1] = random.choice(nodes)
            init[1] = 'np0'
            states.append(init)
            current_state = init.copy()
        else:
            # Other orders
            init = [0] * (O * (T + 1) + 2)
            init[0] = t_restart  # Re-start from time of previous order completion
            init[1] = old_states[-1][1]  # Re-start from node of previous order completion
            states.append(init)
            current_state = init.copy()
        if init[1] != 'np0':
            # Always (re)start from np0
            current_state = new_state_moving('np0', init)
            p_risk = nodes_connections[(init[1], 'np0')]['risk'] / 100
            action = 'move np0'
            if rg_move_stream.random() < p_risk:
                current_state[0] += 5  # reach next state but with 5 second penalty
                action = action + ' ' + '(COLLISION)'
            states.append(current_state)
            actions.append(action)
        j = 0  # object_type and box index
        while j < O:
            # Until all objects of type j are picked or the robot has reached maximum capacity
            while current_state[2 + (T+1) * j] < mission['tray0'][objects[j]] and \
                    objects_collected(current_state) < maxPortableObjs:
                next_state = new_state_picking(j, current_state)
                states.append(next_state)
                actions.append('pick' + ' ' + str(j))
                current_state = next_state.copy()
            if objects_collected(current_state) < maxPortableObjs and j < O - 1:
                # If the robot has not reached maximum capacity move to the next box (if there is one)
                next_state = new_state_moving(nodes[j + 1], current_state)
                action = 'move' + ' ' + nodes[j + 1]
                p_risk = nodes_connections[(current_state[1], nodes[j+1])]['risk'] / 100
                if rg_move_stream.random() < p_risk:
                    next_state[0] += 5  # Reach next state but with 5 second penalty
                    action = action + ' ' + '(COLLISION)'
                states.append(next_state)
                actions.append(action)
                current_state = next_state.copy()
            else:
                # Move to the placing location to throw the objects collected so far
                next_state = new_state_moving(placing_nodes, current_state)
                action = 'move' + ' ' + placing_nodes
                p_risk = nodes_connections[(current_state[1], placing_nodes)]['risk'] / 100
                if rg_move_stream.random() < p_risk:
                    next_state[0] += 5  # Reach next state but with 5 second penalty
                    action = action + ' ' + '(COLLISION)'
                states.append(next_state)
                actions.append(action)
                current_state = next_state.copy()
                for i in range(j+1):  # For all object types collected so far
                    if current_state[2 + (T+1) * i] != current_state[3 + (T+1) * i]:
                        # If the robot has some objects of type i to place
                        for k in range(current_state[3 + (T+1) * i], current_state[2 + (T+1) * i]):
                            # Place all the collected objects of type i
                            next_state = new_state_throwing([i, 0], current_state)
                            action = 'throw' + ' ' + str(i)
                            if rg_throw_stream.random() > throwing_success:
                                next_state[2 + (T + 1) * i] -= 1  # Throwing failed: object lost
                                next_state[3 + (T + 1) * i] -= 1  # Remove object counted as placed
                                action = action + ' ' + '(FAILED)'
                            states.append(next_state)
                            actions.append(action)
                            current_state = next_state.copy()
                if tot_objects_placed(current_state) != list(mission['tray0'].values()):  # Order not completed
                    if tot_objects_placed(current_state)[j] != list(mission['tray0'].values())[j]:
                        # If there are remaining objects of type j to pick
                        # Go back to the last box the robot was at before going to placing node
                        next_state = new_state_moving(nodes[j], current_state)
                        action = 'move' + ' ' + nodes[j]
                        p_risk = nodes_connections[(current_state[1], nodes[j])]['risk'] / 100
                        if rg_move_stream.random() < p_risk:
                            next_state[0] += 5  # Reach next state but with 5 second penalty
                            action = action + ' ' + '(COLLISION)'
                        states.append(next_state)
                        actions.append(action)
                        current_state = next_state.copy()
                        j -=1
                    else:
                        if nodes[j+1] == 'nt0':
                            # Go back to box 0
                            j = -1
                        # Go to the next box wrt the one the robot was at before going to placing node
                        next_state = new_state_moving(nodes[j+1], current_state)
                        action = 'move' + ' ' + nodes[j+1]
                        p_risk = nodes_connections[(current_state[1], nodes[j+1])]['risk'] / 100
                        if rg_move_stream.random() < p_risk:
                            next_state[0] += 5  # Reach next state but with 5 second penalty
                            action = action + ' ' + '(COLLISION)'
                        states.append(next_state)
                        actions.append(action)
                        current_state = next_state.copy()
            if tot_objects_placed(current_state) == list(mission['tray0'].values()):
                completion_times.append(current_state[0])
                break
            j += 1

        print(actions)
        print(states)

        old_states = states.copy()
        t_restart = old_states[-1][0]

        if len(orders) != 0:
            # Change order and start again
            orders = UpdatePriorities(orders, t_restart)
            next_order = NextOrder(orders)
            mission['tray0'] = next_order[0]
            orders.remove(next_order)
            if next_order[1] > t_restart:
                next_arrival_time = math.ceil(next_order[1])
                if next_arrival_time > Thorizon:
                    break
                idle_time += next_arrival_time - t_restart
                t_restart = next_arrival_time
            entering_times.append(t_restart)


    print(f"Idle time is {idle_time}s")

    obj_val_temp_a = np.array(completion_times) - all_arrival_times[:len(completion_times)]
    obj_val_temp_e = np.array(completion_times) - np.array(entering_times)
    orders_completed.append(len(obj_val_temp_a))
    objective_values_a.append(sum(obj_val_temp_a)/len(obj_val_temp_a))
    objective_values_e.append(sum(obj_val_temp_e)/len(obj_val_temp_e))

    # Resetting initial mission and orders for the next iteration
    set_mission(poiss=1)

end = timeit.default_timer()
print(f"Average time taken is {(end - start)/30}s")
print(f"Orders completed on average {sum(orders_completed)/len(orders_completed)}")
print('Mean terminal state value (wrt Arrival): ' + str(sum(objective_values_a)/len(objective_values_a)))
print('Mean terminal state value (wrt Entrance): ' + str(sum(objective_values_e)/len(objective_values_e)))