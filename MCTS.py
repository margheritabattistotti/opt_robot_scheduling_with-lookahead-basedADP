import math
import random
import timeit
import time
from numpy import infty
import mcTree as MT
import MR as MR


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


# ################################################## FORWARD PASS ################################################## #

def forward_pass(initial_s):
    # Defines the optimal strategy to complete the mission following a MonteCarlo Tree Search starting
    # from 'initial_s',
    # OUTPUT
    # action_sequence: sequence of optimal actions
    # states_sequence: optimal state sequence
    # objective: terminal state evaluation
    # idle_time: time the robot spent waiting for more orders to arrive
    s = MT.MCTSPreDecisionNode(initial_s)
    action_sequence = []
    state_sequence = [s.state]
    completion_rank = {}
    idle_time = 0
    cnt = [0] * len(MR.trays)
    # Loop until time horizon is reached or there are no more admissible actions (horizon at risk of being surpassed)
    while s.state[0] < MR.Thorizon and len(MR.all_admissible_actions(s.state)) != 0:
        if MR.transition_matrix_Q(s.state) == infty:
            # All trays are full. Just wait until the next order arrival
            action_sequence.append("Waiting for next order.")
            state_sequence.append("Waiting for next order")
            next_arrival_time = math.ceil(MR.orders_list[0][1])
            if next_arrival_time > MR.Thorizon:
                # objective += MR.terminal_state_evaluation(s.state, idle_time)
                objective_a, objective_e = MR.terminal_state_evaluation(completion_rank)
                return action_sequence, state_sequence, objective_a, objective_e, idle_time, completion_rank
            idle_time += next_arrival_time - s.state[0]
            s.state[0] = next_arrival_time
        elif MR.is_terminal(s.state) and len(MR.orders_list) == 0:
            # All trays are full, but all orders have been served -> Exit
            # objective += MR.terminal_state_evaluation(s.state, idle_time)
            objective_a, objective_e = MR.terminal_state_evaluation(completion_rank)
            return action_sequence, state_sequence, objective_a, objective_e, idle_time, completion_rank
        else:
            best_action_type, best_a = s.MCTS_best_action()
            if best_action_type == 'throw':
                best_action = best_action_type + ' ' + str(best_a)
                p_success = throwing_success(MR.nodes_coordinates[s.state[1]], MR.trays_coordinates[MR.trays[best_a[1]]])
                if random.random() > p_success:
                    best_new_state = MR.failed_new_state_throwing(best_a, s.state)
                    best_action = best_action + ' ' + '(FAILURE)'
                else:
                    best_new_state = MR.new_state_throwing(best_a, s.state)
            elif best_action_type == 'move':
                best_action = best_action_type + ' ' + str(best_a)
                p_risk = MR.nodes_connections[(s.state[1], best_a)]['risk'] / 100
                if random.random() < p_risk:
                    best_new_state = MR.failed_new_state_moving(best_a, s.state)
                    best_action = best_action + ' ' + '(COLLISION)'
                else:
                    best_new_state = MR.new_state_moving(best_a, s.state)
            else:
                best_action = best_action_type + ' ' + str(best_a)
                best_new_state = MR.new_state_picking(best_a, s.state)

            state_sequence.append(best_new_state)
            action_sequence.append(best_action)
            s = MT.MCTSPreDecisionNode(best_new_state)

        check_full_tray = MR.trays_level(s.state)
        # If a tray is full and another order is in line, change the mission by emptying such tray and assigning to
        # it the new order
        for t in range(MR.T):
            if check_full_tray[t] == MR.obj4trays_dict[MR.trays[t]]:
                if cnt[t] == 0:
                    action_sequence.append(MR.trays[t] + ' ' + 'is full!')
                    completion_rank[MR.all_orders.index(MR.tray_has_order[MR.trays[t]])] = str(s.state[0]) + 's'
                    cnt[t] = 1
                orders_queue = [order for order in MR.orders_list if order[1] < s.state[0]]
                orders_queue = MR.UpdatePriorities(orders_queue, s.state[0])
                orders_queue.sort(key=lambda x: x[2])
                if len(orders_queue) != 0:
                    action_sequence.append("Let's change " + MR.trays[t])
                    cnt[t] = 0
                    next_order = MR.NextOrder(orders_queue)
                    # next_order = MR.NextOrder2(orders_queue, s.state[0])
                    # Updating mission and associated information
                    MR.set_mission(initial=0, tray_index=t, new_order=next_order, entering_time=s.state[0])
                    action_sequence.append('Now' + ' ' + MR.trays[t] + ' ' + 'contains a new order'
                                           + ' ' + str(MR.mission[MR.trays[t]]))
                    # Updating current state
                    updated_s = s.state.copy()
                    for j in range(MR.O):
                        # Removing placed objects in tray t from picked entry
                        updated_s[2 + (MR.T + 1) * j] -= updated_s[3 + (MR.T + 1) * j + t]
                        # Resetting to zero the number of placed objects in tray t
                        updated_s[3 + (MR.T + 1) * j + t] = 0
                    state_sequence.append('Changing ' + MR.trays[t])
                    state_sequence.append(updated_s)
                    state_sequence.append(MR.trays[t] + ' changed')
                    s = MT.MCTSPreDecisionNode(updated_s)

    # objective += MR.terminal_state_evaluation(s.state, idle_time)
    objective_a, objective_e = MR.terminal_state_evaluation(completion_rank)

    return action_sequence, state_sequence, objective_a, objective_e, idle_time, completion_rank


# ################################################# MAIN EXECUTION ################################################# #

exec_time = []
objective_values_a = []
objective_values_e = []
sequence_values_max = []
sequence_values_overall = []
tot_actions = []

MR.set_mission(initial=2)

for iters in range(15):

    start = timeit.default_timer()
    time.sleep(1)

    initial_state = [0] * (MR.O * (MR.T + 1) + 2)
    initial_state[1] = 'np0'

    actions_seq, states_seq, objective_value_a, objective_value_e, idle_period, orders_completion_ranking \
        = forward_pass(initial_state)

    tot_actions.append(len(actions_seq))

    sequence_value = MR.completion_sequence_evaluation(orders_completion_ranking, sup=1)
    sequence_values_max.append(sequence_value)
    sequence_value2 = MR.completion_sequence_evaluation(orders_completion_ranking, sup=0)
    sequence_values_overall.append(sequence_value2)
    objective_values_a.append(objective_value_a)
    objective_values_e.append(objective_value_e)

    print(actions_seq)
    print(states_seq)
    print('Robot was idle for ' + str(idle_period) + ' seconds.')
    print('Terminal state evaluation (wrt Arrival): ' + str(objective_value_a))
    print('Terminal state evaluation (wrt Entrance): ' + str(objective_value_e))
    print('Orders entering rank: ' + str(MR.orders_entering_ranking))
    print('Orders completion rank: ' + str(orders_completion_ranking))
    print('Completion sequence evaluation (maximum shift):' + str(sequence_value))
    print('Completion sequence evaluation (overall shift):' + str(sequence_value2))

    end = timeit.default_timer()

    exec_time.append(end - start)

    # Resetting initial mission and orders for the next iteration
    MR.orders_list = MR.all_orders.copy()
    MR.set_mission(initial=2)

    # print(f"Time taken is {end - start}s")

print(f"Mean terminal state value (wrt Arrival) is {sum(objective_values_a)/len(objective_values_a)}")
print(f"Mean terminal state value (wrt Entrance) is {sum(objective_values_e)/len(objective_values_e)}")
print(f"Mean execution time is {sum(exec_time)/15}")
print('Mean completion sequence evaluation (maximum shift):' + str(sum(sequence_values_max)/len(sequence_values_max)))
print('Mean completion sequence evaluation (overall shift):'
      + str(sum(sequence_values_overall)/len(sequence_values_overall)))
print('Mean total number of actions: ' + str(sum(tot_actions)/len(tot_actions)))
