import random
import numpy as np
import MR as MR


# ###################################### PRE DECISION STATE TREE NODE CLASS ####################################### #

class MCTSPreDecisionNode:

    def __init__(self, state, parent=None, parent_outcome=None):
        self.state = state  # Pre-decision state node
        self.parent = parent  # Post-decision state node
        self.parent_outcome = parent_outcome  # Outcome happened after post-decision state node
        self.children = []  # Next post-decision states
        self.number_of_visits = 0
        self.Value = 0
        self.tried_actions = {'move': [], 'pick': [], 'throw': []}
        self.untried_actions = MR.all_admissible_actions(self.state)
        self.simulations = 50
        self.max_actions = 5
        self.eps = np.sqrt(3.5)

    # ###################################### MAIN ITERATIONS LOOP DEFINITION ####################################### #

    def MCTS_best_action(self):
        n = 0
        while n < self.simulations:
            future_state = self.TreePolicy()  # SELECTION AND EXPANSION PHASES
            V = future_state.SimPolicy()  # SIMULATION PHASE
            future_state.Value = V
            future_state.Backup()  # BACKPROPAGATION PHASE
            n += 1
        _, best_action_type, best_action = self.best_action()
        return best_action_type, best_action

    # ###################################### SELECTION AND EXPANSION PHASES ####################################### #

    def TreePolicy(self):
        Sttprime = self
        while not Sttprime.is_terminal():
            # SELECTION PHASE
            if sum(len(list(Sttprime.tried_actions.values())[i])
                   for i in range(len(Sttprime.tried_actions))) < self.max_actions:
                # If maximum number of children not reached choose a decision based on immediate contribution,
                # sampled next state and rollout
                if sum(len(list(Sttprime.tried_actions.values())[i]) for i in range(len(Sttprime.tried_actions))) == 0:
                    # Choose action with best immediate contribution.
                    available_actions = MR.all_admissible_actions(Sttprime.state)
                    max_imm_contr = -1
                    for a_type in available_actions.keys():
                        if MR.rewards[a_type] > max_imm_contr:
                            best_a_type = a_type
                            max_imm_contr = MR.rewards[a_type]

                    best_a = random.choice(available_actions[best_a_type])
                else:
                    V_max = -1000
                    available_actions = MR.all_admissible_actions(Sttprime.state)
                    best_a_type = 'move'
                    best_a = available_actions[best_a_type][0]
                    for a_type in available_actions.keys():
                        for a in available_actions[a_type]:
                            next_state, reward, _ = MR.simulation(Sttprime.state, a_type, a)
                            future_myopic_V = MR.myopic_rollout(next_state, -1)
                            V = reward + future_myopic_V
                            if V > V_max:
                                V_max = V
                                best_a_type = a_type
                                best_a = a

                if best_a_type == 'move':
                    if best_a in Sttprime.untried_actions['move']:  # Action not yet tired
                        # EXPANSION PHASE
                        # Create a new child node - post decisione state
                        SttAprime = MTCSPostDecisionNode(MR.new_state_moving(best_a, Sttprime.state), parent=Sttprime,
                                                         parent_action_type=best_a_type, parent_action=best_a)
                        Sttprime.tried_actions['move'].append(best_a)
                        Sttprime.untried_actions['move'].remove(best_a)
                        Sttprime.children.append(SttAprime)
                    else:
                        # Select the already existing child node
                        SttAprime = [s for s in Sttprime.children if s.parent_action_type == 'move'
                                     and s.parent_action == best_a][0]
                elif best_a_type == 'throw':
                    if best_a in Sttprime.untried_actions['throw']:  # Action not yet tried
                        # EXPANSION PHASE
                        # Create new child node - post decisione state
                        SttAprime = MTCSPostDecisionNode(MR.new_state_throwing(best_a, Sttprime.state), parent=Sttprime,
                                                         parent_action_type=best_a_type, parent_action=best_a)
                        Sttprime.tried_actions['throw'].append(best_a)
                        Sttprime.untried_actions['throw'].remove(best_a)
                        Sttprime.children.append(SttAprime)
                    else:
                        # Select the already existing child node
                        SttAprime = [s for s in Sttprime.children if s.parent_action_type == 'throw'
                                     and s.parent_action == best_a][0]
                else:
                    if best_a in Sttprime.untried_actions['pick']:  # Action not yet tried
                        # EXPANSION PHASE
                        # Create new child node - post decisione state
                        SttAprime = MTCSPostDecisionNode(MR.new_state_picking(best_a, Sttprime.state), parent=Sttprime,
                                                         parent_action_type=best_a_type, parent_action=best_a)
                        Sttprime.tried_actions['pick'].append(best_a)
                        Sttprime.untried_actions['pick'].remove(best_a)
                        Sttprime.children.append(SttAprime)
                    else:
                        # Select the already existing child node
                        SttAprime = [s for s in Sttprime.children if s.parent_action_type == 'pick'
                                     and s.parent_action == best_a][0]

            else:
                # Choose the best action based on the current approximations + UCT
                SttAprime, best_a_type, best_a = Sttprime.best_of_children_UCT()

            # OUTCOMES
            if len(SttAprime.happened_outcomes) < len(SttAprime.max_outcomes):
                # If maximum number of outcomes not reached, choose one from un-happened
                outcome = random.choice(SttAprime.unhappened_outcomes)
                SttAprime.unhappened_outcomes.remove(outcome)
                SttAprime.happened_outcomes.append(outcome)
                # EXPANSION PHASE (continue)
                # Create child node - pre decision state
                if best_a_type == 'move':
                    if outcome == 0:  # Failure
                        failed_s = MR.failed_new_state_moving(best_a, Sttprime.state)
                        Sttprime = MCTSPreDecisionNode(failed_s, parent=SttAprime, parent_outcome=0)
                    elif outcome == 1:  # Success
                        Sttprime = MCTSPreDecisionNode(MR.new_state_moving(best_a, Sttprime.state),
                                                       parent=SttAprime, parent_outcome=1)
                elif best_a_type == 'throw':
                    if outcome == 0:  # Failure
                        failed_s = MR.failed_new_state_throwing(best_a, Sttprime.state)
                        Sttprime = MCTSPreDecisionNode(failed_s, parent=SttAprime, parent_outcome=0)
                    elif outcome == 1:  # Success
                        Sttprime = MCTSPreDecisionNode(MR.new_state_throwing(best_a, Sttprime.state),
                                                       parent=SttAprime, parent_outcome=1)
                else:
                    # Picking actions always succeed
                    Sttprime = MCTSPreDecisionNode(MR.new_state_picking(best_a, Sttprime.state),
                                                   parent=SttAprime, parent_outcome=1)

                SttAprime.children.append(Sttprime)

                return Sttprime

            else:
                # Simulate one step and choose the realized outcome and go next (pre decision state) node
                next_state, _, outcome = MR.simulation(Sttprime.state, best_a_type, best_a)
                Sttprime = [s for s in SttAprime.children if s.state == next_state][0]

        return Sttprime

    # ############################################# SIMULATION PHASE ############################################## #

    def SimPolicy(self):
        return MR.myopic_rollout(self.state)

    # ########################################### BACKPROPAGATION PHASE ############################################ #

    # MY WAY - THE CHOSEN
    def Backup(self):
        self.number_of_visits += 1
        if self.parent:
            self.parent.number_of_visits += 1
            E = 0
            for c in range(len(self.parent.children)):
                if self.parent.children[c].parent_outcome == 0:
                    E += self.parent.outcomes_probs()[0] * self.parent.children[c].Value
                elif self.parent.children[c].parent_outcome == 1:
                    E += self.parent.outcomes_probs()[1] * self.parent.children[c].Value

            self.parent.Value = 1 / sum(list(self.parent.outcomes_probs().values())) * E

            if self.parent.parent_action_type == 'throw':
                k = self.parent.parent_action
                reward = self.parent_outcome * MR.rewards[self.parent.parent_action_type] * \
                         (MR.Thorizon * 2 - self.parent.parent.state[0]) / MR.Thorizon \
                         - MR.trays_entering_times[k[1]] / (max(MR.trays_entering_times) + 1) \
                         + (self.parent.parent.state[0] - MR.trays_entering_times[k[1]]) / \
                         (self.parent.parent.state[0] - min(MR.trays_entering_times) + 1)  \
                         + (1 - self.parent_outcome) * MR.fail_rewards[self.parent.parent_action_type]
            else:
                reward = self.parent_outcome * MR.rewards[self.parent.parent_action_type] * \
                         (MR.Thorizon * 2 - self.parent.parent.state[0]) / MR.Thorizon \
                         + (1 - self.parent_outcome) * MR.fail_rewards[self.parent.parent_action_type]

            delta = self.parent.Value + reward

            self.parent.parent.Value = self.parent.parent.Value + \
                                       (delta - self.parent.parent.Value) / (self.parent.parent.number_of_visits + 1)

            self.parent.parent.Backup()

    # ########################################### ACTION SELECTION RULES ############################################ #

    def best_of_children_UCT(self):
        max_value = -1000
        best_child = None
        best_action_type = None
        best_action = None
        for s_a in self.children:
            if s_a.parent_action_type == 'throw':
                k = s_a.parent_action
                imm_reward = MR.rewards[s_a.parent_action_type] * \
                         (MR.Thorizon * 2 - s_a.state[0]) / MR.Thorizon \
                         - MR.trays_entering_times[k[1]] / (max(MR.trays_entering_times) + 1) \
                         + (s_a.state[0] - MR.trays_entering_times[k[1]]) / \
                         (s_a.state[0] - min(MR.trays_entering_times) + 1)
            else:
                imm_reward = MR.rewards[s_a.parent_action_type] * \
                         (MR.Thorizon * 2 - s_a.state[0]) / MR.Thorizon
            if s_a.Value + imm_reward + self.eps * np.sqrt(np.log(self.number_of_visits) / s_a.number_of_visits) \
                    > max_value:
                max_value = s_a.Value + imm_reward \
                            + self.eps * np.sqrt(2 * np.log(self.number_of_visits) / s_a.number_of_visits)
                best_action_type = s_a.parent_action_type
                best_action = s_a.parent_action
                best_child = s_a
        return best_child, best_action_type, best_action

    def best_action(self):
        max_value = -1000
        best_child = None
        best_action_type = None
        best_action = None
        for s_a in self.children:
            if s_a.parent_action_type == 'throw':
                k = s_a.parent_action
                imm_reward = MR.rewards[s_a.parent_action_type] * \
                             (MR.Thorizon * 2 - s_a.state[0]) / MR.Thorizon \
                             - MR.trays_entering_times[k[1]] / (max(MR.trays_entering_times) + 1) \
                             + (s_a.state[0] - MR.trays_entering_times[k[1]]) / \
                             (s_a.state[0] - min(MR.trays_entering_times) + 1)
            else:
                imm_reward = MR.rewards[s_a.parent_action_type] * \
                             (MR.Thorizon * 2 - s_a.state[0]) / MR.Thorizon
            if s_a.Value + imm_reward + self.eps * np.sqrt(np.log(self.number_of_visits) / s_a.number_of_visits) \
                    > max_value:
                max_value = s_a.Value + imm_reward \
                            + self.eps * np.sqrt(2 * np.log(self.number_of_visits) / s_a.number_of_visits)
                best_action_type = s_a.parent_action_type
                best_action = s_a.parent_action
                best_child = s_a
        return best_child, best_action_type, best_action

    def is_terminal(self):
        return MR.is_terminal(self.state)


# ###################################### POST DECISION STATE TREE NODE CLASS ####################################### #

class MTCSPostDecisionNode:

    def __init__(self, state, parent=None, parent_action_type=None, parent_action=None):
        self.state = state  # Post-decision state node
        self.parent = parent  # Pre-decision state node
        self.parent_action_type = parent_action_type  # Action chosen from pre-decision state node
        self.parent_action = parent_action
        self.children = []  # Next pre-decision states
        self.number_of_visits = 0  # Actually number of visits associated to parent + taken action parent action
        self.Value = 0
        self.happened_outcomes = []
        self.unhappened_outcomes = self.available_outcomes()
        self.max_outcomes = self.available_outcomes()

    def available_outcomes(self):
        # Based on the action chosen by the parent node, creates a list of possible outcomes
        # 0 = FAILURE, 1 = SUCCESS
        if self.parent_action_type == 'move':
            if MR.nodes_connections[(self.parent.state[1], self.state[1])]['risk'] / 100 == 0:
                # If no risk, always success
                return [1]
            else:
                return [0, 1]
        elif self.parent_action_type == 'throw':
            if MR.throwing_success(MR.nodes_coordinates[self.parent.state[1]],
                                   MR.trays_coordinates[MR.trays[self.parent_action[1]]]) == 1:
                # If no risk, always success
                return [1]
            else:
                return [0, 1]
        else:
            return [1]  # Always success for picking actions

    def outcomes_probs(self):
        # Associates to each happened outcome the probability of its realization
        # Outputs a dictionary of the kind {outcome: probability}
        probs = {}
        for outc in self.happened_outcomes:
            if self.parent_action_type == 'move':
                risk = MR.nodes_connections[(self.parent.state[1], self.state[1])]['risk'] / 100
                if outc == 0:
                    probs[0] = risk
                elif outc == 1:
                    probs[1] = 1 - risk
            elif self.parent_action_type == 'throw':
                success = MR.throwing_success(MR.nodes_coordinates[self.parent.state[1]],
                                              MR.trays_coordinates[MR.trays[self.parent_action[1]]])
                if outc == 1:
                    probs[1] = success
                elif outc == 0:
                    probs[0] = 1 - success
            else:
                probs[1] = 1

        return probs