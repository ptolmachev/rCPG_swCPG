import numpy as np
from collections import deque
from copy import deepcopy
from matplotlib import pyplot as plt

def firing_rate(v):
    return 1.0 / (1 + np.exp(-0.4 * v))

class Neuron():
    '''
    A neuron class with simple continuous dynamics exhibiting firing rate adaptation
    '''
    def __init__(self, dt, drive, tau):
        self.dt = dt
        self.state = np.array([np.random.rand(), 0.2 + 0.2 * np.random.rand()])
        self.state_history = deque()
        self.drive = drive
        self.tau = tau

    def rhs(self, inp):
        v, m = self.state
        rhs_v = -0.01 * v - m + self.drive + inp
        rhs_m = (self.fr() - m) / self.tau
        return np.array([rhs_v, rhs_m])

    def fr(self):
        '''
        Calculates firing rate of a neuron based on its average 'voltage'
        '''
        return firing_rate(self.state[0])

    # def fr(self):
    #     '''
    #     Calculates firing rate of a neuron based on its average 'voltage'
    #     '''
    #     limit = 5
    #     return (np.clip(self.state[0], -limit, limit) + limit) / (2 * limit)

    def get_next_state(self, inp):
        state = self.state + self.dt * self.rhs(inp)
        return state

    def get_state_history(self):
        return np.array(self.state_history)

    def update_history(self):
        self.state_history.append(deepcopy(self.state))
        return None

    def step(self, inp):
        self.state += self.dt * self.rhs(inp)
        self.update_history()
        return None


class HCO():
    '''
    Half centre oscillator made from two mutually inhibiting neurons with firing rate adaptation
    '''
    def __init__(self, dt, drives, weights, tau):
        self.weights = weights
        self.drives = drives
        self.dt = dt
        self.tau = tau
        self.state_history = deque()
        self.nrn1 = Neuron(dt=self.dt, drive=drives[0], tau=tau)
        self.nrn2 = Neuron(dt=self.dt, drive=drives[1], tau=tau)

    def fr(self):
        vs = np.array([self.nrn1.state[0], self.nrn2.state[0]])
        return firing_rate(vs)

    def get_next_state(self, inputs):
        # 'inputs' should be a two-elements array
        # calculate synaptic inhibition between the two neurons
        mutual_synaptic_inhibition = (self.weights * self.fr())[::-1]
        synaptic_input_total = mutual_synaptic_inhibition + inputs
        state1 = self.nrn1.state + self.dt * self.nrn1.rhs(synaptic_input_total[0])
        state2 = self.nrn2.state + self.dt * self.nrn2.rhs(synaptic_input_total[1])
        return np.hstack([state1, state2])

    def update_history(self):
        self.nrn1.state_history.append(deepcopy(self.nrn1.state))
        self.nrn2.state_history.append(deepcopy(self.nrn2.state))
        return None

    def step(self, inputs):
        # update the state of hco
        self.state = self.get_next_state(inputs)
        # update the state of individual neurons
        self.nrn1.state = deepcopy(self.state[:2])
        self.nrn2.state = deepcopy(self.state[2:])
        # save values to the history
        self.update_history()
        return None

    def get_state_history(self):
        return np.hstack([self.nrn1.get_state_history(), self.nrn2.get_state_history()])

    def run(self, T_steps):
        for i in range(T_steps):
            self.step(inputs=np.zeros(2))
        return None

class Model_CPG_interaction():
    def __init__(self, params):
        self.dt = params['dt']

        self.swCPG = HCO(dt=self.dt, drives=params['drives']['swCPG'], weights=params['HCO_weights']['swCPG'],
                         tau=params['tau']['swCPG'])
        self.rCPG = HCO(dt=self.dt, drives=params['drives']['rCPG'], weights=params['HCO_weights']['rCPG'],
                         tau=params['tau']['rCPG'])
        self.KF = Neuron(dt=self.dt, drive=params['drives']['KF'], tau=params['tau']['KF'])
        self.Relay = Neuron(dt=self.dt, drive=params['drives']['Relay'], tau=params['tau']['Relay'])

        #Interactions between the blocks
        self.W = dict()
        self.W['Relay', 'rCPG'] = params['W']['Relay', 'rCPG']  # 2 weights from the Relay neuron to swCPG (2 neurons)
        self.W['Relay', 'swCPG'] = params['W']['Relay', 'swCPG']  # 2 weights from the Relay neuron to rCPG (2 neurons)
        self.W['Relay', 'KF'] = params['W']['Relay', 'KF']  # 1 weight from the Relay neuron to KF
        self.W['KF', 'rCPG'] = params['W']['KF', 'rCPG'] # 2 weights from the KF neuron to rCPG (2 neurons)
        self.W['KF', 'swCPG'] = params['W']['KF', 'swCPG'] # 2 weights from the KF neuron to swCPG (2 neurons)
        self.W['rCPG', 'swCPG'] = params['W']['rCPG', 'swCPG']  # 2x2 weight-matrix from rCPG to swCPG
        self.W['swCPG', 'rCPG'] = params['W']['swCPG', 'rCPG']  # 2x2 weight-matrix from swCPG to rCPG


    def calc_inputs(self):
        inputs = dict()
        inputs['Relay', 'swCPG'] = self.W['Relay', 'swCPG'] * self.Relay.fr()
        inputs['Relay', 'rCPG'] = self.W['Relay', 'rCPG'] * self.Relay.fr()
        inputs['Relay', 'KF'] = self.W['Relay', 'KF'] * self.Relay.fr()
        inputs['KF', 'rCPG'] = self.W['KF', 'rCPG'] * self.KF.fr()
        inputs['KF', 'swCPG'] = self.W['KF', 'swCPG'] * self.KF.fr()
        inputs['rCPG', 'swCPG'] = self.W['rCPG', 'swCPG'].T @ self.rCPG.fr()
        inputs['swCPG', 'rCPG'] = self.W['swCPG', 'rCPG'].T @ self.swCPG.fr()

        summed_inputs = dict()
        summed_inputs['swCPG'] = (inputs['Relay', 'swCPG'] + inputs['KF', 'swCPG'] + inputs['rCPG', 'swCPG'])
        summed_inputs['rCPG'] = (inputs['Relay', 'rCPG'] + inputs['KF', 'rCPG'] + inputs['swCPG', 'rCPG'])
        summed_inputs['KF'] = inputs['Relay', 'KF']
        return summed_inputs


    def update_history(self):
        self.Relay.update_history()
        self.KF.update_history()
        self.rCPG.update_history()
        self.swCPG.update_history()
        return None

    def step(self, input_to_Relay):
        inputs = self.calc_inputs()
        # update the state of individual blocks
        self.Relay.step(input_to_Relay)
        self.KF.step(inputs['KF'])
        self.rCPG.step(inputs['rCPG'])
        self.swCPG.step(inputs['swCPG'])
        return None

    def get_state_history(self):
        return np.hstack([self.Relay.get_state_history(), self.KF.get_state_history(),
                          self.rCPG.get_state_history(), self.swCPG.get_state_history()])

    def run(self, T_steps, input_to_Relay):
        for i in range(T_steps):
            self.step(input_to_Relay)
        return None


if __name__ == '__main__':
    # # checking HCO
    # dt = 0.1
    # drives = 0.5 * np.ones(2)
    # weights = 0.25 * np.ones(2)
    # tau = 1500
    # hco = HCO(dt=dt, drives=drives, weights=weights, tau=tau)
    #
    # T = 10 #sec
    # T_steps = int(T * 1000 / dt)
    #
    # hco.run(T_steps)
    # state_history = hco.get_state_history()
    # fig = plt.figure(figsize=(14, 7))
    # plt.plot(state_history[:, 0], label="First Neuron")
    # plt.plot(state_history[:, 2], label="Second Neuron")
    # plt.legend(fontsize=15)
    # plt.title("Half-centre oscillator dynamics", fontsize=25)
    # plt.xlabel("t", fontsize=15)
    # plt.ylabel("v", fontsize=15)
    # plt.show()
    # ##############################################################

    # testing the full model
    params = dict()
    params['dt'] = dt = 0.1
    params['tau'] = dict()
    params['tau']['swCPG'] = 500
    params['tau']['rCPG'] = 2500
    params['tau']['KF'] = 3500
    params['tau']['Relay'] = 3500

    params['drives'] = dict()
    params['drives']['swCPG'] = np.array([0.07, 0.2])
    params['drives']['rCPG'] = np.array([0.5, 0.35])
    params['drives']['KF'] = 0.55
    params['drives']['Relay'] = -0.1

    params['HCO_weights'] = dict()
    params['HCO_weights']['swCPG'] = np.array([-0.65, -0.37])
    params['HCO_weights']['rCPG'] = np.array([-0.25, -0.25])

    params['W'] = dict()
    params['W']['Relay', 'swCPG'] = np.array([0.5, 0.35])
    params['W']['Relay', 'rCPG'] = np.array([0.1, -0.45])
    params['W']['Relay', 'KF'] = 0.6
    params['W']['KF', 'rCPG'] = np.array([0.4, 0.05])
    params['W']['KF', 'swCPG'] = np.array([-0.15, -0.17])
    params['W']['swCPG', 'rCPG'] = 0 * np.array([[+0.01, -0.01], [+0.01, -0.01]])
    params['W']['rCPG', 'swCPG'] = 0 * np.array([[0, -0.01], [0, -0.01]])
    model = Model_CPG_interaction(params)

    T = 10 #sec
    T_steps = int(T * 1000 / dt)
    model.run(T_steps, input_to_Relay=0)
    model.run(T_steps, input_to_Relay=0.55)
    model.run(T_steps, input_to_Relay=0)

    state_history = model.get_state_history()
    N = state_history.shape[-1] // 2
    fig, axes = plt.subplots(N, 1, figsize=(14, 7))
    labels = ['Sensory Relay', "KF", "rCPG expiratory", "rCPG inspiratory", "swCPG 1", "swCPG 2"]
    for i in range(len(axes)):
        axes[i].plot(firing_rate(state_history[:, 2 * i]), label=labels[i], color='k', linewidth=3)
        axes[i].legend(fontsize=15, loc=4)
        axes[i].set_ylim([0, 1.05])
    plt.suptitle("Simplified model dynamics", fontsize=25)
    axes[i].set_xlabel("t", fontsize=15)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()