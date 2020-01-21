from brian2 import *

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pickle
#set_device('cpp_standalone') #, build_on_run=False
start_scope()

eqs_adaptive = '''
du/dt = a*((b/mV)*v - u)/(1*ms) : 1

dv/dt = ( alpha*(v - U0)**2 /(mV) + Ubase - x*u*mV  - IsynE - IsynI)/(1*ms)  : volt # - (INaP)
alpha = 0.0125/3 : 1
U0 = -62.5*mV : volt
Ubase = -0*mV :volt
x = 0.06 : 1
a = 0.0011/2 : 1 # interburst interval. The lesser the a, the greater IBI
b = 0.0: 1

gE = 0.1/3 : 1
g_Edrive = 0.1*(y) : 1 #exitatory drive #1*t/60000/ms
y : 1
v_synE = -10*mV : volt
tauE = 10*ms : second

gI = 1 : 1
v_synI = -75*mV : volt
tauI = 15*ms : second
c = -55*mV   : volt 
d : 1
'''

eqs_bursting = '''
du/dt = a*((b/mV)*v - u)/(1*ms) : 1

dv/dt = ( alpha*(v - U0)**2 /(mV) + Ubase - 0.1*u*mV  - IsynE - IsynI)/(1*ms)  : volt # - (INaP)
alpha = 0.004 : 1
U0 = -62.5*mV : volt
Ubase = -1.6*mV :volt
a = 0.0011 : 1 # interburst interval. The lesser the a, the greater IBI
b = 0.2: 1

gE = 0.1 : 1
y : 1
g_Edrive = 0.1*(y) : 1 #exitatory drive #1*t/60000/ms
v_synE = -10*mV : volt
tauE = 10*ms : second

gI = 0.1 : 1
v_synI = -75*mV : volt
tauI = 15*ms : second
c = -50*mV   : volt #- (1*mV)*(u + 12.5)
d : 1
'''


class Population(NeuronGroup):
    #make inheretant of NeuronGroup
    def __init__(self, name, num_nrns, type, bursting=False, N = 50, threshold = 'v >= 20*mV', reset='v = c; u = u + d', refractory='v >= 20*mV'):
        if bursting==True:
            eqs = eqs_bursting
        else:
            eqs = eqs_adaptive

        super().__init__(N, eqs, threshold, reset, refractory)
        self.add_attribute('group_name')
        self.add_attribute('eqs')
        self.add_attribute('type')
        self.add_attribute('ex_syn_variables')
        self.add_attribute('inh_syn_variables')

        self.group_name = name
        self.eqs = eqs
        self.v = -(30 + 0.70 * rand(1, self.N)) * mV
        self.u = (0 + 0.30 * rand(1, self.N)) * 1
        self.d = (+0.3 + 0.05 * rand(1, self.N))
        self.ex_syn_variables = []
        self.inh_syn_variables = []
        self.type = type

    def init_monitorings(self):
        # StateMonitor(self.neural_group, ['v', 'wE', 'wI_1', 'wI_2', 'wI_3'], record=False)
        self.add_attribute('state_mon')
        self.add_attribute('spike_mon')
        self.add_attribute('fr_mon')
        self.state_mon = StateMonitor(self, 'v', record=True)
        self.spike_mon = SpikeMonitor(self)
        self.fr_mon = PopulationRateMonitor(self)

class Network:
    def __init__(self):
        self.populations = dict()
        self.connections = dict()

    def add_population(self, name, num_nrns, type, bursting):
        self.populations[name] = Population(name, num_nrns, type, bursting)

    def specify_connection(self, presyn_group_name, postsyn_group_name, prob):
        # check if these groups exist
        # add extra synaptic variable and specify effect

        if self.populations[presyn_group_name].type == 'excitatory':
            syn_var_str = f'\nwE_{presyn_group_name}_{postsyn_group_name}'
            self.populations[postsyn_group_name].ex_syn_variables.append(syn_var_str)
            self.populations[presyn_group_name].eqs += f'\nd{syn_var_str}/dt = -{syn_var_str} / tauE: 1'
            effect = f'{syn_var_str}_post += 2*0.08*(1+(0.1 * randn()))'

        elif self.populations[presyn_group_name].type == 'inhibitory':
            syn_var_str = f'wI_{presyn_group_name}_{postsyn_group_name}'
            self.populations[postsyn_group_name].inh_syn_variables.append(syn_var_str)
            self.populations[presyn_group_name].eqs += f'\nd{syn_var_str}/dt = -{syn_var_str} / tauI: 1'
            effect = f'{syn_var_str}_post += 2*0.08*(1+(0.1 * randn()))'
        else:
            raise NameError(f'Unknown type of population: {self.populations[presyn_group_name].type}')

        connection = Synapses(self.populations[presyn_group_name],
                              self.populations[postsyn_group_name],
                              on_pre=effect)
        connection.connect(p=prob)
        connection.delay = 2 * ms

        self.connections[f'{presyn_group_name}_{postsyn_group_name}'] = connection

    def connect(self):
        # go through all the synaptic variables and add them into equations
        for name, connection in self.connections.items():
            postsyn_group_name = name.split('_')[-1]
            population = self.populations[postsyn_group_name]
            #excitatory connections
            ex_syn_variables = population.ex_syn_variables
            sum_of_ex_syn_vars_str = ''
            for i in range(len(ex_syn_variables)):
                syn_var_str = ex_syn_variables[i]
                sum_of_ex_syn_vars_str += f'+{syn_var_str}'
            IsynE = f'(gE * ({sum_of_ex_syn_vars_str}) + g_Edrive) * (v - v_synE)'
            population.eqs += f'\nIsynE = {IsynE} : volt'

            #inhibitory connections
            inh_syn_variables = population.inh_syn_variables
            sum_of_inh_syn_vars_str = ''
            for i in range(len(inh_syn_variables)):
                syn_var_str = inh_syn_variables[i]
                population.eqs += f'\nd{syn_var_str}/dt = -{syn_var_str} / tauI: 1'
                sum_of_inh_syn_vars_str += f'+{syn_var_str}'
            IsynI = f'gI * ({sum_of_ex_syn_vars_str}) * (v - v_synI)'
            population.eqs += f'\nIsynI = {IsynI} : volt'

    def set_connectivity(self, connectivity_matrix):
        # for nonzero elements in connectivity matrix
        for postsyn_group_name in connectivity_matrix.keys():
            for presyn_group_name in connectivity_matrix.postsyn_group_name.keys():
                if not connectivity_matrix.postsyn_group_name.presyn_group_name is None:
                    prob = connectivity_matrix.postsyn_group_name.presyn_group_name
                    if prob != 0:
                        # connect groups
                        self.specify_connection(presyn_group_name, postsyn_group_name, prob)
        self.connect()

    def set_drives(self, drives):
        for name, population in self.populations.items():
            population.y = drives.name

    def init_monitorings(self):
        for name, population in self.populations.items():
            population.init_monitorings()
        pass

    def run(self, duration):
        device.build(directory='output', compile=True, run=True, debug=False)
        run(duration * ms, report='text')

if __name__ == '__main__':
    # device.reinit()
    # device.activate()
    # set_device('cpp_standalone', build_on_run=False)
    start_scope()

    N = 100
    Net = Network()
    population_names = ['Pop1', 'Pop2']
    for name in population_names:
        Net.add_population(name, N, type='inhibitory', bursting=False)

    Net.specify_connection('Pop1', 'Pop2', 0.5)
    Net.specify_connection('Pop2', 'Pop1', 0.5)
    Net.connect()

    drives = pd.DataFrame([0.2, 0.2], index=population_names)
    duration = 10000
    Net.init_monitorings()
    device.build()
    run(duration * ms, report='text')
    print(Net.populations['Pop1'].state_mon.v)
    # plt.plot(sm.v)
    # plt.show()



