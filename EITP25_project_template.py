# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 09:34:34 2019

EITP25 GROUP PROJECT TEMPLATE

@author: Mattias Borg

"""

from brian2 import *

import timeit
import math
import glob

#for loading training data
from datahandling import *
#for plotting stuff
from plottingtools import *

import os
import numpy as np
import matplotlib.pyplot as plt

debug = True #if set to false you will compile the code into c++ code which is slightly faster, but then you can't plot or save stuff during training
useFashion = False #if set to true you will use the MNIST Fashion data set instead of MNIST numbers. Fashion is a harder challenge to be tested at the end of the project

if debug:
    prefs.codegen.target = 'numpy'  # use the Python fallback
    set_device('runtime', build_on_run=False)
else:
    set_device('cpp_standalone', build_on_run=False)
    prefs.devices.cpp_standalone.openmp_threads = 4

if useFashion:
    MNIST_data_path = 'fashion-mnist-master/data/fashion'
else:
    MNIST_data_path = 'number-mnist-master'




# Load MNIST data set
print('LOADING TRAINING DATA - ',end='')
training_data = get_MNIST_data('training', MNIST_data_path)
print('Done.')
print('LOADING TESTING DATA - ',end='')
testing_data = get_MNIST_data('testing', MNIST_data_path, bTrain = False)
print('Done.')


start_scope() #brian2 code, never mind it

rows = training_data['rows']
cols = training_data['cols']
N_input = rows*cols;

#GLOBAL OPTIONS #########################
train = True #if true it trains the network
test = False #only important if train=False, then if test=False then determine neuron classes, if true then test network
restart_sim = False # if set to true the training will start over regardless of whether there are previous versions of the network

plotEvery = 30 #plot information every X images trained (only done in debug mode)
saveEvery = 1000 #save the network state every X images trained (only possible in debug mode)
monitorSpikes = False #if true will create a SpikeMonitor on the network (makes training slower) and will plot the spike activity as a map at every <plotEvery> interval
N_out = 196;  #Number of neurons in output layer, should be N^2, where N is an integer (so that one can plot a nice square matrix of the synapse weights)

latInh = 2 # if =2 use winner takes all as lateral inhibition, if =1 uses soft lateral inhibition, if < 1 use no lateral inhibition at all!

rate_max = 50 #Hz (unit is added later) Maximum spiking rate of input neurons
presentation_time = 250*ms #the time that an image is presented to the network
rest_time = 150*ms #the time inbetween image presentation, to allow for dynamic variables to reach resting states

epochs = 1 #the number of epochs to train for (I have only tested the code for 1 epoch)


network_file = 'network_N=' + str(N_out) + '_latInh='+str(latInh)

#########################################

#STDP parameters

if train:
    learning_rate = 0.01
    T_plus = 10.*mV #adaptive threshold
else:
    learning_rate = 0. #don't change weights
    T_plus = 0.*mV #keep thresholds constant

gmax = 100*nS
Q_max = 1E-15*coulomb
C_post = 1E-12*farad
dV_max = Q_max/C_post #with the standard C_post and Q_max this gives dV_max = 2.5 mV

pot_win = 20*ms #time window for STDP potentiation
dgp = learning_rate*gmax # change of conductivity during potentiation
depression_domination = 0.5 #a factor to make depression dominate over potentation
dgd = learning_rate*gmax*depression_domination # change of conductivity during depression


# tpre is a timer that is reset at each pre spike. On post-spike it is used to check 
# how long since the last prespike and to determine whether to potentiate or depress

stdp_eq = '''
    g : siemens
    dtpre/dt = 1. : second (event-driven)
    '''
on_pre_eq = '''
    v_post += dV_max * g/gmax
    tpre = 0.*ms
    '''
on_pre_eq_inh = 'tpre = 0.*ms'
if latInh > 1:
    on_pre_eq_inh += '\n v_post = v_0'
elif latInh == 1:
    on_pre_eq_inh += '\n v_post -= dV_max'
#else have no lateral inhibition at all!

on_post_eq =  '''    
    g = clip(g + dgp*(tpre <= pot_win)-dgd*(tpre>pot_win),0*nS, gmax)
    '''

#Neuron differential equations
tau_v = 20*ms #time constant for neuron potential decay
v_0 = 0*mV #resting state potential
tau_T = 40*ms #time constant for the threshold


# neuron diff eq
eqs = '''
dv/dt = -(v-v_0)/tau_v : volt
dT/dt = -T/tau_T : volt
T_0 : volt
'''

#Neuron rules
thres = '(v > T_0 + T)'
res = '''
v=v_0
T+=T_plus
'''

####### BUILD NETWORK ########

#Create Explicit Network
N = Network()

# create input layer
I = PoissonGroup(N_input, rates=10*Hz, name='Input_Neurons')


# create output layer
O = NeuronGroup(N_out, eqs, threshold=thres, reset=res, method='exact', name='Output_Neurons')

O.T_0 = 'v_0 + 50.0*mV' #equilibrium threshold

N.add(I,O) # add neuron layers to network

if monitorSpikes:
    OM = SpikeMonitor(O)
    N.add(OM)
   
# connect to input layer
S_i = Synapses(I, O, stdp_eq,on_pre=on_pre_eq, on_post=on_post_eq, name='Excitatory_Synapses')
S_inh = Synapses(O,O, stdp_eq,on_pre=on_pre_eq_inh, name='Inhibitory_Synapses') #inhibitory connections within output layer
N.add(S_i)
N.add(S_inh)

print('CONNECTING INPUT LAYER: ',end='')
allInputs = range(N_input)
for o in range(N_out):
    S_i.connect(i=allInputs, j = o)
S_i.g = 'rand()*gmax'

#connect inhibitory synapses
S_inh.connect(condition='i!=j')
S_inh.g = 'gmax'

print('Done')



if train:

    ###### CONTINUE WITH TRAINING ON A PREVIOUS VERSION OF THE NETWORK? ######
    
    latest_version, images_trained = findLatestVersion(network_file)
    
    if latest_version and not restart_sim:
        #previous version of network exists, so reload the latest version
        
        N.restore(filename=latest_version)
        print('Loaded network from file: ' + latest_version + ' (' + str(images_trained) + ' images trained)')
    else:
        print('No previous training found with this network name.')
    
    ####### TRAIN THE NETWORK ####
    
    print('TRAINING...')
    print('')

    for e in range(epochs):
        #train on each image
        training_counter = 0
        ratess = [0 for x in range(N_input)] #create rate array to fill with rates for each image
        rest_rates = [0 for x in range(N_input)]*Hz #a blank copy to use in the resting state
        
        start = timeit.default_timer() #for checking computing time
        t_s = N.t #simulation start time
        
        for image in training_data['x']:        
            
            training_counter = training_counter + 1 #counts how many images have been trained
            
            if training_counter <= images_trained: #in case we continue an old sim
                continue #skip until next untrained image
                
            #### SET RATES ON INPUT
            j = 0 #counter for transferring into rates
            for r in range(0, rows):
                for c in range(0, cols):                
                    ratess[j] = image[r, c]*rate_max/255.0*Hz #converts greyscale to frequency (rate)
                    
                    j = j + 1       
            I.rates = ratess
            
            
            ######
            
            N.run(presentation_time, profile=True) #present data to network for given time
            
            #turn off input       
            I.rates = rest_rates
            N.run(rest_time, profile=True)          
            
            #FOR PLOTTING AND SAVING DURING A RUN          
            if training_counter % 10 == 0:            
                endtime = timeit.default_timer()
                print('\r\tTrained on #%s images (%.2f s/image )' % (training_counter, (endtime - start)/10))

                if debug and training_counter % plotEvery == 0:
                    plotSynapseMap(S_i, N_input, N_out, rows, cols, gmax)                    
                    if monitorSpikes:
                        t_e = N.t #simulation end time
                        plotLayerActivity(OM, N_out, t_s, t_e, plotEvery)
                        t_s = t_e #new simulation start time
                        #reset spike monitor (for memory purposes)
                        N.remove(OM) #remove from network
                        OM = SpikeMonitor(O) #create a new spike monitor
                        N.add(OM)
                        
                #SAVE NETWORK
                if debug and training_counter % saveEvery == 0:
                    current_version = network_file+'_'+str(training_counter)
                    N.store(filename=current_version) #saves the state (not the objects) of the network
                    print('\r\t---- Saved network state to disk ----')
                    
                start = timeit.default_timer() #for checking computing time
        
else:
    
    if not test:
        #CLASSIFICATION ------
        # NOT IMPLEMENTED. HERE YOU FIRST NEED TO PICK A RANDOM SET OF IMAGES AND LABELS 
        # AND SEE WHICH NEURONS SPIKE MOST TO WHICH. IN THIS WAY ONE CAN GIVE EACH NEURON A "CLASS" FOR TESTING
        print('not yet implemented')    
    else:
        #TESTING -----
        #NOT IMPLEMENTED. HERE YOU GO THROUGH THE TEST DATA AND DETERMINE WHETHER THE NETWORK RESPONDS IN THE WAY THAT
        # IT SHOULD GIVEN THE INPUT DATA. I.E IF THE INPUT IS A "2" DOES THE NEURONS CLASSIFIED AS "2" NEURONS SPIKE MOST?
        #DETERMINE THE ACCURACY OF THE NETWORK HERE.
        print('not yet implemented')    
#Fpr cpp implementation, now compile the code and run it
device.build(directory='output', compile=True, run=True, debug=False)

        




    


