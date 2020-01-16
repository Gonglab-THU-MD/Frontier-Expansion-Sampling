#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Juanrong Zhang
"""

import numpy as np
import mdtraj as md

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit
import os

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull

from numpy import *
from numpy.linalg import *





#build the system and it can be used by initial short simulation and seed simulation
def build_system(prmtop):
    system = prmtop.createSystem(nonbondedMethod = omma.PME, 
                                 nonbondedCutoff = 1*unit.nanometer, 
                                 constraints = omma.HBonds)
    integrator = omm.LangevinIntegrator(310*unit.kelvin, 1/unit.picoseconds, 
                                        0.002*unit.picoseconds)
    platform = omm.Platform.getPlatformByName('CUDA')
    #properties = {'DeviceIndex':'2'}
    simulation = omma.Simulation(prmtop.topology, system, integrator, platform)
    #simulation = omma.Simulation(prmtop.topology, system, integrator)

    return simulation

#run short eq simulation
def run_short_eq(simulation, inpcrd, steps):
    simulation.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    simulation.minimizeEnergy()

    simulation.reporters.append(omma.DCDReporter('eq.dcd', 5000))
    simulation.reporters.append(omma.StateDataReporter('eq_log.txt', 5000, 
                                                       step=True, 
                                                       potentialEnergy=True, 
                                                       temperature=True))
    simulation.step(steps)

    return 1

#run seed simulation by reading the coordinates of seed and initialize velocity based on temperature
def run_seed_simulation(simulation, inpcrd, steps, seed_index, coming_cycle):
    simulation.context.setPositions(inpcrd.positions)
    simulation.context.setVelocitiesToTemperature(310)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    simulation.reporters.append(omma.DCDReporter('./'+str(coming_cycle)+'/'+
                                                 str(seed_index)+'.dcd',5000))
    simulation.reporters.append(omma.StateDataReporter('./'+str(coming_cycle)+'/'
                                                       +str(seed_index)+'.txt', 
                                                       5000, step=True, 
                                                       potentialEnergy=True, 
                                                       temperature=True))

    simulation.step(steps)
    


def get_proj_and_rst(coming_cycle,lis):
    os.system('mkdir '+str(coming_cycle))
    
    #read coordinates, perform PCA and get projection
    coord=[]
    ensemble = md.load_pdb('./open_amber.pdb')
    topology = ensemble.topology
    ref=md.load_pdb('./open_amber.pdb',atom_indices=topology.select('name CA'))
    if coming_cycle==1:
        dcd_bb=md.load('./eq.dcd',top='./open_amber.pdb',
                       atom_indices=topology.select('name CA'))
        dcd=md.load('./eq.dcd',top='./open_amber.pdb')
    else: 
        
        filennames=['./eq.dcd']+['./'+str(index_cycle+1)+'/'+str(index_seed+1)+'.dcd'  
                    for index_cycle in range(len(lis))
                    for index_seed in range(lis[index_cycle]) 
                    ]
        dcd_bb=md.load(filennames, top='./open_amber.pdb', atom_indices=topology.select('name CA'))
        dcd=md.load(filennames, top='./open_amber.pdb')
        
    dcd_bb=dcd_bb.superpose(reference=ref)
    coord.extend(dcd_bb.xyz)
    coord=np.array(coord)
    res=coord
    d_1,d_2,d_3=res.shape
    res=res.reshape(d_1,d_2*d_3)
    
    mean=np.mean(res,axis=0)
    res_new=res-mean
    pca=PCA(n_components=3)
    pca.fit(res_new)
    print(pca.explained_variance_ratio_)
    proj=np.dot(res_new,pca.components_.T)
    np.savetxt('./'+str(coming_cycle)+'/proj.txt',proj,fmt="%.2f")
    
    #get the index of seeds based on SDS
    choose=zeros(len(proj),dtype=bool)
    choose[norm(proj,axis=1).argmax()]=True
    seq=[norm(proj,axis=1).argmax()]
    needchoose=100
    while choose.sum()<needchoose:
        id1=proj[choose].dot(proj[~choose].T).sum(axis=0).argmin()
        seq.append((~choose).nonzero()[0][id1])
        choose[(~choose).nonzero()[0][id1]]=True  


    np.savetxt('./'+str(coming_cycle)+'/seq.txt',seq,fmt='%d')
    dcd_save=dcd[seq]
    dcd_save.save_amberrst7('./'+str(coming_cycle)+'/rst')
    
    return seq









#read prmtop and inpcrd file of system
prmtop = omma.AmberPrmtopFile('./open_amber.prmtop')
initial_inpcrd = omma.AmberInpcrdFile('./open_amber.inpcrd')

simulation = build_system(prmtop)

#run 10 ns eq simulation
run_short_eq(simulation, initial_inpcrd, 5000000)

#the number of seeds at each cycle is saved here
lis=[]

#set the cycle number
num_cycle=30
for i in range(20,num_cycle,1):
    coming_cycle=i+1
    vertix_index=get_proj_and_rst(coming_cycle,lis)
    lis.append(len(vertix_index))
    for j in range(len(vertix_index)):
        simulation_seed=build_system(prmtop)
        seed_inpcrd=omma.AmberInpcrdFile('./'+str(coming_cycle)+'/rst.'+
                                         ("{0:0"+str(len(str(len(vertix_index))))+"d}").format(j+1))
        run_seed_simulation(simulation_seed,seed_inpcrd,50000,j+1,coming_cycle)

            
                
            

    
         
         

         
         
         
         


