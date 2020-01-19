#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Juanrong Zhang
"""

#set the name of system
system_name="open_amber"

#set the number of cycle
num_cycle=30

#the list that save the number of seeds in each cycle
lis=[]


import numpy as np
import mdtraj as md

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit
import os

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull

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
    ensemble = md.load_pdb(system_name+'.pdb')
    topology = ensemble.topology
    ref=md.load_pdb(system_name+'.pdb',atom_indices=topology.select('name CA'))
    if coming_cycle==1:
        dcd_bb=md.load('./eq.dcd',top=system_name+'.pdb',
                       atom_indices=topology.select('name CA'))
        dcd=md.load('./eq.dcd',top=system_name+'.pdb')
    else: 
        
        filennames=['./eq.dcd']+['./'+str(index_cycle+1)+'/'+str(index_seed+1)+'.dcd'  
                    for index_cycle in range(len(lis))
                    for index_seed in range(lis[index_cycle]) 
                    ]
        dcd_bb=md.load(filennames, top=system_name+'.pdb', atom_indices=topology.select('name CA'))
        dcd=md.load(filennames, top=system_name'.pdb')
        
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
    
    #perform GMM
    probability_cutoff=0.1
    BIC_cutoff=0.3    

    n_components=np.arange(1,21)         
    models=[GaussianMixture(n, covariance_type='full',random_state=0).fit(proj) for n in n_components]
    bic=[m.bic(proj) for m in models]
    print(bic)
    
    #determine the number of components of GMM using BIC_cutoff 0.3
    slope=(bic-min(bic))/(max(bic)-min(bic))<BIC_cutoff
    model_index=np.where(slope==True)[0][0]
    components=model_index+1
    print(components)


    #seperate points using probability 0.1 and then use convex hull at each set
    gmm2=models[model_index]
    prob=gmm2.fit(proj).predict_proba(proj).round(3)
    

    index=[]
    hull_index=[]
    index_not_hull=[]
    for i in range(components):
        index.append(np.argwhere((prob[:,i]> probability_cutoff )==True)[:,0])
        hull=ConvexHull(proj[index[i]])
        hull_index_Xmoon=index[i][hull.vertices]
        hull_index.append(hull_index_Xmoon)
        index_not_hull.append(set(index[i]).difference(set(hull_index[i])))
    
    #get the unique index of seeds after eliminating the intersection points 
    vertix_index=[]
    for i in range(components):
        hull=ConvexHull(proj[index[i]])
        hull_index_res=index[i][hull.vertices]
        for j in hull_index_res:
            for k in range(components):
                mark=True
                
                if i==k:
                    continue
                else:
                    if j in index_not_hull[k]:
                        mark=False
                        break
            if mark==True:
                vertix_index.append(j)
    
                
    vertix_index=np.unique(vertix_index)
    np.savetxt('./'+str(coming_cycle)+'/vertix_index.txt',vertix_index,fmt='%d')
    dcd_save=dcd[vertix_index]
    dcd_save.save_amberrst7('./'+str(coming_cycle)+'/rst')
    
    return vertix_index









#read prmtop and inpcrd file of system
prmtop = omma.AmberPrmtopFile(system_name+'./prmtop')
initial_inpcrd = omma.AmberInpcrdFile(system_name+'.inpcrd')

simulation = build_system(prmtop)

#run 10 ns eq simulation
run_short_eq(simulation, initial_inpcrd, 5000000)




for i in range(num_cycle):
    coming_cycle=i+1
    vertix_index=get_proj_and_rst(coming_cycle,lis)
    lis.append(len(vertix_index))
    for j in range(len(vertix_index)):
        simulation_seed=build_system(prmtop)
        seed_inpcrd=omma.AmberInpcrdFile('./'+str(coming_cycle)+'/rst.'+
                                         ("{0:0"+str(len(str(len(vertix_index))))+"d}").format(j+1))
        run_seed_simulation(simulation_seed,seed_inpcrd,50000,j+1,coming_cycle)

            
                
            

    
         
         

         
         
         
         

