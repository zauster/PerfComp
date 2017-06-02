#!/bin/bash
#
BIN="nls_eco_problem"
OPTIONS1="-snes_mf -pc_type none"
OPTIONS2="-snes_max_it 10000"

for iter in 1 #2 3 4 5
do
    for DIM in 100 500 1000 #1500 2000 2500 3000
    do
        mpiexec -n 1 ./$BIN $OPTIONS1 -snes_type newtonls $OPTIONS2 -m $DIM
        mpiexec -n 1 ./$BIN $OPTIONS1 -snes_type newtontr $OPTIONS2 -m $DIM
        mpiexec -n 1 ./$BIN $OPTIONS1 -snes_type anderson $OPTIONS2 -m $DIM
        mpiexec -n 1 ./$BIN $OPTIONS1 -snes_type qn $OPTIONS2 -m $DIM
    done 
done
