#!/bin/bash
#
PETSC_BIN="petsc/nls_eco_problem"
OPTIONS1="-snes_mf -pc_type none"
OPTIONS2="-snes_max_it 10000"
OPTIONS="$OPTIONS1 $OPTIONS2"
NPROC=1

for iter in 1 2 3 4 5
do
    for DIM in 100 500 1000 1500 2000 2500 3000
    do
        mpiexec -n $NPROC ./$PETSC_BIN $OPTIONS -snes_type newtonls -m $DIM
        mpiexec -n $NPROC ./$PETSC_BIN $OPTIONS -snes_type newtontr -m $DIM
        mpiexec -n $NPROC ./$PETSC_BIN $OPTIONS -snes_type anderson -m $DIM
        mpiexec -n $NPROC ./$PETSC_BIN $OPTIONS -snes_type qn -m $DIM
        Rscript ./R/nls_eco_problem.R $DIM $NPROC
    done 
done