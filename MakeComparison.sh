#!/bin/bash
#
PETSC_SERIAL="petsc/nls_eco_problem"
PETSC_PARALLEL="petsc/nls_eco_problem_parallel"
OPTIONS1="-snes_mf -pc_type none"
OPTIONS2="-snes_max_it 10000"
OPTIONS="$OPTIONS1 $OPTIONS2"
NPROC=1

for iter in 1 #2 3 #4 5
do
    for DIM in 100 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
    do
        # Serial PETSc
        mpiexec -n $NPROC ./$PETSC_SERIAL $OPTIONS -snes_type newtonls -m $DIM
        mpiexec -n $NPROC ./$PETSC_SERIAL $OPTIONS -snes_type newtontr -m $DIM
        # mpiexec -n $NPROC ./$PETSC_SERIAL $OPTIONS -snes_type anderson -m $DIM
        # mpiexec -n $NPROC ./$PETSC_SERIAL $OPTIONS -snes_type qn -m $DIM

        # Parallel PETSc
        mpiexec -n 2 ./$PETSC_PARALLEL $OPTIONS -snes_type newtonls -m $DIM
        mpiexec -n 2 ./$PETSC_PARALLEL $OPTIONS -snes_type newtontr -m $DIM
        # mpiexec -n 2 ./$PETSC_PARALLEL $OPTIONS -snes_type anderson -m $DIM
        # mpiexec -n 2 ./$PETSC_PARALLEL $OPTIONS -snes_type qn -m $DIM
        
        # Rscript ./R/nls_eco_problem.R $DIM $NPROC # takes way too long
    done 
done
