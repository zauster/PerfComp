#
# Makefile for the CGE Toyproblems
#

#CPPFLAGS        = -I/PETSc3/geodynamics/PetscSimulationsViewers/src
CFLAGS           =
FFLAGS           =
CPPFLAGS         =
FPPFLAGS         =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

nls_eco_problem: nls_eco_problem.o
	-${CLINKER} -o nls_eco_problem nls_eco_problem.o ${PETSC_SNES_LIB}
	${RM} nls_eco_problem.o

run:
	-@${MPIEXEC} -n 1 ./nls_eco_problem -snes_mf -pc_type none -snes_type newtonls -snes_max_it 1000

run_debug:
	-@${MPIEXEC} -n 1 ./nls_eco_problem -snes_mf -pc_type none -snes_type newtonls -snes_max_it 1000 -snes_monitor -snes_monitor_short



# --------------------
# old stuff
# --------------------
# nls_eco_problem_2x: nls_eco_problem_2x.o  chkopts
# 	-${CLINKER} -o nls_eco_problem_2x nls_eco_problem_2x.o ${PETSC_SNES_LIB}
# 	${RM} nls_eco_problem_2x.o
# run_2x:
# 	-@${MPIEXEC} -n 1 ./nls_eco_problem_2x -snes_type newtonls -snes_max_it 1000
# run_exp_2x:
# 	-@${MPIEXEC} -n 1 ./nls_eco_problem_2x -snes_mf -pc_type none -snes_type newtonls -snes_max_it 1000 -snes_view -snes_monitor_short
