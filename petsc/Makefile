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


#
# Serial version
#
nls_eco_problem: nls_eco_problem.o
	-${CLINKER} -o nls_eco_problem nls_eco_problem.o ${PETSC_SNES_LIB}
	${RM} nls_eco_problem.o

run:
	-@${MPIEXEC} -n 1 ./nls_eco_problem -snes_mf -pc_type none -snes_type newtonls -snes_max_it 1000 -m 40

run_debug:
	-@${MPIEXEC} -n 1 ./nls_eco_problem -snes_mf -pc_type none -snes_type newtonls -snes_max_it 1000 -snes_monitor -snes_monitor_short


#
# Parallel version
#
nls_eco_problem_parallel: nls_eco_problem_parallel.o
	-${CLINKER} -o nls_eco_problem_parallel nls_eco_problem_parallel.o ${PETSC_SNES_LIB}
	${RM} nls_eco_problem_parallel.o

runp:
	-@${MPIEXEC} -n 2 ./nls_eco_problem_parallel -snes_mf -pc_type none -snes_max_it 1000

runp_debug:
	-@${MPIEXEC} -n 2 ./nls_eco_problem_parallel -snes_mf -pc_type none -snes_type newtonls -snes_max_it 1000 -snes_monitor -snes_monitor_short -m 40

# newtonls
# newtontr
# nrichardson
# ncg
# ngmres
# qn
# composite
# nasm
# fas
# ngs
# anderson
# shell

test: test_main.o
	-${CLINKER} -o test_main test_main.o ${PETSC_SNES_LIB}
	${RM} test_main.o

runtest:
	-@${MPIEXEC} -n 2 ./test_main -snes_mf -pc_type none
