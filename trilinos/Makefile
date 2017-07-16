# CMAKE File for "MyApp" application building against an installed Trilinos

#This file is an adaptation of the CMakeLists.txt file that was
#converted from the buildAgainstTrilinos example. This Makefile was
#designed to be used in a flat directory structure. If you would like
#to run this example you will need put this file and src_file.cpp,
#src_file.hpp, main_file.cpp from buildAgainstTrilinos into a new
#directory. You will then need to set the environment variable
#MYAPP_TRILINOS_DIR to point to your base installation of
#Trilinos. Note that this example assumes that the installation of
#Trilinos that you point to has Epetra enabled.

MYAPP_TRILINOS_DIR=/usr

# Get Trilinos as one entity
include $(MYAPP_TRILINOS_DIR)/include/Makefile.export.Trilinos

# Make sure to use same compilers and flags as Trilinos
CXX=$(Trilinos_CXX_COMPILER)
CC=$(Trilinos_C_COMPILER)
FORT=$(Trilinos_Fortran_COMPILER)

CXX_FLAGS=$(Trilinos_CXX_COMPILER_FLAGS) $(USER_CXX_FLAGS)
C_FLAGS=$(Trilinos_C_COMPILER_FLAGS) $(USERC_FLAGS)
FORT_FLAGS=$(Trilinos_Fortran_COMPILER_FLAGS) $(USER_FORT_FLAGS)

INCLUDE_DIRS=$(Trilinos_INCLUDE_DIRS) $(Trilinos_TPL_INCLUDE_DIRS)
LIBRARY_DIRS=$(Trilinos_LIBRARY_DIRS) $(Trilinos_TPL_LIBRARY_DIRS)
LIBRARIES=$(Trilinos_LIBRARIES) $(Trilinos_TPL_LIBRARIES)

LINK_FLAGS=$(Trilinos_EXTRA_LD_FLAGS)

#just assuming that epetra is turned on.
DEFINES=-DMYAPP_EPETRA


# default: print_info NOXNewton1
default: test

# Echo trilinos build info just for fun
print_info:
	@echo "\nFound Trilinos!  Here are the details: "
	@echo "   Trilinos_VERSION = $(Trilinos_VERSION)"
	@echo "   Trilinos_PACKAGE_LIST = $(Trilinos_PACKAGE_LIST)"
	@echo "   Trilinos_LIBRARIES = $(Trilinos_LIBRARIES)"
	@echo "   Trilinos_INCLUDE_DIRS = $(Trilinos_INCLUDE_DIRS)"
	@echo "   Trilinos_LIBRARY_DIRS = $(Trilinos_LIBRARY_DIRS)"
	@echo "   Trilinos_TPL_LIST = $(Trilinos_TPL_LIST)"
	@echo "   Trilinos_TPL_INCLUDE_DIRS = $(Trilinos_TPL_INCLUDE_DIRS)"
	@echo "   Trilinos_TPL_LIBRARIES = $(Trilinos_TPL_LIBRARIES)"
	@echo "   Trilinos_TPL_LIBRARY_DIRS = $(Trilinos_TPL_LIBRARY_DIRS)"
	@echo "   Trilinos_BUILD_SHARED_LIBS = $(Trilinos_BUILD_SHARED_LIBS)"
	@echo "End of Trilinos details\n"


# run the given test
test: nls_eco_problem
	mpirun -n 2 nls_eco_problem

# build the application
nls_eco_problem: nls_eco_problem.cpp
	@$(CXX) $(CXX_FLAGS) nls_eco_problem.cpp -o nls_eco_problem $(LINK_FLAGS) $(INCLUDE_DIRS) $(DEFINES) $(LIBRARY_DIRS) $(LIBRARIES)
	@echo "  => Successfully compiled!"


# # run the given test
# test:
# 	./NOXNewton1

# # build the application
# NOXNewton1: NOXNewton1.cpp
# 	$(CXX) $(CXX_FLAGS) NOXNewton1.cpp -o NOXNewton1 $(LINK_FLAGS) $(INCLUDE_DIRS) $(DEFINES) $(LIBRARY_DIRS) $(LIBRARIES)

# NOXNewton1.a: NOXNewton1.o
# 	$(Trilinos_AR) cr NOXNewton1.a NOXNewton1.o

# NOXNewton1.o:
# 	$(CXX) -c $(CXX_FLAGS) $(INCLUDE_DIRS) $(DEFINES) NOXNewton1.cpp

.PHONY: clean
clean:
	rm -f *.o *.a
