
#include <petscsnes.h>

#undef __FUNCT__
#define __FUNCT__ "ProdFunction"

PetscErrorCode ProdFunction(SNES snes, Vec x, Vec f, void *ctx)
{
    AppCtx *params = (AppCtx*)ctx;
    PetscErrorCode ierr;
    
    PetscScalar *ff;
    const PetscScalar *xx;
    PetscScalar *ggamma;
    PetscReal rho, drts, Y, prodSum, temp_gamma;
    PetscInt n;
    PetscMPIInt myRank;
    PetscInt mySize, NumberProcesses;

    MPI_Comm_rank(PETSC_COMM_WORLD, &myRank);

    /* get parameters from the user context */
    NumberProcesses = params->NumberProcesses;
    rho = params->rho;
    drts = params->drts;
    Y = params->Y;
    n = params->n;
    temp_gamma = params->gamma;

    PetscInt xStartIndex, xEndIndex;

    ierr = VecGetLocalSize(x, &mySize);

    //
    // calc inner sum of the production function
    // must be efficient

    PetscPrintf(PETSC_COMM_WORLD, "xx: ------\n");
    ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    
    ierr = VecCopy(x, f); CHKERRQ(ierr); // f = x
    ierr = VecPow(f, rho); CHKERRQ(ierr); // f^rho
    ierr = VecPointwiseMult(f, f, params->beta); CHKERRQ(ierr); // f[i] * beta[i]
    ierr = VecSum(f, &prodSum); CHKERRQ(ierr);// sum(f)

    temp_gamma = Y + temp_gamma * pow(prodSum, drts/rho);

    prodSum = temp_gamma * pow(prodSum, drts/rho - 1);

    ierr = VecCopy(x, f); CHKERRQ(ierr); // f = x
    ierr = VecScale(f, prodSum); CHKERRQ(ierr); // f * prodSum
    ierr = VecPointwiseMult(f, f, params->beta); CHKERRQ(ierr); // f[i] * beta[i]
    ierr = VecAYPX(f, 1, params->prices); CHKERRQ(ierr); // f[i] + prices[i]

    /* PetscPrintf(PETSC_COMM_WORLD, " ----- ff before gamma ------\n"); */
    /* VecView(f, PETSC_VIEWER_STDOUT_WORLD); */
    
    /* params->gamma = temp_gamma; */
    ierr = VecGetArray(f, &ff); CHKERRQ(ierr);
    if(myRank == (NumberProcesses - 1)) {
        ff[mySize - 1] = 1.2;
        /* printf(" -> end: %i | temp_gamma = %f | saved = %f\n", mySize, temp_gamma, ff[mySize - 1]); */
    }
    ierr = VecRestoreArray(f, &ff); CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, " ----- ff after gamma ------\n");
    VecView(f, PETSC_VIEWER_STDOUT_WORLD);
    
    return 0;
}
