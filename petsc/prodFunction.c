
#include <petscsnes.h>

#undef __FUNCT__
#define __FUNCT__ "ProdFunction"

PetscErrorCode ProdFunction(SNES snes, Vec x, Vec f, void *ctx)
{
    AppCtx *params = (AppCtx*)ctx;
    PetscErrorCode ierr;
    PetscScalar *ff;
    const PetscScalar *xx;
    PetscReal rho, drts, Y;
    PetscInt n;
    PetscReal *beta, *prices;
    PetscReal prodSum;

    /* get parameters from the user context */
    rho = params->rho;
    drts = params->drts;
    Y = params->Y;
    n = params->n;
    beta = params->beta;
    prices = params->prices;

    /* Get pointers to vector data. */
    ierr = VecGetArrayRead(x,&xx); CHKERRQ(ierr);
    ierr = VecGetArray(f,&ff); CHKERRQ(ierr);

    /* PetscPrintf(PETSC_COMM_WORLD, "xx: ------\n"); */
    /* ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr); */

    //
    // TODO: replace with vector operations
    //

    /* Calculate the _inner_ sum of the production function */
    prodSum = 0;
    for(int i = 0; i < n; i++) {
        prodSum += beta[i] * pow(xx[i], rho);
    }
  
    /* Compute function */
    ff[n] = Y + xx[n] * pow(prodSum, drts / rho);
    /* PetscReal tmpgamma = Y + xx[n] * pow(prodSum, drts / rho); */
    /* printf("   --> gamma = %f\n", tmpgamma); */
    
    prodSum = xx[n] * pow(prodSum, drts/rho - 1);
    for(int i = 0; i < n; i++) {
        ff[i] = prices[i] + beta[i] * xx[i] * prodSum;    
    }

    /* PetscPrintf(PETSC_COMM_WORLD, " ----- ff ------\n"); */
    /* VecView(f, PETSC_VIEWER_STDOUT_WORLD); */
    

    /* Restore vectors */
    ierr = VecRestoreArrayRead(x,&xx); CHKERRQ(ierr);
    ierr = VecRestoreArray(f,&ff); CHKERRQ(ierr);
    return 0;
}
