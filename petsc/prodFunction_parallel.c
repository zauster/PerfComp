
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
    PetscReal rho, drts, Y, prodSum, temp_gamma, calc_gamma;
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
    // temp_gamma = 11.0;

    PetscInt xStartIndex, xEndIndex;
    ierr = VecGetLocalSize(x, &mySize);

    //
    // get gamma out of the x-vector and spread it out
    //
    Vec gammaSend;
    ierr = VecCreate(PETSC_COMM_WORLD, &gammaSend); CHKERRQ(ierr);
    ierr = VecSetSizes(gammaSend, 1, PETSC_DECIDE); CHKERRQ(ierr);
    ierr = VecSetFromOptions(gammaSend); CHKERRQ(ierr);
    Vec gammaReceive;
    VecScatter scatter_ctx;
    
    /* Get pointers to vector data. Read from x, write to f */
    ierr = VecGetArrayRead(x, &xx); CHKERRQ(ierr);
    if(myRank == (NumberProcesses - 1)) {
        temp_gamma = xx[mySize - 1];
    }
    // printf("%i: tempgamma_before = %f\n", myRank, temp_gamma);
    /* Restore vectors */
    ierr = VecRestoreArrayRead(x, &xx); CHKERRQ(ierr);

    ierr = VecSetValue(gammaSend, 0, temp_gamma, INSERT_VALUES);
    ierr = VecSetValue(gammaSend, 1, temp_gamma, INSERT_VALUES);
    CHKERRQ(ierr);
    ierr = VecAssemblyBegin(gammaSend); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(gammaSend); CHKERRQ(ierr);

    // ierr = VecGetArray(gammaSend, &ggamma); CHKERRQ(ierr);
    // ierr = VecGetOwnershipRange(gammaSend, &xStartIndex, &xEndIndex);
    // CHKERRQ(ierr);
    // printf("%i: %i - %i | gammaSend_0: %f\n",
    //        myRank, xStartIndex, xEndIndex, ggamma[0]);
    // // printf("%i: %i - %i | gammaSend_1: %f\n",
    // //        myRank, xStartIndex, xEndIndex, ggamma[1]);
    // ierr = VecRestoreArray(gammaSend, &ggamma); CHKERRQ(ierr);
    
    ierr = VecScatterCreateToAll(gammaSend, &scatter_ctx, &gammaReceive);
    CHKERRQ(ierr);
    ierr = VecScatterBegin(scatter_ctx, gammaSend, gammaReceive,
                           INSERT_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter_ctx, gammaSend, gammaReceive,
                         INSERT_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scatter_ctx); CHKERRQ(ierr);


    ierr = VecGetArray(gammaReceive, &ggamma); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(gammaReceive, &xStartIndex, &xEndIndex);
    CHKERRQ(ierr);
    // printf("%i: %i - %i | gammaReceive_0: %f\n",
    //        myRank, xStartIndex, xEndIndex, ggamma[0]);
    // printf("%i: %i - %i | gammaReceive_1: %f\n",
    //        myRank, xStartIndex, xEndIndex, ggamma[1]);

    temp_gamma = ggamma[0];
    // printf("%i: tempgamma_later = %f\n", myRank, temp_gamma);

    ierr = VecRestoreArray(gammaReceive, &ggamma); CHKERRQ(ierr);
    
    
    
    //
    // calc inner sum of the production function
    // must be efficient

    // PetscPrintf(PETSC_COMM_WORLD, "xx: ------\n");
    // ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    
    ierr = VecCopy(x, f); CHKERRQ(ierr); // f = x
    ierr = VecPow(f, rho); CHKERRQ(ierr); // f^rho
    ierr = VecPointwiseMult(f, f, params->beta); CHKERRQ(ierr); // f[i] * beta[i]
    ierr = VecSum(f, &prodSum); CHKERRQ(ierr);// sum(f)

    calc_gamma = Y + temp_gamma * pow(prodSum, drts/rho);

    prodSum = temp_gamma * pow(prodSum, drts/rho - 1);

    ierr = VecCopy(x, f); CHKERRQ(ierr); // f = x
    ierr = VecScale(f, prodSum); CHKERRQ(ierr); // f * prodSum
    ierr = VecPointwiseMult(f, f, params->beta); CHKERRQ(ierr); // f[i] * beta[i]
    ierr = VecAYPX(f, 1, params->prices); CHKERRQ(ierr); // f[i] + prices[i]
    
    ierr = VecGetArray(f, &ff); CHKERRQ(ierr);
    if(myRank == (NumberProcesses - 1)) {
        ff[mySize - 1] = calc_gamma;
    }
    ierr = VecRestoreArray(f, &ff); CHKERRQ(ierr);

    // PetscPrintf(PETSC_COMM_WORLD, " ----- ff after gamma ------\n");
    // VecView(f, PETSC_VIEWER_STDOUT_WORLD);
    
    return 0;
}
