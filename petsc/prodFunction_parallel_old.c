
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

    Vec gammaSend;
    Vec gammaReceive;
    VecScatter scatter_ctx;
    VecCreate(PETSC_COMM_WORLD, &gammaSend);
    VecSetSizes(gammaSend, 1, PETSC_DECIDE);
    VecSetFromOptions(gammaSend);

    MPI_Comm_rank(PETSC_COMM_WORLD, &myRank);

    /* get parameters from the user context */
    NumberProcesses = params->NumberProcesses;
    rho = params->rho;
    drts = params->drts;
    Y = params->Y;
    n = params->n;
    /* temp_gamma = params->gamma; */

    PetscInt xStartIndex, xEndIndex;

    /* Get pointers to vector data. Read from x, write to f */
    ierr = VecGetArrayRead(x, &xx); CHKERRQ(ierr);
    if(myRank == (NumberProcesses - 1)) {

        VecGetOwnershipRange(gammaSend, &xStartIndex, &xEndIndex);
        ierr = VecSetValues(gammaSend, 1, &xStartIndex, &xx[mySize-1],
                            INSERT_VALUES);
        CHKERRQ(ierr);
        
        // check if gammaSend is correctly set
        /* printf("====== %i: %i - %i -> %f =====\n", */
        /*        myRank, xStartIndex, xEndIndex, xx[mySize-1]); */
        /* ierr = VecGetArray(gammaSend, &ggamma); */
        /* VecGetOwnershipRange(gammaSend, &xStartIndex, &xEndIndex); */
        /* printf("%i: gammaSend: %i - %i\n", myRank, xStartIndex, xEndIndex); */
        /* printf("%i: gammaSend: %f\n", myRank, ggamma[0]); */
        /* ierr = VecRestoreArray(gammaSend, &ggamma); CHKERRQ(ierr); */
        
    }
    /* Restore vectors */
    ierr = VecRestoreArrayRead(x, &xx); CHKERRQ(ierr);

    //
    // do the scattering of gamma
    //
    
    // TODO:
    // TODO: check how to correctly send gamma to all procs
    // TODO:

    ierr = VecScatterCreateToAll(gammaSend, &scatter_ctx, &gammaReceive);
    CHKERRQ(ierr);
    ierr = VecScatterBegin(scatter_ctx, gammaSend, gammaReceive,
                           INSERT_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter_ctx, gammaSend, gammaReceive,
                           INSERT_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scatter_ctx); CHKERRQ(ierr);

    ierr = VecGetOwnershipRange(gammaReceive, &xStartIndex, &xEndIndex);
    CHKERRQ(ierr);
    ierr = VecGetLocalSize(gammaReceive, &mySize);

    printf("%i: %i from %i to %i\n", myRank, mySize,
           xStartIndex, xEndIndex);

    ierr = VecGetArray(gammaReceive, &ggamma); CHKERRQ(ierr);
    for(int i = 0; i < mySize; i++) {
        printf("%i: %i from %i to %i | gamma = %f\n", myRank, mySize,
               xStartIndex, xEndIndex, ggamma[i]);
    }
    ierr = VecRestoreArray(gammaReceive, &ggamma); CHKERRQ(ierr);
    
    ierr = VecGetLocalSize(x, &mySize); CHKERRQ(ierr);

/* #ifdef DEBUG */
/*     /\* Calculate the _inner_ sum of the production function *\/ */
/*     PetscInt xStartIndex, xEndIndex, fStartIndex, fEndIndex; */
/*     VecGetOwnershipRange(x, &xStartIndex, &xEndIndex); */
/*     VecGetOwnershipRange(f, &fStartIndex, &fEndIndex); */
/*     xEndIndex = n < xEndIndex? n : xEndIndex; */
/*     fEndIndex = n < fEndIndex? n : fEndIndex; */
/*     VecGetLocalSize(x, &mySize); */
/*     PetscScalar betasum; */
/*     VecSum(params->beta, &betasum); */
/*     printf("%i: Local size: %i | betasum: %f\n", myRank, mySize, betasum); */
/*     printf("%i: xstart: %i  | xend:  %i\n", myRank, xStartIndex, xEndIndex); */
/*     printf("%i: fstart: %i  | fend:  %i\n", myRank, fStartIndex, fEndIndex); */
/* #endif */

    //
    // calc inner sum of the production function
    // must be efficient

    /* PetscPrintf(PETSC_COMM_WORLD, "xx: ------\n"); */
    /* ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr); */
    
    ierr = VecCopy(x, f); CHKERRQ(ierr); // f = x
    ierr = VecPow(f, rho); CHKERRQ(ierr); // f^rho
    ierr = VecPointwiseMult(f, f, params->beta); CHKERRQ(ierr); // f[i] * beta[i]
    ierr = VecSum(f, &prodSum); CHKERRQ(ierr);// sum(f)

    temp_gamma = Y + temp_gamma * pow(prodSum, drts/rho);

#ifdef DEBUG
    printf("%i: prodSum = %f | tempgamma = %f | gamma = %f\n", myRank, prodSum, temp_gamma, params->gamma);
#endif

    prodSum = temp_gamma * pow(prodSum, drts/rho - 1);

    ierr = VecCopy(x, f); CHKERRQ(ierr); // f = x
    ierr = VecScale(f, prodSum); CHKERRQ(ierr); // f * prodSum
    ierr = VecPointwiseMult(f, f, params->beta); CHKERRQ(ierr); // f[i] * beta[i]
    ierr = VecAYPX(f, 1, params->prices); CHKERRQ(ierr); // f[i] + prices[i]


    /* PetscPrintf(PETSC_COMM_WORLD, " ----- ff before gamma ------\n"); */
    /* VecView(f, PETSC_VIEWER_STDOUT_WORLD); */
    
    params->gamma = temp_gamma;
    ierr = VecGetArray(f, &ff); CHKERRQ(ierr);
    if(myRank == (NumberProcesses - 1)) {
        ff[mySize - 1] = temp_gamma;
        printf(" -> end: %i | temp_gamma = %f | saved = %f\n", mySize, temp_gamma, ff[mySize - 1]);
    }
    ierr = VecRestoreArray(f, &ff); CHKERRQ(ierr);

    /* PetscPrintf(PETSC_COMM_WORLD, " ----- ff after gamma ------\n"); */
    /* VecView(f, PETSC_VIEWER_STDOUT_WORLD); */
    
    return 0;
}



/*     /\* the last process iterates only until mySize - 1 *\/ */
/*     if((myRank + 1) == NumberProcesses) { */
/*         mySize = mySize - 1; */
/*     } */

    /* prodSum = 0; */
    /* for(int i = 0; i < mySize; i++) { */
        
    /*     // TODO: get the right betas! */
    /*     printf("Proc%i: %i: xx = %f  |  beta = %f\n", */
    /*            myRank, i, xx[i], localbeta[i]); */
    /*     /\* prodSum += beta[i] * pow(xx[i], rho); *\/ */
    /*     /\* ff[i] = beta[i] * pow(xx[i], rho); *\/ */
    /*     /\* printf("%i: ff[%i]: %f\n", myRank, i, ff[i]); *\/ */
    /* } */
    /* printf("%i: prodsum: %f\n", myRank, prodSum); */

    /* VecSum(ff, &ffprodSum); */
    /* printf("%i: ffprodsum: %f\n", myRank, ffprodSum); */
  
    /* /\* Compute function *\/ */
    /* ff[n] = Y + xx[n] * pow(prodSum, drts / rho); */
    /* prodSum = xx[n] * pow(prodSum, drts/rho - 1); */
    /* for(int i = 0; i < n; i++) { */
    /*     ff[i] = prices[i] + beta[i] * xx[i] * prodSum;     */
    /* } */
