static char help[] = "Newton's method to solve _in parallel_ a nonlinear system that resembles an economic production function.\n\n";

#include <petscsnes.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>

typedef struct {
    PetscInt NumberProcesses;   /* Number of MPI Processes in total */
    PetscReal rho;           /* rho parameter */
    PetscReal gamma;
    PetscReal drts;          /* decreasing-returns to scale parameter */
    PetscReal Y;             /* production function output */
    PetscInt n;             /* number of industries */
    Vec beta;         /* technology vector */
    Vec prices;       /* industry good prices */
} AppCtx;

#define DEBUG

#include "prodFunction_parallel.c"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
    /* solution, residual vectors */
    Vec x, residual;

    /* betas */
    Vec betas, xorigin, prices;
    PetscScalar *tmpVec, betaSum, prodSum, tmp;

    /* nonlinear solver context */
    SNES snes;
    
    PetscErrorCode ierr;
    PetscInt NumberIterations, n, nplus1;
    PetscScalar zero = 0;
    PetscInt StartIndex, EndIndex, globalPos;
    PetscScalar guess, *result;
    PetscReal drts, rho, gamma;
    SNESConvergedReason reason;
    AppCtx params;
    double startTimer, endTimer, residualNorm = 0.0;
    PetscMPIInt NumberProcesses, myRank, mySize;

    PetscInitialize(&argc, &argv, (char*) 0, help);
    MPI_Comm_size(PETSC_COMM_WORLD, &NumberProcesses);
    MPI_Comm_rank(PETSC_COMM_WORLD, &myRank);

    n = 20;
    ierr = PetscOptionsGetInt(NULL, NULL, "-m", &n, NULL); CHKERRQ(ierr);
    nplus1 = n + 1;

    /* TODO: set drts and rho from inputs */
    drts = 0.8;
    rho = 2;
    
    // ----------------------------------------
    // Initialize parameters
    // ----------------------------------------
    params.drts = drts;
    params.rho = rho;
    params.n = n;
    params.NumberProcesses = NumberProcesses;

    gamma = (double) ((rand() % 40) + 100) / 100.0;
    /* printf("%i: gamma = %f", myRank, gamma); */

    /* initialize random seed: */
    /* srand(12345678); */
    srand(123 * (myRank + 1));
    /* srand(time(NULL) * (myRank + 1)); */

    PetscPrintf(PETSC_COMM_WORLD, " => betas\n");

    //
    // betas (in parallel)
    //
    ierr = VecCreate(PETSC_COMM_WORLD, &betas); CHKERRQ(ierr);
    ierr = VecSetSizes(betas, PETSC_DECIDE, nplus1); CHKERRQ(ierr);
    ierr = VecSetType(betas, VECMPI); CHKERRQ(ierr);
    ierr = VecSetFromOptions(betas); CHKERRQ(ierr);
    ierr = VecDuplicate(betas, &xorigin); CHKERRQ(ierr);
    ierr = VecDuplicate(betas, &prices); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(betas, &StartIndex, &EndIndex); CHKERRQ(ierr);
    ierr = VecGetLocalSize(betas, &mySize); CHKERRQ(ierr);


#ifdef DEBUG
    printf("%i: Local size: %i\n", myRank, mySize);
    printf("%i: xstart: %i  | xend:  %i\n", myRank, StartIndex, EndIndex);
#endif
    
    for(int i = 0; i < mySize; i++) {
        globalPos = StartIndex + i;
        tmp = (double) (rand() % 10000) + 1.0;
        VecSetValues(betas, 1, &globalPos, &tmp, INSERT_VALUES);
        tmp = (double) (rand() % 1000000) / 10000 + 100.0;
        VecSetValues(xorigin, 1, &globalPos, &tmp, INSERT_VALUES);
        VecSetValues(prices, 1, &globalPos, &tmp, INSERT_VALUES);
    }
    // last element must be zero!
    VecSetValues(betas, 1, &n, &zero, INSERT_VALUES);
    VecSetValues(xorigin, 1, &n, &zero, INSERT_VALUES);
    VecSetValues(prices, 1, &n, &zero, INSERT_VALUES);
    
    ierr = VecAssemblyBegin(betas);
    ierr = VecAssemblyEnd(betas);
    
    ierr = VecAssemblyBegin(xorigin);
    ierr = VecAssemblyEnd(xorigin);

    ierr = VecAssemblyBegin(prices);
    ierr = VecAssemblyEnd(prices);

    VecSum(betas, &betaSum);
    VecScale(betas, 1 / betaSum);

    //
    // Calculate the _inner_ sum of the production function
    // 
    // prices and xorigin hold the same values
    // so we "keep" xorigin for later comparisons
    // and compute the prices based on the xorigin values in prices
    VecPow(prices, rho); // actually: xorigin^pow
    VecPointwiseMult(prices, prices, betas);

    VecSum(prices, &prodSum);

    params.Y = -1 * gamma * pow(prodSum, drts / rho);

    /* Compute prices */
    prodSum = -1 * gamma * pow(prodSum, drts/rho - 1);
    VecCopy(xorigin, prices);
    VecPow(prices, rho - 1);
    VecScale(prices, prodSum);
    VecPointwiseMult(prices, prices, betas);

    params.prices = prices;
    params.beta = betas;

/* #ifdef DEBUG */
/*     VecSum(betas, &betaSum); */
/*     PetscPrintf(PETSC_COMM_WORLD, " => betaSum: %f\n", betaSum); */
/*     PetscPrintf(PETSC_COMM_WORLD, "betas: ------\n"); */
/*     VecView(betas, PETSC_VIEWER_STDOUT_WORLD); */
/*     PetscPrintf(PETSC_COMM_WORLD, "xorigin: ------\n"); */
/*     VecView(xorigin, PETSC_VIEWER_STDOUT_WORLD); */
/*     PetscPrintf(PETSC_COMM_WORLD, "Y: %f\n", params.Y); */
/*     PetscPrintf(PETSC_COMM_WORLD, "prices: ------\n"); */
/*     VecView(prices, PETSC_VIEWER_STDOUT_WORLD); */
/* #endif */

    // ------------------------------
    //
    // Start solving this thing
    //
    // ------------------------------
    startTimer = MPI_Wtime();
    
    //
    // Create nonlinear solver context
    //
    ierr = SNESCreate(PETSC_COMM_WORLD, &snes); CHKERRQ(ierr);
    
    /* Create vectors for solution and nonlinear function */
    /* ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr); */
    /* ierr = VecSetSizes(x, PETSC_DECIDE, nplus1); CHKERRQ(ierr); */
    /* ierr = VecSetType(x, VECMPI); CHKERRQ(ierr); */
    /* ierr = VecSetFromOptions(x); CHKERRQ(ierr); */
    ierr = VecDuplicate(betas, &x); CHKERRQ(ierr);
    ierr = VecDuplicate(x, &residual); CHKERRQ(ierr);

    ierr = VecGetOwnershipRange(x, &StartIndex, &EndIndex); CHKERRQ(ierr);
    ierr = VecGetLocalSize(x, &mySize);
    
    // Set initial guesses / values
    // I use random variables from 100 to 200
    for(int i = StartIndex; i < EndIndex; i++) {
        guess = (double) (rand() % 100) + 100.0;
/* #ifdef DEBUG */
/*         printf("%i: x[%i] = %f\n", myRank, i, guess); */
/* #endif */
        VecSetValues(x, 1, &i, &guess, INSERT_VALUES);
        /* VecSetValues(residual, 1, &i, &guess, INSERT_VALUES); */
    }
    guess = 1.1;
    VecSetValues(x, 1, &n, &guess, INSERT_VALUES);
    params.gamma = guess;

    ierr = VecAssemblyBegin(x);
    ierr = VecAssemblyEnd(x);

    ierr = VecAssemblyBegin(residual);
    ierr = VecAssemblyEnd(residual);

/* #ifdef DEBUG */
/*     PetscPrintf(PETSC_COMM_WORLD, "VecView: ------\n"); */
/*     VecView(x, PETSC_VIEWER_STDOUT_WORLD); */
/*     VecView(residual, PETSC_VIEWER_STDOUT_WORLD); */
/* #endif */

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Customize nonlinear solver; set runtime options
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscPrintf(PETSC_COMM_WORLD, " -----------------------\n");
    PetscPrintf(PETSC_COMM_WORLD, " => Starting solving ...\n");
    ierr = SNESSetFunction(snes, residual, ProdFunction, &params);
    CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

    ierr = SNESSolve(snes, NULL, x); CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(snes, &NumberIterations); CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes, &reason); CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, " -----------------------\n");
    PetscPrintf(PETSC_COMM_WORLD, " => Finished solving ...\n");
    
    endTimer = MPI_Wtime();
    
/*     ierr  = VecGetArray(x, &result); CHKERRQ(ierr); */
/*     for(int i = 0; i < n; i++) { */
/* #ifdef DEBUG */
/*         ierr = PetscPrintf(PETSC_COMM_WORLD, */
/*                            "%i: res = %f \torig = %f \tdiff = %f\n", */
/*                            i, result[i], xvec[i], result[i] - xvec[i]); */
/*         CHKERRQ(ierr); */
/* #endif */
/*         residualNorm += abs(result[i] - xvec[i]); */
/*     } */
/* #ifdef DEBUG */
/*     ierr = PetscPrintf(PETSC_COMM_WORLD, "gamma = %f | %f\n", result[n], */
/*                        gamma); */
/*     CHKERRQ(ierr); */
/* #endif */
    
/*     ierr = VecRestoreArray(x, &result); CHKERRQ(ierr); */
/*     ierr = PetscPrintf(PETSC_COMM_WORLD, */
/*                        "PETSc, %s, %f, %i, %D, %i, %f", */
/*                        reason>0 ? "CONVERGED" : (char*) SNESConvergedReasons[reason], */
/*                        residualNorm, */
/*                        n, NumberIterations, NumberProcesses, */
/*                        endTimer - startTimer); */
/*     CHKERRQ(ierr); */

/*     for(int i = 1; i < argc; i++) { */
/*         PetscPrintf(PETSC_COMM_WORLD, ", %s", argv[i]); */
/*     } */
/*     PetscPrintf(PETSC_COMM_WORLD, "\n"); */
    
    
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&residual); CHKERRQ(ierr);
    ierr = SNESDestroy(&snes); CHKERRQ(ierr);
    
    ierr = PetscFinalize();
    return 0;
}
