static char help[] = "Newton's method to solve a nonlinear system that resembles an economic production function.\n\n";

#include <petscsnes.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>

typedef struct {
    PetscReal rho;           /* rho parameter */
    PetscReal drts;          /* decreasing-returns to scale parameter */
    PetscReal Y;             /* production function output */
    PetscInt n;             /* number of industries */
    PetscReal *beta;         /* technology vector */
    PetscReal *prices;       /* industry good prices */
} AppCtx;

#define DEBUG

#include "prodFunction.c"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
    /* nonlinear solver context */
    SNES snes;
    
    /* solution, residual vectors */
    Vec x, r;
    
    PetscErrorCode ierr;
    PetscInt NumberIterations, n;
    PetscScalar *guess, *result;
    PetscReal drts, rho, gamma;
    SNESConvergedReason reason;
    AppCtx params;
    double startTimer, endTimer;
    PetscMPIInt NumberProcesses;

    PetscInitialize(&argc, &argv, (char*) 0, help);
    MPI_Comm_size(PETSC_COMM_WORLD, &NumberProcesses);
    
    /* TODO: set n, drts and rho from inputs */
    n = 20;
    ierr = PetscOptionsGetInt(NULL, NULL, "-m", &n, NULL); CHKERRQ(ierr);
    drts = 0.8;
    rho = 2;
    
    double betas[n], xvec[n], prices[n];


    // ----------------------------------------
    // Initialize parameters
    // ----------------------------------------
    params.drts = drts;
    params.rho = rho;
    params.n = n;

    /* initialize random seed: */
    /* srand(1234567); // this seed may lead to errors */
    srand(12345678);
    /* srand(time(NULL)); */

    /* generate betas between 0 and 10000 and scale them afterwards: */
    /* generate xvec between 100 and 200: */
    double betaSum = 0;
    for(int i = 0; i < n; i++) {
        betas[i] = (double) (rand() % 10000) + 1.0;
        xvec[i] = (double) (rand() % 1000000) / 10000 + 100.0;
        betaSum += betas[i];
    }
    /* generate gamma between 1 and 1.4: */
    gamma = (double) ((rand() % 40) + 100) / 100.0;
    /* PetscPrintf(PETSC_COMM_WORLD,"gamma = %f\n", gamma); */

    /* scale beta to \sum_i beta_i = 1 */
    for(int i = 0; i < n; i++) {
        betas[i] = betas[i] / betaSum;
    }
    params.beta = betas;

    /* Calculate the _inner_ sum of the production function */
    double prodSum = 0;
    for(int i = 0; i < n; i++) {
        prodSum += betas[i] * pow(xvec[i], rho);
    }
    params.Y = -1 * gamma * pow(prodSum, drts / rho);

    /* Compute prices */
    prodSum = gamma * pow(prodSum, drts/rho - 1);
    for(int i = 0; i < n; i++) {
        prices[i] = -1 * betas[i] * pow(xvec[i], rho - 1) * prodSum;
    }
    params.prices = prices;

#ifdef DEBUG
    for(int i = 0; i < n; i++) {
        PetscPrintf(PETSC_COMM_WORLD,"%i: x = %f \tbeta = %f \tprice = %f\n", i, xvec[i], betas[i], prices[i]);
    }
#endif
    

    startTimer = MPI_Wtime();
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create nonlinear solver context
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = SNESCreate(PETSC_COMM_WORLD, &snes); CHKERRQ(ierr);

    /* Create vectors for solution and nonlinear function */
    ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, n + 1); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);
    ierr = VecDuplicate(x, &r); CHKERRQ(ierr);

    //
    // Evaluate initial guess; then solve nonlinear system
    //
    srand(time(NULL));
    ierr  = VecGetArray(x,&guess); CHKERRQ(ierr);
    for(int i = 0; i < n; i++) {
        /* guess[i] = 110.0; */
        guess[i] = (double) (rand() % 100) + 100.0;
#ifdef DEBUG
        PetscPrintf(PETSC_COMM_WORLD, "x[%i] = %f\n", i, guess[i]);
#endif
    }
    guess[n] = 1.1;
    ierr  = VecRestoreArray(x, &guess); CHKERRQ(ierr);

    /* Create Jacobian matrix data structure */
    /* ierr = MatCreate(PETSC_COMM_WORLD,&J); CHKERRQ(ierr); */
    /* ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,2,2); CHKERRQ(ierr); */
    /* ierr = MatSetFromOptions(J); CHKERRQ(ierr); */
    /* ierr = MatSetUp(J); CHKERRQ(ierr); */

    /* Set function evaluation routine and vector. */
    ierr = SNESSetFunction(snes, r, ProdFunction, &params); CHKERRQ(ierr);

    /* Set Jacobian matrix data structure and Jacobian evaluation routine */
    /* ierr = SNESSetJacobian(snes,J,J,FormJacobian1,NULL); CHKERRQ(ierr); */

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Customize nonlinear solver; set runtime options
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    printf(" => Start solving ...\n");
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);
    ierr = SNESSolve(snes, NULL, x); CHKERRQ(ierr);
  
    /* ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr); */
    ierr = SNESGetIterationNumber(snes, &NumberIterations); CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes, &reason); CHKERRQ(ierr);

    endTimer = MPI_Wtime();
    printf(" => Finished!\n");
    
    double residualNorm = 0.0;
    ierr  = VecGetArray(x, &result); CHKERRQ(ierr);
    for(int i = 0; i < n; i++) {
#ifdef DEBUG
        ierr = PetscPrintf(PETSC_COMM_WORLD,
                           "%i: res = %f \torig = %f \tdiff = %f\n",
                           i, result[i], xvec[i], result[i] - xvec[i]);
        CHKERRQ(ierr);
#endif
        residualNorm += abs(result[i] - xvec[i]);
    }
#ifdef DEBUG
    ierr = PetscPrintf(PETSC_COMM_WORLD, "gamma = %f | %f\n",
                       result[n], gamma);
    CHKERRQ(ierr);
#endif
    
    ierr  = VecRestoreArray(x,&result); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "PETSc, serial, %s, %f, %i, %D, %i, %f",
                       reason>0 ? "CONVERGED" : (char*) SNESConvergedReasons[reason],
                       residualNorm,
                       n, NumberIterations, NumberProcesses,
                       endTimer - startTimer);
    CHKERRQ(ierr);

    for(int i = 1; i < argc; i++) {
        printf(", %s", argv[i]);
    }
    printf("\n");

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&r); CHKERRQ(ierr);
    /* ierr = MatDestroy(&J); CHKERRQ(ierr); */
    ierr = SNESDestroy(&snes); CHKERRQ(ierr);
    
    ierr = PetscFinalize();
    return 0;
}
