

/* /\* ------------------------------------------------------------------- *\/ */
/* #undef __FUNCT__ */
/* #define __FUNCT__ "FormJacobian1" */
/* /\* */
/*    FormJacobian1 - Evaluates Jacobian matrix. */

/*    Input Parameters: */
/* .  snes - the SNES context */
/* .  x - input vector */
/* .  dummy - optional user-defined context (not used here) */

/*    Output Parameters: */
/* .  jac - Jacobian matrix */
/* .  B - optionally different preconditioning matrix */
/* .  flag - flag indicating matrix structure */
/* *\/ */
/* PetscErrorCode FormJacobian1(SNES snes,Vec x,Mat jac,Mat B,void *dummy) */
/* { */
/*   const PetscScalar *xx; */
/*   PetscScalar       A[4]; */
/*   PetscErrorCode    ierr; */
/*   PetscInt          idx[2] = {0,1}; */

/*   /\* */
/*      Get pointer to vector data */
/*   *\/ */
/*   ierr = VecGetArrayRead(x,&xx); CHKERRQ(ierr); */

/*   /\* */
/*      Compute Jacobian entries and insert into matrix. */
/*       - Since this is such a small problem, we set all entries for */
/*         the matrix at once. */
/*   *\/ */
/*   A[0]  = 2.0 + 1200.0*xx[0]*xx[0] - 400.0*xx[1]; */
/*   A[1]  = -400.0*xx[0]; */
/*   A[2]  = -400.0*xx[0]; */
/*   A[3]  = 200; */
/*   ierr  = MatSetValues(B,2,idx,2,idx,A,INSERT_VALUES); CHKERRQ(ierr); */

/*   /\* */
/*      Restore vector */
/*   *\/ */
/*   ierr = VecRestoreArrayRead(x,&xx); CHKERRQ(ierr); */

/*   /\* */
/*      Assemble matrix */
/*   *\/ */
/*   ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); */
/*   ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); */
/*   if (jac != B) { */
/*     ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); */
/*     ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); */
/*   } */
/*   return 0; */
/* } */
