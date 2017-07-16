//
// ------------------------------
//
// This program solves an instance of a non-linear problem that is
// regularly encountered in economics.
//
// The function F(x) that one solves is
//
// f_i = price_i + beta_i * x_i * (sum_i { beta_i * x_i^rho})^(alpha/(rho-1))
//
// which is a (simplified) first-order derivative of a production
// function. Solving F(x) = 0 gives you the equilibrium values of the
// input to the production function x.


//
// Part of this is copied from NOXNewton1.cpp from the HandsOn-Tutorial
// 
#include <iostream>

#include "Epetra_ConfigDefs.h"
#ifdef HAVE_MPI
#  include "mpi.h"
#  include "Epetra_MpiComm.h"
#else
#  include "Epetra_SerialComm.h"
#endif 



#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_CrsMatrix.h"

#include "NOX.H"
#include "NOX_Epetra_Interface_Required.H"
#include "NOX_Epetra_Interface_Jacobian.H"
#include "NOX_Epetra_LinearSystem_AztecOO.H"
#include "NOX_Epetra_Group.H"
#include <NOX_Epetra_MatrixFree.H>

// public NOX::Epetra::Interface::Jacobian


class ProductionFunction : public NOX::Epetra::Interface::Required
{
public:

    // Constructor
    ProductionFunction (Epetra_Vector& InitialGuess) : InitialGuess_ (new Epetra_Vector (InitialGuess))
        {
        }

    // Destructor
    ~ProductionFunction() {}


    //
    //
    bool computeF (const Epetra_Vector& input,
                   Epetra_Vector& output,
                   NOX::Epetra::Interface::Required::FillType F)
        {
            return true;
        };

    bool
    computeJacobian(const Epetra_Vector & input,
                    Epetra_Operator & outputJac)
        {
            throw std::runtime_error ("*** SimpleProblemInterface does not implement "
                                      "computing a Jacobian from an "
                                      "Epetra_RowMatrix ***");
        }

    bool 
    computePrecMatrix (const Epetra_Vector & x, 
                       Epetra_RowMatrix & M) 
        {
            throw std::runtime_error ("*** SimpleProblemInterface does not implement "
                                      "computing an explicit preconditioner from an "
                                      "Epetra_RowMatrix ***");
        }  

    bool 
    computePreconditioner (const Epetra_Vector & x, 
                           Epetra_Operator & O)
        {
            throw std::runtime_error ("*** SimpleProblemInterface does not implement "
                                      "computing an explicit preconditioner from an "
                                      "Epetra_Operator ***");
        }  
    

private:

    Teuchos::RCP<Epetra_Vector> InitialGuess_;
    // Teuchos::RCP<NOX::Epetra::MatrixFree> MF;
};


//
// ------------------------------
//
// main function
//
// ------------------------------
//

int main(int argc, char **argv)
{

    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using Teuchos::parameterList;
    
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    Epetra_MpiComm CommWorld (MPI_COMM_WORLD);
#else
    Epetra_SerialComm CommWorld;
#endif

    //
    // General Parameters
    int myPID = CommWorld.MyPID();
    int NumberProcesses = CommWorld.NumProc();
    int n = 20;


    //
    // Parameter of the nonlinear problem
    double drts = 0.8;
    double rho = 2;
    double gamma = 1.15; // TODO: make random
    double tmp;
    double betaSum, prodSum, Y;

    if(myPID == 0) {
        std::cout << "Number of Processes: " << NumberProcesses << std::endl;
    }

    // Set the mapping of the values to the processors
    Epetra_Map Map (n, 0, CommWorld);

    // std::cout << Map << std::endl;
    // std::cout << myPID << ": "
    //           << Map.MinAllGID() << ", "
    //           << Map.MaxAllGID() << ".   "
    //           << Map.MinMyGID() << ", "
    //           << Map.MaxMyGID() << ".   "
    //           << Map.MinLID() << ", "
    //           << Map.MaxLID() << "."
    //           << std::endl;

    // And create the vectors
    Epetra_Vector betas (Map);
    Epetra_Vector xorigin (Map);
    Epetra_Vector prices (Map);
    Epetra_Vector xguess (Map);


    // beta vector
    
    betas.Random();
    betas.Abs(betas);

    betas.Norm1(&betaSum);
    betas.Scale(1 / betaSum);

    if (myPID == 0) {
        std::cout << "Betas: " << betaSum << std::endl;
    }
    std::cout << betas << std::endl;

    // control if scaling worked
    // if (myPID == 0) {
    //     std::cout << "Betas: " << std::endl;
    // }
    // std::cout << betas << std::endl;
    // betas.Norm1(&betaSum);
    // std::cout << "BetaSum: " << betaSum << std::endl;

    

    // "true" x values
    prices.PutScalar(150);
    xorigin.Random();
    xorigin.Abs(xorigin);
    xorigin.Update(1, prices, 40);

    
    // if (myPID == 0) {
    //     std::cout << "xorigin: " << std::endl;
    // }
    // std::cout << xorigin << std::endl;

    // prices
    for(int i = Map.MinLID(); i <= Map.MaxLID(); i++) {
        tmp = pow(xorigin[i], rho);
        prices.ReplaceMyValues(1, &tmp, &i);
    }
    prices.Multiply(1, prices, betas, 0);
    prices.Norm1(&prodSum);

    Y = -1 * gamma * pow(prodSum, drts/rho);

    prodSum = pow(prodSum, drts/rho - 1);
    prices.Scale(prodSum, xorigin);
    prices.Multiply(-1, prices, betas, 0);

    if (myPID == 0) {
        std::cout << "prices: " << std::endl;
    }
    std::cout << prices << std::endl;


    // x guess
    xguess.PutScalar(175);

    // TODO: start timer

    // TODO: instantiate interface
    RCP<ProductionFunction> prodFunction =
        rcp(new ProductionFunction (xguess));
    // TODO: instantiate matrix-free jacobian
    // TODO: instantiate ParameterLists


    
    RCP<Nox::Epetra::Interface::Required> iReq = prodFunction;
    



    
    // 
    // End
    
#ifdef HAVE_MPI
    // Make sure that everybody is done before calling MPI_Finalize().
    MPI_Barrier (MPI_COMM_WORLD);
    MPI_Finalize();
#endif
    return EXIT_SUCCESS;    
}
