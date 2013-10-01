/*
  License: BSD
  Authors: Konstantin Lipnikov (version 2) (lipnikov@lanl.gov)
           Ethan Coon (ecoon@lanl.gov) (ATS version)
  MatrixMFD provides a mimetic discretization for the elliptic operator div K grad u.

*/

#ifndef OPERATORS_MATRIX_MFD_HH_
#define OPERATORS_MATRIX_MFD_HH_

#include <strings.h>

#include "Epetra_Map.h"
#include "Epetra_Operator.h"
#include "Epetra_Vector.h"
#include "Epetra_MultiVector.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_FECrsMatrix.h"
#include "ml_MultiLevelPreconditioner.h"

#include "Ifpack.h"
#include "Ifpack_ILU.h"
#include "Ifpack_AdditiveSchwarz.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_LAPACK.hpp"

#include "Mesh.hh"
#include "Point.hh"
#include "TreeVector.hh"
#include "CompositeVector.hh"
#include "CompositeMatrix.hh"
#include "DenseMatrix.hh"
#include "mfd3d_diffusion.hh"

#include "Matrix.hh"


namespace Amanzi {
namespace Operators {

const int MFD_HEX_FACES = 6;  // Hexahedron is the common element
const int MFD_HEX_NODES = 8;
const int MFD_HEX_EDGES = 12;

const int MFD_QUAD_FACES = 4;  // Quadrilateral is the common element
const int MFD_QUAD_NODES = 4;
const int MFD_QUAD_EDGES = 4;

const int MFD_MAX_FACES = 14;  // Kelvin's tetrakaidecahedron
const int MFD_MAX_NODES = 47;  // These polyhedron parameters must
const int MFD_MAX_EDGES = 60;  // be calculated in Init().


class MatrixMFD : public Matrix,
                  public CompositeMatrix {
 public:

  enum MFDMethod {
    MFD3D_NULL = 0,
    MFD3D_POLYHEDRA,
    MFD3D_POLYHEDRA_SCALED,
    MFD3D_OPTIMIZED,
    MFD3D_OPTIMIZED_SCALED,
    MFD3D_HEXAHEDRA_MONOTONE,
    MFD3D_TWO_POINT_FLUX,
    MFD3D_SUPPORT_OPERATOR
  };

  // Constructor
  MatrixMFD(Teuchos::ParameterList& plist,
            const Teuchos::RCP<const AmanziMesh::Mesh> mesh);

  MatrixMFD(const MatrixMFD& other);


  // Virtual destructor
  virtual ~MatrixMFD() {};

  // CompositeMatrix stuff... FIX ME!
  virtual Teuchos::RCP<const CompositeVectorSpace> domain() const {
    return Teuchos::null; }

  virtual Teuchos::RCP<const CompositeVectorSpace> range() const {
    return Teuchos::null; }

  virtual Teuchos::RCP<CompositeMatrix> Clone() const {
    return Teuchos::rcp(new MatrixMFD(*this)); }



  // Access to local matrices for external tweaking.
  std::vector<double>& Acc_cells() {
    return Acc_cells_;
  }
  std::vector<Teuchos::SerialDenseMatrix<int, double> >& Aff_cells() {
    return Aff_cells_;
  }
  std::vector<Epetra_SerialDenseVector>& Acf_cells() {
    return Acf_cells_;
  }
  std::vector<Epetra_SerialDenseVector>& Afc_cells() {
    return Afc_cells_;
  }
  std::vector<double>& Fc_cells() {
    return Fc_cells_;
  }
  std::vector<Epetra_SerialDenseVector>& Ff_cells() {
    return Ff_cells_;
  }

  // Const access to assembled matrices.
  Teuchos::RCP<const Epetra_Vector> Acc() {
    AssertAssembledOperator_or_die_();
    return Acc_;
  }
  Teuchos::RCP<const Epetra_FECrsMatrix> Aff() {
    AssertAssembledOperator_or_die_();
    return Aff_;
  }
  Teuchos::RCP<const Epetra_CrsMatrix> Afc() {
    AssertAssembledOperator_or_die_();
    return Afc_;
  }
  Teuchos::RCP<const Epetra_CrsMatrix> Acf() {
    AssertAssembledOperator_or_die_();
    return Acf_;
  }
  Teuchos::RCP<const Epetra_FECrsMatrix> Schur() {
    AssertAssembledSchur_or_die_();
    return Sff_;
  }
  Teuchos::RCP<const CompositeVector> rhs() {
    AssertAssembledOperator_or_die_();
    return rhs_;
  }

  // Other accessors/mutators.
  bool symmetric() { return flag_symmetry_; }
  void set_symmetric(bool flag_symmetry) { flag_symmetry_ = flag_symmetry; }
  const Epetra_Comm& Comm() const { return *(mesh_->get_comm()); }

  // Main computational methods
  // -- local matrices
  virtual void CreateMFDmassMatrices(
      const Teuchos::Ptr<std::vector<WhetStone::Tensor> >& K);
  virtual void CreateMFDstiffnessMatrices(
      const Teuchos::Ptr<const CompositeVector>& Krel);
  virtual void CreateMFDrhsVectors();

  virtual void ApplyBoundaryConditions(const std::vector<MatrixBC>& bc_markers,
          const std::vector<double>& bc_values);

  // -- global matrices
  virtual void SymbolicAssembleGlobalMatrices();
  virtual void AssembleGlobalMatrices();
  virtual void ComputeSchurComplement(const std::vector<MatrixBC>& bc_markers,
          const std::vector<double>& bc_values);

  // Operator methods.
  virtual int Apply(const CompositeVector& X,
                     CompositeVector& Y) const;
  virtual int ApplyInverse(const CompositeVector& X,
                            CompositeVector& Y) const;

  virtual void Apply(const TreeVector& X,
                     const Teuchos::Ptr<TreeVector>& Y) const {
    Apply(*X.Data(), *Y->Data());
  }
  virtual void ApplyInverse(const TreeVector& X,
                            const Teuchos::Ptr<TreeVector>& Y) const {
    ApplyInverse(*X.Data(), *Y->Data());
  }

  virtual void ComputeResidual(const CompositeVector& X,
                       const Teuchos::Ptr<CompositeVector>& F) const;
  virtual void ComputeNegativeResidual(const CompositeVector& X,
          const Teuchos::Ptr<CompositeVector>& F) const;

  // Solver methods.
  virtual void InitPreconditioner();
  virtual void UpdatePreconditioner();

  // First derivative quantities.
  virtual void DeriveFlux(const CompositeVector& solution,
                          const Teuchos::Ptr<CompositeVector>& flux) const;
  virtual void DeriveCellVelocity(const CompositeVector& flux,
          const Teuchos::Ptr<CompositeVector>& velocity) const;

  // Consistency methods
  virtual void UpdateConsistentFaceConstraints(const Teuchos::Ptr<CompositeVector>& u);
  virtual void UpdateConsistentFaceCorrection(const CompositeVector& u,
          const Teuchos::Ptr<CompositeVector>& Pu);
  virtual void UpdateConsistentCellCorrection(const CompositeVector& u,
          const Teuchos::Ptr<CompositeVector>& Pu);


 protected:
  void InitializeFromPList_();

  // Assertions of assembly process
  void AssertAssembledOperator_or_die_() const;
  void AssertAssembledSchur_or_die_() const;

  virtual void FillMatrixGraphs_(const Teuchos::Ptr<Epetra_CrsGraph> cf_graph,
          const Teuchos::Ptr<Epetra_FECrsGraph> ff_graph);
  virtual void CreateMatrices_(const Epetra_CrsGraph& cf_graph,
          const Epetra_FECrsGraph& ff_graph);

 protected:
  Teuchos::RCP<const AmanziMesh::Mesh> mesh_;
  Teuchos::ParameterList plist_;
  bool flag_symmetry_;

  // lazy assembly
  bool assembled_operator_;
  bool assembled_schur_;

  MFDMethod method_;

  // local matrices
  std::vector<WhetStone::DenseMatrix > Mff_cells_;
  std::vector<Teuchos::SerialDenseMatrix<int, double> > Aff_cells_;
  std::vector<Epetra_SerialDenseVector> Acf_cells_, Afc_cells_;
  std::vector<double> Acc_cells_;  // duplication may be useful later
  std::vector<Epetra_SerialDenseVector> Ff_cells_;
  std::vector<double> Fc_cells_;

  // global matrices
  Teuchos::RCP<Epetra_Vector> Acc_;
  Teuchos::RCP<Epetra_CrsMatrix> Acf_;
  Teuchos::RCP<Epetra_CrsMatrix> Afc_;  // We generate transpose of this matrix block.
  Teuchos::RCP<Epetra_FECrsMatrix> Aff_;
  Teuchos::RCP<Epetra_FECrsMatrix> Sff_;  // Schur complement

  Teuchos::RCP<CompositeVector> rhs_;

  // diagnostics
  int nokay_;
  int npassed_; // performance of algorithms generating mass matrices

  // available solver methods
  enum PrecMethod { PREC_METHOD_NULL,
                    TRILINOS_ML,
                    TRILINOS_ILU,
                    TRILINOS_BLOCK_ILU,
                    HYPRE_AMG,
                    HYPRE_EUCLID,
                    HYPRE_PARASAILS };
  PrecMethod prec_method_;

  Teuchos::RCP<ML_Epetra::MultiLevelPreconditioner> ml_prec_;
  Teuchos::ParameterList ml_plist_;

#ifdef HAVE_HYPRE
  Teuchos::RCP<Ifpack_Hypre> IfpHypre_Sff_;
  Teuchos::ParameterList hypre_plist_;
  int hypre_ncycles_, hypre_nsmooth_;
  double hypre_tol_, hypre_strong_threshold_;
  int hypre_relax_type_, hypre_coarsen_type_, hypre_cycle_type_;
  int hypre_print_level_,hypre_max_row_sum_,hypre_max_levels_;
  int hypre_max_iter_, hypre_relax_wt_, hypre_interp_type_;
  int hypre_agg_num_levels_, hypre_agg_num_paths_;
#endif

  Teuchos::RCP<Ifpack_ILU> ilu_prec_;
  Teuchos::ParameterList ilu_plist_;

  Teuchos::RCP<Ifpack_Preconditioner> ifp_prec_;
  Teuchos::ParameterList ifp_plist_;

  friend class MatrixMFD_Coupled;
};


}  // namespace AmanziFlow
}  // namespace Amanzi

#endif
