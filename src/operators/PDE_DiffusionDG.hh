/*
  Operators

  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Author: Konstantin Lipnikov (lipnikov@lanl.gov)

  DiffusionDG implements the Diffusion interface using
  discontinuous polynomials.
*/

#ifndef AMANZI_OPERATOR_PDE_DIFFUSION_DG_HH_
#define AMANZI_OPERATOR_PDE_DIFFUSION_DG_HH_

#include <strings.h>

// TPLs
#include "Teuchos_RCP.hpp"

// Amanzi
#include "CompositeVector.hh"
#include "DenseMatrix.hh"
#include "Tensor.hh"

// Operators
#include "PDE_HelperDiscretization.hh"

namespace Amanzi {
namespace Operators {

class BCs;

class PDE_DiffusionDG : public PDE_HelperDiscretization {
 public:
  PDE_DiffusionDG(Teuchos::ParameterList& plist,
                  const Teuchos::RCP<const AmanziMesh::Mesh>& mesh) :
      PDE_HelperDiscretization(mesh),
      plist_(plist),
      K_(Teuchos::null)
  {
    global_op_ = Teuchos::null;
    operator_type_ = OPERATOR_DIFFUSION_DG;
    Init_(plist);
  }

  // main virtual members
  // -- setup
  void SetTensorCoefficient(const Teuchos::RCP<std::vector<WhetStone::Tensor> >& K);

  // -- creation of an operator
  using PDE_HelperDiscretization::UpdateMatrices;
  virtual void UpdateMatrices(const Teuchos::Ptr<const CompositeVector>& u,
                              const Teuchos::Ptr<const CompositeVector>& p) override;

  // -- postprocessing: calculated flux u from potential p
  virtual void UpdateFlux(const Teuchos::Ptr<const CompositeVector>& u,
                          const Teuchos::Ptr<CompositeVector>& flux) override;

 private:
  virtual void Init_(Teuchos::ParameterList& plist);

 private:
  Teuchos::ParameterList plist_;
  std::string method_, matrix_;
  int method_order_;

  Teuchos::RCP<std::vector<WhetStone::Tensor> > K_;

  // operator and schemas
  Schema global_schema_col_, global_schema_row_;
  Schema local_schema_col_, local_schema_row_;

  // other operators
  Teuchos::RCP<Op> jump_up_op_, jump_pu_op_, jump_uu_op_;
};

}  // namespace Operators
}  // namespace Amanzi


#endif
