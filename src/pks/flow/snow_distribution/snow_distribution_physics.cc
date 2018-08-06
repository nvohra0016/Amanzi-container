/* -*-  mode: c++; indent-tabs-mode: nil -*- */

/* -----------------------------------------------------------------------------
This is the overland flow component of ATS.
License: BSD
Authors: Ethan Coon (ecoon@lanl.gov)
----------------------------------------------------------------------------- */

#include "snow_distribution.hh"

namespace Amanzi {
namespace Flow {

// -------------------------------------------------------------
// Diffusion term, div K grad (h + elev)
// -------------------------------------------------------------
void SnowDistribution::ApplyDiffusion_(const Teuchos::Ptr<State>& S,
        const Teuchos::Ptr<CompositeVector>& g) {

  // update the rel perm according to the scheme of choice.
  UpdatePermeabilityData_(S_next_.ptr());

  // update the stiffness matrix
  matrix_->Init();
  Teuchos::RCP<const CompositeVector> cond =
    S_next_->GetFieldData(Keys::getKey(domain_,"upwind_snow_conductivity"), name_);

  matrix_diff_->SetScalarCoefficient(cond, Teuchos::null);
  matrix_diff_->UpdateMatrices(Teuchos::null, Teuchos::null);
  matrix_diff_->ApplyBCs(true, true, true);
  
  // update the potential
  S->GetFieldEvaluator(Keys::getKey(domain_,"snow_skin_potential"))->HasFieldChanged(S.ptr(), name_);
  Teuchos::RCP<const CompositeVector> potential = S->GetFieldData(Keys::getKey(domain_,"snow_skin_potential"));

  // calculate the residual
  matrix_->ComputeNegativeResidual(*potential, *g);
};


// -------------------------------------------------------------
// Accumulation of water, dh/dt
// -------------------------------------------------------------
void
SnowDistribution::AddAccumulation_(const Teuchos::Ptr<CompositeVector>& g) {
  // get these fields
  Teuchos::RCP<const CompositeVector> h1 = S_next_->GetFieldData(key_);
  Epetra_MultiVector h1_positive(*h1->ViewComponent("cell",false));
  const auto& h1_v(*h1->ViewComponent("cell",false));
  for (int c=0; c!=h1_positive.MyLength(); ++c)
    h1_positive[0][c] = h1_v[0][c] > 0. ? h1_v[0][c] : 0.;

  Teuchos::RCP<const CompositeVector> h0 = S_inter_->GetFieldData(key_);
  Epetra_MultiVector h0_positive(*h0->ViewComponent("cell",false));
  const auto& h0_v(*h0->ViewComponent("cell",false));
  for (int c=0; c!=h0_positive.MyLength(); ++c)
    h0_positive[0][c] = h0_v[0][c] > 0. ? h0_v[0][c] : 0.;
  
  
  Teuchos::RCP<const CompositeVector> cv1 =
      S_next_->GetFieldData(Keys::getKey(domain_,"cell_volume"));

  // note 10 is for conversion from precip m SWE to actual m
  double dt = S_next_->time() - S_inter_->time();
  g->ViewComponent("cell",false)->Multiply(10*dt_factor_/dt,
          *cv1->ViewComponent("cell",false), h1_positive, 1.);
  g->ViewComponent("cell",false)->Multiply(-10*dt_factor_/dt,
          *cv1->ViewComponent("cell",false), h0_positive, 1.);
}

} //namespace
} //namespace