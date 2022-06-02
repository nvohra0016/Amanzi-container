/*
  MultiPhase PK

  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Authors: Konstantin Lipnikov (lipnikov@lanl.gov)

  Calculates liquid mole fraction from partial gas pressure.
*/

#include "ModelMeshPartition.hh"
#include "MultiphaseDefs.hh"
#include "MoleFractionLiquid.hh"

namespace Amanzi {
namespace Multiphase {

/* ******************************************************************
* Simple constructor
****************************************************************** */
MoleFractionLiquid::MoleFractionLiquid(Teuchos::ParameterList& plist)
  : MultiphaseBaseEvaluator(plist)
{
  if (my_keys_.size() == 0) {
    my_keys_.push_back(std::make_pair(plist_.get<std::string>("my key"), Tags::DEFAULT));
  }
  pressure_gas_key_ = plist_.get<std::string>("pressure gas key");
  x_gas_key_ = plist_.get<std::string>("mole fraction gas key");

  dependencies_.insert(std::make_pair(pressure_gas_key_, Tags::DEFAULT));
  dependencies_.insert(std::make_pair(x_gas_key_, Tags::DEFAULT));
}


/* ******************************************************************
* Copy constructor.
****************************************************************** */
Teuchos::RCP<Evaluator> MoleFractionLiquid::Clone() const {
  return Teuchos::rcp(new MoleFractionLiquid(*this));
}


/* ******************************************************************
* Required member function.
****************************************************************** */
void MoleFractionLiquid::Evaluate_(
    const State& S, const std::vector<CompositeVector*>& results)
{
  const auto& pg_c = *S.Get<CompositeVector>(pressure_gas_key_).ViewComponent("cell");
  const auto& xg_c = *S.Get<CompositeVector>(x_gas_key_).ViewComponent("cell");
  auto& result_c = *results[0]->ViewComponent("cell");

  int ncells = result_c.MyLength();
  for (int c = 0; c != ncells; ++c) {
    result_c[0][c] = pg_c[0][c] * xg_c[n_][c] * kH_;
  }
}


/* ******************************************************************
* Required member function.
****************************************************************** */
void MoleFractionLiquid::EvaluatePartialDerivative_(
    const State& S, const Key& wrt_key, const Tag& wrt_tag,
    const std::vector<CompositeVector*>& results)
{
  const auto& pg_c = *S.Get<CompositeVector>(pressure_gas_key_).ViewComponent("cell");
  const auto& xg_c = *S.Get<CompositeVector>(x_gas_key_).ViewComponent("cell");
  auto& result_c = *results[0]->ViewComponent("cell");
 
  int ncells = result_c.MyLength();
  if (wrt_key == pressure_gas_key_) {
    for (int c = 0; c != ncells; ++c) {
      result_c[0][c] = xg_c[n_][c] * kH_;
    }
  } else if (wrt_key == x_gas_key_) {
    for (int c = 0; c != ncells; ++c) {
      result_c[0][c] = pg_c[0][c] * kH_;
    }
  }
}

}  // namespace Multiphase
}  // namespace Amanzi
