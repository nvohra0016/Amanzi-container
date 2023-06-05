/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon
*/

/*
  State

  A field evaluator for an unchanging cell volume.
*/

#include "EvaluatorIndependentFromFile.hh"
#include "EvaluatorIndependentFunction.hh"
#include "EvaluatorIndependentConstant.hh"
#include "EvaluatorIndependentTensorFunction.hh"
#include "EvaluatorSecondaryMonotypeFromFunction.hh"
#include "EvaluatorPrimaryStaticMesh.hh"
#include "EvaluatorSecondaryMeshedQuantity.hh"

namespace Amanzi {

Utils::RegisteredFactory<Evaluator, EvaluatorIndependentFunction>
  EvaluatorIndependentFunction::fac_("independent variable");
Utils::RegisteredFactory<Evaluator, EvaluatorIndependentFromFile>
  EvaluatorIndependentFromFile::fac_("independent variable from file");
Utils::RegisteredFactory<Evaluator, EvaluatorIndependentConstant>
  EvaluatorIndependentConstant::fac_("independent variable constant");
Utils::RegisteredFactory<Evaluator, EvaluatorIndependentTensorFunction>
  EvaluatorIndependentTensorFunction::fac_("independent variable tensor");

Utils::RegisteredFactory<Evaluator, EvaluatorSecondaryMonotypeFromFunction>
  EvaluatorSecondaryMonotypeFromFunction::fac_("secondary variable from function");

Utils::RegisteredFactory<Evaluator, EvaluatorPrimaryStaticMesh>
  EvaluatorPrimaryStaticMesh::fac_("static mesh");


template<>
Utils::RegisteredFactory<Evaluator, EvaluatorCellVolume>
  EvaluatorCellVolume::fac_("cell volume");
template<>
Utils::RegisteredFactory<Evaluator, EvaluatorFaceArea>
  EvaluatorFaceArea::fac_("face area");
template<>
Utils::RegisteredFactory<Evaluator, EvaluatorMeshElevation>
  EvaluatorMeshElevation::fac_("meshed elevation");
template<>
Utils::RegisteredFactory<Evaluator, EvaluatorMeshSlopeMagnitude>
  EvaluatorMeshSlopeMagnitude::fac_("meshed slope magnitude");

} // namespace Amanzi
