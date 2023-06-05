/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon (coonet@ornl.gov)
*/

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "Teuchos_RCP.hpp"
#include "MeshFunction.hh"
#include "CompositeVector.hh"

namespace Amanzi {
namespace Functions {

class CompositeVectorFunction : public MeshFunction {
 public:
  using MeshFunction::MeshFunction;
  void Compute(double time, CompositeVector& vec);

  void FlagToVector(CompositeVector_<int>& flag_vec);
};


//
// Creates a function without a mesh, which must be set later.  For use by
// evaluators.
//
// inline Teuchos::RCP<CompositeVectorFunction>
// createCompositeVectorFunction(Teuchos::ParameterList& plist)
// {
//   return Teuchos::rcp(new CompositeVectorFunction(plist));
// }


//
// Creates a function with a mesh.  Preferred version of the above, which
// should get deprecated eventually.
//
// inline Teuchos::RCP<CompositeVectorFunction>
// createCompositeVectorFunction(Teuchos::ParameterList& plist,
//                               const Teuchos::RCP<const AmanziMesh::Mesh>& mesh)
// {
//   auto func = createCompositeVectorFunction(plist);
//   func->setMesh(mesh);
//   return func;
// }


} // namespace Functions
} // namespace Amanzi
