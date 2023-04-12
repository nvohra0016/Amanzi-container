/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon
*/

//! <MISSING_ONELINE_DOCSTRING>
#pragma once

#include <utility>
#include <vector>
#include <string>

#include "Teuchos_RCP.hpp"

#include "Mesh.hh"
#include "MultiFunction.hh"
#include "Patch.hh"
#include "CompositeVectorSpace.hh"
#include "CompositeVector.hh"

namespace Amanzi {
namespace Functions {


//
// Class used to hold space and a functor to evaluate on that space
//
template<class Functor, typename Marker=bool>
class MeshFunction {

public:
  using Spec = std::tuple<std::string, PatchSpace, Teuchos::RCP<const Functor>, Marker>;
  using SpecList = std::vector<Spec>;
  using Marker_type = Marker;

  MeshFunction() {}
  MeshFunction(const Teuchos::RCP<const AmanziMesh::Mesh>& mesh)
    : mesh_(mesh) {}

  Teuchos::RCP<const AmanziMesh::Mesh>& getMesh() const { return mesh_; }
  void setMesh(const Teuchos::RCP<const AmanziMesh::Mesh>& mesh) {
    mesh_ = mesh;
    for (auto& spec : *this) std::get<1>(spec).mesh = mesh;
  }

  // add a spec -- others may inherit this and overload to do some checking?
  virtual void addSpec(const Spec& spec) {
    if (mesh_ == Teuchos::null) setMesh(std::get<1>(spec).mesh);
    AMANZI_ASSERT(std::get<1>(spec).mesh == mesh_);
    spec_list_.push_back(spec);
  }

  // access specs
  using spec_iterator = typename SpecList::const_iterator;
  using size_type = typename SpecList::size_type;
  spec_iterator begin() const { return spec_list_.begin(); }
  spec_iterator end() const { return spec_list_.end(); }
  size_type size() const { return spec_list_.size(); }

  using nc_spec_iterator = typename SpecList::iterator;
  nc_spec_iterator begin() { return spec_list_.begin(); }
  nc_spec_iterator end() { return spec_list_.end(); }

  // data creation
  Teuchos::RCP<CompositeVectorSpace> createCVS(bool ghosted) const {
    auto cvs = Teuchos::rcp(new CompositeVectorSpace());
    cvs->SetMesh(mesh_)
      ->SetGhosted(ghosted);
    for (auto [compname, ps, functor, marker] : *this) {
      cvs->AddComponent(compname,
                        ps.entity_kind,
                        ps.num_vectors);
    }
    return cvs;
  };

  Teuchos::RCP<MultiPatchSpace> createMPS(bool ghosted) const {
    auto mps = Teuchos::rcp(new MultiPatchSpace(mesh_, ghosted));
    for (auto spec : *this) mps->addPatch(std::get<1>(spec));
    return mps;
  }

 protected:
  Teuchos::RCP<const AmanziMesh::Mesh> mesh_;
  SpecList spec_list_;
};


namespace Impl {
//
// Computes function on a patch.
//
//template<class Device=DefaultDevice>
void
computeFunction(const MultiFunction& f, double time, Patch& p);

void
computeFunctionDepthCoordinate(const MultiFunction& f, double time, Patch& p);


} // namespace Impl
} // namespace Functions
} // namespace Amanzi
