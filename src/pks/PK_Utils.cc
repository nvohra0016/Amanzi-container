/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

/*
  Process Kernels

  Miscalleneous collection of simple non-member functions.
*/

#include "EvaluatorPrimary.hh"

#include "PK_Utils.hh"

namespace Amanzi {

/* ******************************************************************
* Deep copy of state fields.
****************************************************************** */
void
StateArchive::Add(std::vector<std::string>& fields, const Tag& tag)
{
  tag_ = tag;

  for (const auto& name : fields) {
    if (S_->HasEvaluator(name, tag_)) {
      if (S_->GetEvaluatorPtr(name, tag_)->get_type() == EvaluatorType::PRIMARY) {
        fields_.emplace(name, S_->Get<CompositeVector>(name, tag));
      }
    } else {
      fields_.emplace(name, S_->Get<CompositeVector>(name, tag));
    }
  }
}


/* ******************************************************************
* Deep copy of state fields.
****************************************************************** */
void
StateArchive::Restore(const std::string& passwd)
{
  for (auto it = fields_.begin(); it != fields_.end(); ++it) {
    S_->GetW<CompositeVector>(it->first, tag_, passwd) = it->second;

    if (vo_->getVerbLevel() > Teuchos::VERB_MEDIUM) {
      Teuchos::OSTab tab = vo_->getOSTab();
      *vo_->os() << "reverted field \"" << it->first << "\"" << std::endl;
    }

    if (S_->HasEvaluator(it->first, tag_)) {
      if (S_->GetEvaluatorPtr(it->first, tag_)->get_type() == EvaluatorType::PRIMARY) {
        Teuchos::rcp_dynamic_cast<EvaluatorPrimary<CompositeVector, CompositeVectorSpace>>(
          S_->GetEvaluatorPtr(it->first, tag_))
          ->SetChanged();

        if (vo_->getVerbLevel() > Teuchos::VERB_MEDIUM) {
          Teuchos::OSTab tab = vo_->getOSTab();
          *vo_->os() << "changed status of primary field \"" << it->first << "\"" << std::endl;
        }
      }
    }
  }
}


/* *******************************************************************
* Copy: Evaluator (BASE) -> Field (prev_BASE)
******************************************************************* */
void
StateArchive::CopyFieldsToPrevFields(std::vector<std::string>& fields, const std::string& passwd)
{
  for (auto it = fields.begin(); it != fields.end(); ++it) {
    auto name = Keys::splitKey(*it);
    std::string prev = Keys::getKey(name.first, "prev_" + name.second);
    if (S_->HasRecord(prev, tag_)) {
      fields_.emplace(prev, S_->Get<CompositeVector>(prev));
      S_->GetW<CompositeVector>(prev, tag_, passwd) = S_->Get<CompositeVector>(*it);
    }
  }
}


/* ******************************************************************
* Return a copy
****************************************************************** */
const CompositeVector&
StateArchive::get(const std::string& name)
{
  auto it = fields_.find(name);
  if (it != fields_.end()) return it->second;

  AMANZI_ASSERT(false);
  return std::move(CompositeVector(Teuchos::null));
}


/* ******************************************************************
* Average permeability tensor in horizontal direction.
****************************************************************** */
void
PKUtils_CalculatePermeabilityFactorInWell(const Teuchos::Ptr<State>& S,
                                          Teuchos::RCP<Vector_type>& Kxy)
{
  if (!S->HasRecord("permeability", Tags::DEFAULT)) return;

  const auto& cv = S->Get<CompositeVector>("permeability", Tags::DEFAULT);
  cv.scatterMasterToGhosted("cell");

  int ncells_wghost = S->GetMesh()->getNumEntities(AmanziMesh::Entity_kind::CELL, AmanziMesh::Parallel_kind::ALL);

  Kxy = Teuchos::rcp(new Vector_type(S->GetMesh()->getMap(AmanziMesh::Entity_kind::CELL,true)));
  {
    auto perm = cv.viewComponent<Kokkos::HostSpace>("cell", true);
    auto Kxy_v = Kxy->getLocalViewHost(Tpetra::Access::ReadWrite);
    int idim = std::max(1, (int)perm.extent(1) - 1);
    for (int c = 0; c < ncells_wghost; c++) {
      Kxy_v(c,0) = 0.0;
      for (int i = 0; i < idim; i++) Kxy_v(c,0) += perm(c,i);
      Kxy_v(c,0) /= idim;
    }
  }
}


/* ******************************************************************
* Return coordinate of mesh entity (
****************************************************************** */
AmanziGeometry::Point
PKUtils_EntityCoordinates(int id, AmanziMesh::Entity_ID kind, const AmanziMesh::Mesh& mesh)
{
  if (kind == AmanziMesh::Entity_kind::FACE) {
    return mesh.getFaceCentroid(id);
  } else if (kind == AmanziMesh::Entity_kind::CELL) {
    return mesh.getCellCentroid(id);
  } else if (kind == AmanziMesh::Entity_kind::NODE) {
    int d = mesh.getSpaceDimension();
    AmanziGeometry::Point xn(d);
    xn = mesh.getNodeCoordinate(id);
    return xn;
  } else if (kind == AmanziMesh::Entity_kind::EDGE) {
    return mesh.getEdgeCentroid(id);
  }
  return AmanziGeometry::Point();
}

} // namespace Amanzi
