/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon
*/

//! <MISSING_ONELINE_DOCSTRING>

#include "Key.hh"
#include "MeshDefs.hh"
#include "DataStructuresHelpers.hh"
#include "MeshFunction.hh"

namespace Amanzi {
namespace Functions {
namespace Impl {

//
// Computes function of t,x,y,{z} on a patch
//
// template<class Device>
void
computeFunction(const MultiFunction& f, double time, Patch& p)
{
  const AmanziMesh::Mesh* mesh = p.space.mesh.get();
  int dim = mesh->getSpaceDimension();
  Kokkos::View<double**> txyz("txyz", dim + 1, p.size());

  auto ids = mesh->getSetEntities(p.space.region,
          p.space.entity_kind,
          p.space.ghosted ? AmanziMesh::Parallel_kind::ALL :
          AmanziMesh::Parallel_kind::OWNED);

  // if empty, nothing to do
  if (ids.size() == 0) return;

  if (p.space.entity_kind == AmanziMesh::NODE) {
    Kokkos::parallel_for(
        "computeMeshFunction txyz init node",
        p.size(),
        KOKKOS_LAMBDA(const int& i) {
          txyz(0,i) = time;
          auto cc = mesh->getNodeCoordinate(ids[i]);
          txyz(1,i) = cc[0];
          txyz(2,i) = cc[1];
          if (mesh->getSpaceDimension() == 3)
            txyz(3,i) = cc[2];
        });

  } else if (p.space.entity_kind == AmanziMesh::CELL) {
    Kokkos::parallel_for(
      "computeMeshFunction txyz init cell", p.size(), KOKKOS_LAMBDA(const int& i) {
        txyz(0, i) = time;
        auto cc = mesh->getCellCentroid(ids[i]);
        txyz(1, i) = cc[0];
        txyz(2, i) = cc[1];
        if (dim == 3) txyz(3, i) = cc[2];
      });

  } else if (p.space.entity_kind == AmanziMesh::FACE) {
    Kokkos::parallel_for(
      "computeMeshFunction txyz init face", p.size(), KOKKOS_LAMBDA(const int& i) {
        txyz(0, i) = time;
        auto cc = mesh->getFaceCentroid(ids[i]);
        txyz(1, i) = cc[0];
        txyz(2, i) = cc[1];
        if (dim == 3) txyz(3, i) = cc[2];
      });
  }
  Kokkos::fence();
  f.apply(txyz, p.data);
}


} // namespace Impl
} // namespace Functions
} // namespace Amanzi
