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
// helper function to get coordinates, txyz
//
Kokkos::View<double**> getMeshFunctionCoordinates(double time, const PatchSpace& ps)
{
  const AmanziMesh::Mesh* mesh = ps.mesh.get();
  int dim = mesh->getSpaceDimension();
  Kokkos::View<double**> txyz("txyz", dim + 1, ps.size());

  auto ids = mesh->getSetEntities(ps.region,
          ps.entity_kind,
          ps.ghosted ? AmanziMesh::Parallel_kind::ALL :
          AmanziMesh::Parallel_kind::OWNED);

  // if empty, nothing to do
  if (ids.size() == 0) return Kokkos::View<double**>();

  if (ps.entity_kind == AmanziMesh::NODE) {
    Kokkos::parallel_for(
        "computeMeshFunction txyz init node",
        ps.size(),
        KOKKOS_LAMBDA(const int& i) {
          txyz(0,i) = time;
          auto cc = mesh->getNodeCoordinate(ids[i]);
          txyz(1,i) = cc[0];
          txyz(2,i) = cc[1];
          if (mesh->getSpaceDimension() == 3)
            txyz(3,i) = cc[2];
        });

  } else if (ps.entity_kind == AmanziMesh::CELL) {
    Kokkos::parallel_for(
      "computeMeshFunction txyz init cell", ps.size(), KOKKOS_LAMBDA(const int& i) {
        txyz(0, i) = time;
        auto cc = mesh->getCellCentroid(ids[i]);
        txyz(1, i) = cc[0];
        txyz(2, i) = cc[1];
        if (dim == 3) txyz(3, i) = cc[2];
      });

  } else if (ps.entity_kind == AmanziMesh::FACE) {
    Kokkos::parallel_for(
      "computeMeshFunction txyz init face", ps.size(), KOKKOS_LAMBDA(const int& i) {
        txyz(0, i) = time;
        auto cc = mesh->getFaceCentroid(ids[i]);
        txyz(1, i) = cc[0];
        txyz(2, i) = cc[1];
        if (dim == 3) txyz(3, i) = cc[2];
      });
  }
  return txyz;
}



//
// Computes function of t,x,y,{z} on a patch
//
// template<class Device>
void
computeFunction(const MultiFunction& f, double time, Patch<double>& p)
{
  auto txyz = getMeshFunctionCoordinates(time, p.space);
  Kokkos::fence();
  f.apply(txyz, p.data);
}


//
// Computes function of t,x,y,{z} on patch, sticking the result into a vector
//
// template<class Device>
void
computeFunction(const MultiFunction& f, double time, const PatchSpace& ps, CompositeVector& cv)
{
  AMANZI_ASSERT(ps.mesh == cv.getMesh()); // precondition -- same mesh
  auto txyz = getMeshFunctionCoordinates(time, ps);

  auto ids = ps.mesh->getSetEntities(ps.region,
          ps.entity_kind,
          ps.ghosted ? AmanziMesh::Parallel_kind::ALL :
          AmanziMesh::Parallel_kind::OWNED);

  Kokkos::fence();
  Kokkos::View<double**, Kokkos::LayoutLeft> cv_v = cv.viewComponent(AmanziMesh::to_string(ps.entity_kind), ps.ghosted);
  f.apply(txyz, cv_v, &ids);
}



} // namespace Impl
} // namespace Functions
} // namespace Amanzi
