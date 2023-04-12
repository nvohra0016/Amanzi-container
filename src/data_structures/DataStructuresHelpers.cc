/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon
*/
// Helpers for copying data between various data structures.

#include "Mesh_Algorithms.hh"
#include "DataStructuresHelpers.hh"

namespace Amanzi {

void
patchToCompositeVector(const Patch& p, const std::string& component, CompositeVector& cv)
{
  auto cv_c = cv.viewComponent(component, p.space.ghosted);

  const auto& mesh = cv.getMesh();
  auto ids = mesh->getSetEntities(p.space.region,
          p.space.entity_kind,
          p.space.ghosted ? AmanziMesh::Parallel_kind::ALL : AmanziMesh::Parallel_kind::OWNED);

  if (component != "boundary_face") {
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> range({ 0, 0 }, { p.data.extent(0), p.data.extent(1) });
    Kokkos::parallel_for(
      "patchToCompositeVector", range, KOKKOS_LAMBDA(const int& i, const int& j) {
        cv_c(ids[i], j) = p.data(i, j);
      });
  } else {
    AMANZI_ASSERT(false && "Not yet implemented: patchToCompositeVector with boundary_face");
    // have to do some dancing here... this is not correct because p.data is
    // based on faces, but component is based on boundary faces.  Need to
    // either create temporary space, then import, or more likely, unpack the
    // mapping to make sure we only access cv_c on boundary faces.
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> range({ 0, 0 }, { p.data.extent(0), p.data.extent(1) });
    Kokkos::parallel_for(
      "patchToCompositeVector boundary_face", range, KOKKOS_LAMBDA(const int& i, const int& j) {
        cv_c(ids[i], j) = p.data(i, j);
      });
  }
}

void
patchToCompositeVector(const Patch& p,
                       const std::string& component,
                       CompositeVector& cv,
                       CompositeVector_<int>& flag_cv)
{
  auto ids = cv.getMesh()->getSetEntities(p.space.region,
          p.space.entity_kind,
          p.space.ghosted ? AmanziMesh::Parallel_kind::ALL :
          AmanziMesh::Parallel_kind::OWNED);

  if (component != "boundary_face") {
    // AMANZI_ASSERT(ids.extent(0) == p.data.extent(0));
    auto flag_type = p.space.flag_type;

    auto cv_c = cv.viewComponent(component, p.space.ghosted);
    auto flag_c = flag_cv.viewComponent(component, p.space.ghosted);

    Kokkos::parallel_for(
      "patchToCompositeVector", p.data.extent(0), KOKKOS_LAMBDA(const int& i) {
        cv_c(ids[i], 0) = p.data(i, 0);
        flag_c(ids[i], 0) = flag_type;
      });
  } else {
    AMANZI_ASSERT(false && "Not yet implemented: patchToCompositeVector with boundary_face");
    // have to do some dancing here... this is not correct because p.data is
    // based on faces, but component is based on boundary faces.  Need to
    // either create temporary space, then import, or more likely, unpack the
    // mapping to make sure we only access cv_c on boundary faces.
    // Kokkos::parallel_for(
    //     "patchToCompositeVector boundary_face",
    //     p.data.extent(0),
    //     KOKKOS_LAMBDA(const int& i) {
    //       cv_c(ids(i),0) = p.data(i,0);
    //       flag_c(ids(i), 0) = p.space.flag_type;
    //     });
  }
}


//
// Copies values from a set of patches into a vector.
//
void
multiPatchToCompositeVector(const MultiPatch& mp, const std::string& component, CompositeVector& cv)
{
  for (const auto& p : mp) { patchToCompositeVector(p, component, cv); }
}

//
// Copies values and flag from a set of patches into a vector and a flag vector.
//
void multiPatchToCompositeVector(const MultiPatch& mp,
                                        const std::string& component,
                                        CompositeVector& cv,
                                        CompositeVector_<int>& flag)
{
  for (const auto& p : mp) { patchToCompositeVector(p, component, cv, flag); }
}

// -----------------------------------------------------------------------------
// Interpolate pressure ICs on cells to ICs for lambda (faces).
// -----------------------------------------------------------------------------
void
DeriveFaceValuesFromCellValues(CompositeVector& cv)
{
  if (cv.hasComponent("face")) {
    cv.scatterMasterToGhosted("cell");

    const auto cv_c = cv.viewComponent("cell", true);
    auto cv_f = cv.viewComponent("face", false);
    const AmanziMesh::Mesh* mesh = &*cv.getMesh();

    Kokkos::parallel_for(
      "CompositeVector::DeriveFaceValuesFromCellValues loop 1",
      cv_f.extent(0),
      KOKKOS_LAMBDA(decltype(cv_f)::size_type f) {
        int ncells = mesh->getFaceNumCells(f, AmanziMesh::Parallel_kind::ALL);
        double face_value = 0.0;
        for (int n = 0; n != ncells; ++n) { face_value += cv_c(mesh->getFaceCell(f,n), 0); }
        cv_f(f, 0) = face_value / ncells;
      });
  } else if (cv.hasComponent("boundary_face")) {
    AmanziMesh::copyCellsToBoundaryFaces(*cv.getMesh(), *cv.getComponent("cell", false), *cv.getComponent("boundary_face", false));
  }
}


} // namespace Amanzi
