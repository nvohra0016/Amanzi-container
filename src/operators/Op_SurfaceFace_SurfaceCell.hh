/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon (ecoon@lanl.gov)
*/

/*
  Operators

*/

#ifndef AMANZI_OP_SURFACEFACE_SURFACECELL_HH_
#define AMANZI_OP_SURFACEFACE_SURFACECELL_HH_

#include <vector>
#include "DenseMatrix.hh"
#include "Operator.hh"
#include "Op_Face_Cell.hh"

namespace Amanzi {
namespace Operators {

class Op_SurfaceFace_SurfaceCell : public Op_Face_Cell {
 public:
  Op_SurfaceFace_SurfaceCell(const std::string& name,
                             const Teuchos::RCP<const AmanziMesh::Mesh> surf_mesh_)
    : Op_Face_Cell(name, surf_mesh_){};

  virtual void
  ApplyMatrixFreeOp(const Operator* assembler, const CompositeVector& X, CompositeVector& Y) const
  {
    assembler->ApplyMatrixFreeOp(*this, X, Y);
  }

  virtual void SumLocalDiag(CompositeVector& X) const
  {
    AmanziMesh::Mesh const* mesh_ = mesh.get();
    AmanziMesh::Mesh const* surf_mesh_ = surf_mesh.get();
    auto Xv = X.viewComponent<Amanzi::MirrorHost>("face", true);

    Kokkos::parallel_for(
      "Op_SurfaceFace_SurfaceCell::SumLocalDiag", A.size(), KOKKOS_LAMBDA(const int& sf) {
        AmanziMesh::Entity_ID_View cells;
        mesh_->face_get_cells(sf, AmanziMesh::Parallel_kind::ALL, cells);

        auto lm = A[sf];
        auto f0 = surf_mesh_->getEntityParent(AmanziMesh::Entity_kind::CELL, cells(0));
        Kokkos::atomic_add(&Xv(f0, 0), lm(0, 0));
        if (cells.extent(0) > 1) {
          auto f1 = surf_mesh_->getEntityParent(AmanziMesh::Entity_kind::CELL, cells(1));
          Kokkos::atomic_add(&Xv(f1, 0), lm(1, 1));
        }
      });
  }

  virtual void SymbolicAssembleMatrixOp(const Operator* assembler,
                                        const SuperMap& map,
                                        GraphFE& graph,
                                        int my_block_row,
                                        int my_block_col) const
  {
    assembler->SymbolicAssembleMatrixOp(*this, map, graph, my_block_row, my_block_col);
  }

  virtual void AssembleMatrixOp(const Operator* assembler,
                                const SuperMap& map,
                                MatrixFE& mat,
                                int my_block_row,
                                int my_block_col) const
  {
    assembler->AssembleMatrixOp(*this, map, mat, my_block_row, my_block_col);
  }

  virtual void Rescale(const CompositeVector& scaling)
  {
    if (scaling.hasComponent("cell") &&
        scaling.viewComponent("cell", false).extent(1) ==
          mesh->getNumEntities(AmanziMesh::Entity_kind::CELL, AmanziMesh::Parallel_kind::OWNED)) {
      // scaling's cell entry is defined on the surface mesh
      Op_Face_Cell::Rescale(scaling);
    }

    if (scaling.hasComponent("face") &&
        scaling.getComponent("face", false)->getLocalLength() ==
          mesh->parent()->getNumEntities(AmanziMesh::Entity_kind::FACE, AmanziMesh::Parallel_kind::OWNED)) {
      Exceptions::amanzi_throw("Scaling surface cell entities with subsurface "
                               "face vector not yet implemented");
    }
  }


 public:
  Teuchos::RCP<const AmanziMesh::Mesh> surf_mesh;
};

} // namespace Operators
} // namespace Amanzi


#endif
