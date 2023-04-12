/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

/*
  Operators

*/

#include <vector>

// TPLs

#include "DeRham_Node.hh"

#include "PDE_MagneticDiffusion_TM.hh"
#include "Op.hh"

namespace Amanzi {
namespace Operators {

/* ******************************************************************
 * System modification before solving the problem.
 * **************************************************************** */
void
PDE_MagneticDiffusion_TM::ModifyMatrices(CompositeVector& E, CompositeVector& B, double dt)
{
  B.ScatterMasterToGhosted("face");
  global_op_->rhs()->PutScalarGhosted(0.0);

  const Epetra_MultiVector& Bf = *B.viewComponent("face", true);
  Epetra_MultiVector& rhs_v = *global_op_->rhs()->viewComponent("node", true);

  std::vector<int> dirs;
  AmanziMesh::Entity_ID_List faces, nodes;

  for (int c = 0; c < ncells_owned; ++c) {
    WhetStone::DenseMatrix<>& Acell = local_op_->matrices[c];
    Acell.Scale(dt / 2);

    const WhetStone::DenseMatrix<>& Mcell = mass_op_[c];
    const WhetStone::DenseMatrix<>& Ccell = curl_op_[c];

    mesh_->getCellFacesAndDirs(c, faces, &dirs);
    mesh_->getCellNodes(c, nodes);

    int nfaces = faces.size();
    int nnodes = nodes.size();

    WhetStone::DenseVector<> v1(nfaces), v2(nfaces), v3(nnodes);

    for (int n = 0; n < nfaces; ++n) {
      int f = faces[n];
      v1(n) = Bf[0][f] * dirs[n] * mesh_->getFaceArea(f)
    }

    Mcell.Multiply(v1, v2, false);
    Ccell.Multiply(v2, v3, true);

    for (int n = 0; n < nnodes; ++n) {
      int v = nodes[n];
      rhs_v[0][v] += v3(n);
    }
  }

  global_op_->rhs()->GatherGhostedToMaster("node", Add);
}


/* ******************************************************************
 * Solution postprocessing
 * **************************************************************** */
void
PDE_MagneticDiffusion_TM::ModifyFields(CompositeVector& E, CompositeVector& B, double dt)
{
  B.ScatterMasterToGhosted("face");

  Epetra_MultiVector& Ev = *E.viewComponent("node", true);
  Epetra_MultiVector& Bf = *B.viewComponent("face", false);

  std::vector<int> dirs;
  AmanziMesh::Entity_ID_List faces, nodes;

  std::vector<bool> fflag(nedges_wghost, false);

  for (int c = 0; c < ncells_owned; ++c) {
    const WhetStone::DenseMatrix<>& Ccell = curl_op_[c];

    mesh_->getCellFacesAndDirs(c, faces, &dirs);
    mesh_->getCellNodes(c, nodes);

    int nfaces = faces.size();
    int nnodes = nodes.size();

    WhetStone::DenseVector<> v1(nnodes), v2(nfaces);

    for (int n = 0; n < nnodes; ++n) {
      int v = nodes[n];
      v1(n) = Ev[0][v];
    }

    Ccell.Multiply(v1, v2, false);

    for (int n = 0; n < nfaces; ++n) {
      int f = faces[n];
      if (!fflag[f]) {
        Bf[0][f] -= dt * v2(n) * dirs[n] / mesh_->getFaceArea(f)
        fflag[f] = true;
      }
    }
  }
}


/* ******************************************************************
 * Apply boundary conditions to the local matrices. We always zero-out
 * matrix rows for essential test BCs. As to trial BCs, there are
 * options: (a) eliminate or not, (b) if eliminate, then put 1 on
 * the diagonal or not.
 ****************************************************************** */
void
PDE_MagneticDiffusion_TM::ApplyBCs(bool primary, bool eliminate, bool essential_eqn)
{
  if (local_op_schema_ == (OPERATOR_SCHEMA_BASE_CELL | OPERATOR_SCHEMA_DOFS_NODE)) {
    Teuchos::RCP<const BCs> bc_f, bc_v;
    for (auto bc = bcs_trial_.begin(); bc != bcs_trial_.end(); ++bc) {
      if ((*bc)->kind() == AmanziMesh::Entity_kind::FACE) {
        bc_f = *bc;
      } else if ((*bc)->kind() == AmanziMesh::Entity_kind::NODE) {
        bc_v = *bc;
      }
    }
    ApplyBCs_Node_(bc_f.ptr(), bc_v.ptr(), primary, eliminate, essential_eqn);
  }
}


/* ******************************************************************
 * Apply BCs on cell operators
 ****************************************************************** */
void
PDE_MagneticDiffusion_TM::ApplyBCs_Node_(const Teuchos::Ptr<const BCs>& bc_f,
                                         const Teuchos::Ptr<const BCs>& bc_v,
                                         bool primary,
                                         bool eliminate,
                                         bool essential_eqn)
{
  AmanziMesh::Entity_ID_List nodes, faces, cells;
  std::vector<int> fdirs;

  global_op_->rhs()->PutScalarGhosted(0.0);
  Epetra_MultiVector& rhs_node = *global_op_->rhs()->viewComponent("node", true);

  // calculate number of cells for each node
  // move to properties of BCs (lipnikov@lanl.gov)
  int nnodes_wghost = mesh_->getNumEntities(AmanziMesh::Entity_kind::NODE, AmanziMesh::Parallel_kind::ALL);
  std::vector<int> node_get_cells(nnodes_wghost, 0);
  for (int c = 0; c != ncells_wghost; ++c) {
    mesh_->getCellNodes(c, nodes);
    int nnodes = nodes.size();

    for (int n = 0; n < nnodes; ++n) { getNodeCells[nodes[n]]++; }
  }

  int nn(0), nm(0);
  for (int c = 0; c != ncells_owned; ++c) {
    bool flag(true);
    WhetStone::DenseMatrix<>& Acell = local_op_->matrices[c];

    // BCs of edges: typically this is electric field
    if (bc_v != Teuchos::null) {
      const std::vector<int>& bc_model = bc_v->bc_model();
      const std::vector<double>& bc_value = bc_v->bc_value();

      mesh_->getCellNodes(c, nodes);
      int nnodes = nodes.size();

      // essential conditions for test functions
      for (int n = 0; n != nnodes; ++n) {
        int v = nodes[n];
        if (bc_model[v] == OPERATOR_BC_DIRICHLET) {
          if (flag) { // make a copy of elemental matrix
            local_op_->matrices_shadow[c] = Acell;
            flag = false;
          }
          for (int m = 0; m < nnodes; m++) Acell(n, m) = 0.0;
        }
      }

      for (int n = 0; n != nnodes; ++n) {
        int v = nodes[n];
        double value = bc_value[v];

        if (bc_model[v] == OPERATOR_BC_DIRICHLET) {
          if (flag) { // make a copy of cell-based matrix
            local_op_->matrices_shadow[c] = Acell;
            flag = false;
          }

          if (eliminate) {
            for (int m = 0; m < nnodes; m++) {
              rhs_node[0][nodes[m]] -= Acell(m, n) * value;
              Acell(m, n) = 0.0;
            }
          }

          if (essential_eqn) {
            rhs_node[0][v] = value;
            Acell(n, n) = 1.0 / getNodeCells[v];
          }
        }
      }
    }
  }

  global_op_->rhs()->GatherGhostedToMaster("node", Add);
}


/* ******************************************************************
 * Calculates Ohmic heating
 ****************************************************************** */
double
PDE_MagneticDiffusion_TM::CalculateOhmicHeating(const CompositeVector& E)
{
  E.ScatterMasterToGhosted("node");
  const Epetra_MultiVector& Ev = *E.viewComponent("node", true);

  AmanziMesh::Entity_ID_List nodes;

  double energy(0.0);
  for (int c = 0; c < ncells_owned; ++c) {
    mesh_->getCellNodes(c, nodes);
    int nnodes = nodes.size();

    double volume = mesh_->getCellVolume(c)
    double tmp = volume / (*K_)[c](0, 0) / nnodes;
    for (int n = 0; n < nnodes; ++n) {
      int v = nodes[n];
      energy += Ev[0][v] * Ev[0][v] * tmp;
    }
  }

  double tmp(energy);
  Teuchos::reduceAll<int>(*mesh_->get_comm(),Teuchos::REDUCE_SUM, 1,&tmp, &energy);

  return energy;
}

} // namespace Operators
} // namespace Amanzi
