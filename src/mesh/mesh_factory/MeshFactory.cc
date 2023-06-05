/*
  Copyright 2010-201x held jointly by LANL, ORNL, LBNL, and PNNL.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon (coonet@ornl.gov)
*/

#include "exceptions.hh"
#include "errors.hh"

#include "MeshFramework.hh"
#include "Mesh.hh"
#include "MeshFrameworkColumn.hh"
#include "MeshSurfaceCell.hh"
#include "MeshLogical.hh"
#include "MeshLogicalFactory.hh"
#include "MeshFactory.hh"

namespace Amanzi {
namespace AmanziMesh {

Teuchos::RCP<Mesh>
MeshFactory::create(const Teuchos::RCP<const Mesh>& parent_mesh,
                    const MeshFramework::Entity_ID_View& setids,
                    const Entity_kind setkind,
                    const bool flatten)
{
  if (parent_mesh->getMeshFramework() == Teuchos::null) {
    Errors::Message msg("Cannot create an extracted mesh from a parent whose framework has been deleted.");
  }
  Teuchos::RCP<MeshFramework> mesh_fw =
    MeshFrameworkFactory::create(parent_mesh, setids, setkind, flatten);
  auto mesh = Teuchos::rcp(new Mesh(mesh_fw, Teuchos::rcp(new MeshAlgorithms()), Teuchos::null));
  mesh->setParentMesh(parent_mesh);
  return mesh;
}


Teuchos::RCP<Mesh>
MeshFactory::create(const Teuchos::RCP<const Mesh>& parent_mesh,
       const std::vector<std::string>& setnames,
       const Entity_kind setkind,
       const bool flatten)
{
  if (parent_mesh->getMeshFramework() == Teuchos::null) {
    Errors::Message msg("Cannot create an extracted mesh from a parent whose framework has been deleted.");
  }
  Teuchos::RCP<MeshFramework> mesh_fw =
    MeshFrameworkFactory::create(parent_mesh, setnames, setkind, flatten);
  auto mesh = Teuchos::rcp(new Mesh(mesh_fw, Teuchos::rcp(new MeshAlgorithms()), Teuchos::null));
  mesh->setParentMesh(parent_mesh);
  return mesh;
}



Teuchos::RCP<Mesh>
MeshFactory::createLogical(Teuchos::ParameterList& log_plist)
{
  MeshLogicalFactory log_fac(comm_, gm_);
  Teuchos::RCP<MeshFramework> mesh_fw = log_fac.Create(log_plist);
  auto mesh = Teuchos::rcp(new Mesh(mesh_fw,
          Teuchos::rcp(new MeshLogicalAlgorithms()), Teuchos::null));
  return mesh;
}


// Create a 1D Column Mesh from a columnar structured volume mesh.
//
Teuchos::RCP<Mesh>
MeshFactory::createColumn(const Teuchos::RCP<Mesh>& parent,
                          int col_id,
                          const Teuchos::RCP<Teuchos::ParameterList>& plist)
{
  AMANZI_ASSERT(col_id >= 0);
  AMANZI_ASSERT(col_id < parent->columns.num_columns_owned);

  // create a framework of the extracted 3D column
  parent->buildColumns();

  // create the extracted mesh of the column of cells
  MeshFrameworkFactory fac(getCommSelf(), gm_, plist);
  auto col_list = parent->columns.getCells<MemSpace_kind::HOST>(col_id);
  auto extracted_mesh = fac.create(parent, col_list, Entity_kind::CELL, false);

  // create the MeshColumn object
  return Teuchos::rcp(new Mesh(extracted_mesh, Teuchos::rcp(new MeshColumnAlgorithms()), plist));
}

// Create a MeshSurfaceCell from a MeshFrameworkColumn
Teuchos::RCP<Mesh>
MeshFactory::createSurfaceCell(const Teuchos::RCP<const Mesh>& parent)
{
  if (parent->getMeshFramework() == Teuchos::null) {
    Errors::Message msg("Cannot create a surface cell mesh from a column whose framework has been deleted.");
  }
  Teuchos::RCP<MeshFramework> mesh_surf_cell_fw =
    Teuchos::rcp(new MeshSurfaceCell(parent->getMeshFramework()));
  auto mesh = Teuchos::rcp(new Mesh(mesh_surf_cell_fw, Teuchos::rcp(new MeshAlgorithms()), Teuchos::null));
  mesh->setParentMesh(parent);
  return mesh;
}

} // namespace AmanziMesh
} // namespace Amanzi




