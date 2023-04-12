/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors:
*/

#include "Reductions.hh"
#include "AmanziVector.hh"
#include "DataDebug.hh"

namespace Amanzi {

DataDebug::DataDebug(Teuchos::RCP<AmanziMesh::Mesh> mesh) : mesh_(mesh) {}


void
DataDebug::write_region_data(std::string& region_name,
                             const Vector_type& data,
                             std::string& description)
{
  if (!mesh_->isValidSetName(region_name, AmanziMesh::Entity_kind::CELL)) { throw std::exception(); }
  unsigned int mesh_block_size =
    mesh_->getSetSize(region_name, AmanziMesh::Entity_kind::CELL, AmanziMesh::Parallel_kind::OWNED);
  auto cell_ids = mesh_->getSetEntities(
    region_name, AmanziMesh::Entity_kind::CELL, Amanzi::AmanziMesh::Parallel_kind::OWNED);

  std::cerr << "Printing " << description << " on region " << region_name << std::endl;
  {
    auto data_v = data.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (auto c : cell_ids) {
      std::cerr << std::fixed << description << "(" << data.getMap()->getGlobalElement(c) << ") = " << data_v(c,0)
                << std::endl;
    }
  }
}


void
DataDebug::write_region_statistics(std::string& region_name,
                                   const Vector_type& data,
                                   std::string& description)
{
  if (!mesh_->isValidSetName(region_name, AmanziMesh::Entity_kind::CELL)) { throw std::exception(); }
  unsigned int mesh_block_size =
    mesh_->getSetSize(region_name, AmanziMesh::Entity_kind::CELL, AmanziMesh::Parallel_kind::OWNED);
  auto cell_ids = mesh_->getSetEntities(
    region_name, AmanziMesh::Entity_kind::CELL, Amanzi::AmanziMesh::Parallel_kind::OWNED);

  // find min and max and their indices
  auto min_loc = Reductions::reduceAllMinLoc(data, &cell_ids);
  auto max_loc = Reductions::reduceAllMaxLoc(data, &cell_ids);

  // result is the same on all processors, only print once
  if (mesh_->getComm()->getRank() == 0) {
    std::cerr << "Printing min/max of " << description << " on region " << region_name << std::endl;
    std::cerr << std::fixed << description << " min = " << min_loc.val << " in cell " << min_loc.loc
              << std::endl;
    std::cerr << std::fixed << description << " max = " << max_loc.val << " in cell " << max_loc.loc
              << std::endl;
  }
}

} // namespace Amanzi
