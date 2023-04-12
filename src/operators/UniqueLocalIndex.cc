/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Kontantin Lipnikov (lipnikov@lanl.gov)
*/

/*
  Operators

  Helper functions for unique numbering of entries in mesh lists.

  Some mesh funtions return list of entities that are have no specific order.
  Helper function calculate unique position of entities using their global IDs.

*/

#ifndef UNIQUE_LOCAL_INDEX_HH_
#define UNIQUE_LOCAL_INDEX_HH_

#include <iterator>
#include <set>

#include "Teuchos_RCP.hpp"

#include "MeshFramework.hh"

namespace Amanzi {
namespace Operators {

/* ******************************************************************
 * Local index of cells for common internal face
 ****************************************************************** */
int
UniqueIndexFaceToCells(const AmanziMesh::Mesh& mesh, int f, int c)
{
  int pos = 0;
  AmanziMesh::Entity_ID_List cells;

  mesh.getFaceCells(f, AmanziMesh::Parallel_kind::ALL, cells);
  int ncells = cells.size();
  if (ncells > 1) {
    std::set<int> gids;
    const Epetra_BlockMap& cmap = mesh.getMap(AmanziMesh::Entity_kind::CELL,true);

    for (int i = 0; i < ncells; ++i) gids.insert(cmap.getGlobalElement(cells[i]));

    auto it = std::find(gids.begin(), gids.end(), cmap.getGlobalElement(c));
    pos = (it != gids.end()) ? std::distance(gids.begin(), it) : -1;
  }

  return pos;
}


} // namespace Operators
} // namespace Amanzi


#endif
