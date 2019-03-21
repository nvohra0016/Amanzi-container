#include <UnitTest++.h>

#include <fstream>

#include "../Mesh_MSTK.hh"
#include "MeshAudit.hh"

#include "AmanziMap.hh"
#include "AmanziComm.hh"
#include "AmanziVector.hh"


TEST(MSTK_EXTFACE_MAP_4P)
{

  int i, j, k, err, nc, nf, nv;
  std::vector<Amanzi::AmanziMesh::Entity_ID> faces(6), nodes(8);
  std::vector<int> facedirs(6);
  std::vector<Amanzi::AmanziGeometry::Point> ccoords(8), fcoords(4);

  auto comm = Amanzi::getDefaultComm();
  int rank = comm->getRank();
  int size = comm->getSize();
  CHECK_EQUAL(4,size);

  Teuchos::RCP<Amanzi::AmanziMesh::Mesh> mesh(new Amanzi::AmanziMesh::Mesh_MSTK("test/hex_3x3x3_sets.exo",comm));

  auto face_map = mesh->face_map(false);
  auto extface_map = mesh->exterior_face_map(false);

  auto all_to_extface_importer = mesh->exterior_face_importer();

  for (int f = extface_map->MinLID(); f <= extface_map->MaxLID(); f++)
    {
      int gid = extface_map->GID(f);
      int f2 = face_map->LID(gid); // f2 is local face id in face_map

      CHECK_EQUAL(face_map->GID(f2),gid);

      Amanzi::AmanziMesh::Entity_ID_List fcells;
      mesh->face_get_cells(f2, Amanzi::AmanziMesh::Parallel_type::OWNED, &fcells);
      CHECK_EQUAL(1,fcells.size());
    }

  Amanzi::Vector_type allvec(*face_map);
  Amanzi::Vector_type bdryvec(*extface_map);

  // Insert the GlobalID of each face offsetted by 3 into the allvec

  for (int f = face_map->MinLID(); f < face_map->MaxLID(); f++) 
      allvec[f] = face_map->GID(f)+3;

  bdryvec.Import(allvec, *all_to_extface_importer, Insert);

  // Check if the importer got the right values from allvec into bdryvec
  // by checking if the values in the bdryvec minus the offset correspond
  // to the correct global IDs.

  for (int f = extface_map->MinLID(); f < extface_map->MaxLID(); f++) 
    CHECK_EQUAL(extface_map->GID(f),bdryvec[f]-3);

  // Check if ghostmap contains only boundary faces

  auto extface_map_wghost = mesh->exterior_face_map(true);

  int nowned_bnd = extface_map->NumMyElements();
  int nnotowned_bnd = extface_map_wghost->NumMyElements() - nowned_bnd;

  std::vector<int> gl_id(nnotowned_bnd), pr_id(nnotowned_bnd), lc_id(nnotowned_bnd);

  for (int f=0; f<nnotowned_bnd; f++){
    gl_id[f] = extface_map_wghost->GID(f + nowned_bnd);
  }

  extface_map->RemoteIDList(nnotowned_bnd, gl_id.data(), pr_id.data(), lc_id.data());

  for (int f=0; f<nnotowned_bnd; f++){
    CHECK(pr_id[f] >= 0);
  }

}

