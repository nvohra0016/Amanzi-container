/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

/*
  Transport PK

*/

#include <iostream>
#include <cstdlib>

// TPLs
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterXMLFileReader.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include <UnitTest++.h>

// Amanzi::Mesh
#include "Mesh.hh"
#include "MeshAudit.hh"
#include "MeshExtractedManifold.hh"
#ifdef HAVE_MESH_MOAB
#include "Mesh_MOAB.hh"
#endif
#ifdef HAVE_MESH_MSTK
#include "Mesh_MSTK.hh"
#endif
#include "Mesh_simple.hh"

using namespace Amanzi;
using namespace Amanzi::AmanziMesh;

/* **************************************************************** */
template<typename Audit_type=MeshAudit, typename Mesh_type=Mesh>
void RunTest(const std::string regname, int* cells, int* edges, int nframeworks=3) {

  Comm_ptr_type comm = Amanzi::getDefaultComm();

  // read parameter list
  std::string xmlFileName = "test/mesh_extracted_fracture.xml";
  Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::getParametersFromXmlFile(xmlFileName);

  std::string setname(regname);

  Teuchos::ParameterList region_list = plist->get<Teuchos::ParameterList>("regions");
  auto gm = Teuchos::rcp(new Amanzi::AmanziGeometry::GeometricModel(3, region_list, *comm));

  auto mesh_list = Teuchos::sublist(plist, "mesh", false);
  mesh_list->set<bool>("request edges", true);

  for (int i = 0; i < nframeworks; ++i) {
    Teuchos::RCP<MeshFramework> mesh3D;
    std::string framework = "none";
    if (i == 0) {
#ifdef HAVE_MESH_MSTK
      std::cout << "\nMesh framework: MSTK (" << regname << ")\n";
      // mesh3D = Teuchos::rcp(new Mesh_MSTK(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 10, 10, 10, comm, gm, mesh_list, true, true));
      mesh3D = Teuchos::rcp(new Mesh_MSTK("test/mesh_extracted_fracture.exo", comm, gm, mesh_list));
      framework = "MSTK";
#endif
    } else if (i == 1 && comm->getSize() == 1) {
#if 0
      // SIMPLE cannot be extracted from since it does not support full edge
      // topological relationships
      std::cout << "\nMesh framework: simple\n";
      mesh3D = Teuchos::rcp(new Mesh_simple(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 10, 10, 10, comm, gm, mesh_list));
      framework = "SIMPLE";
#endif
    } else if (i == 2) {
#ifdef HAVE_MESH_MOAB
      if (comm->getSize() > 1) continue;
      std::cout << "\nMesh framework: MOAB\n";
      mesh3D = Teuchos::rcp(new Mesh_MOAB("test/mesh_extracted_fracture.exo", comm, gm, mesh_list, true, true));
      framework = "MOAB";
#endif
    }
    if (mesh3D == Teuchos::null) continue;

    // extract fractures mesh
    auto mesh3D_cache_dev = Teuchos::rcp(new Mesh(mesh3D, Teuchos::rcp(new AmanziMesh::MeshAlgorithms()), Teuchos::null));
    auto mesh3D_cache = onMemSpace<MemSpace_kind::HOST>(mesh3D_cache_dev);
    Teuchos::RCP<MeshFramework> mesh = Teuchos::rcp(new MeshExtractedManifold(
              mesh3D_cache,
              setname, AmanziMesh::Entity_kind::FACE, comm, gm, plist));

    MeshMaps maps;
    maps.initialize(*mesh);

    int ncells_tmp = mesh->getNumEntities(AmanziMesh::Entity_kind::CELL,
            AmanziMesh::Parallel_kind::OWNED);
    int nfaces_tmp = mesh->getNumEntities(AmanziMesh::Entity_kind::FACE,
            AmanziMesh::Parallel_kind::OWNED);
    int mfaces_tmp = maps.getNBoundaryFaces(AmanziMesh::Parallel_kind::OWNED);

    int ncells(ncells_tmp), nfaces(nfaces_tmp), mfaces(mfaces_tmp);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &ncells_tmp, &ncells);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &nfaces_tmp, &nfaces);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, (int)1, (int*)&mfaces_tmp, (int*)&mfaces);

    std::cout << i << " pid=" << comm->getRank() << " cells: " << ncells
              << " faces: " << nfaces << " bnd faces: " << mfaces << std::endl;
    CHECK(cells[i] == ncells);
    CHECK(edges[i] == mfaces);

    auto plist_tmp = Teuchos::rcp(new Teuchos::ParameterList);
    plist_tmp->set<bool>("natural map ordering", true);

    Teuchos::RCP<Mesh_type> mesh_cache;
    if constexpr(std::is_same_v<Mesh_type, Mesh>) {
      mesh_cache = Teuchos::rcp(new Mesh(mesh, Teuchos::rcp(new AmanziMesh::MeshAlgorithms()), plist_tmp));
      AmanziMesh::cacheAll(*mesh_cache);
    } else {
      auto mesh_cache_dev = Teuchos::rcp(new Mesh(mesh, Teuchos::rcp(new AmanziMesh::MeshAlgorithms()), plist_tmp));
      mesh_cache = onMemSpace<MemSpace_kind::HOST>(mesh_cache_dev);
    }

    // verify mesh
    Audit_type audit(mesh_cache);
    int ok = audit.Verify();
    CHECK(ok == 0);
  }
}


TEST(MESH_EXTRACTED_FRACTURE_NETWORK2_HOST) {
  int cells[3] = {108, 200, 0};
  int edges[3] = {48, 0, 0};
  RunTest<MeshAuditHost,MeshHost>("fractures-two", cells, edges);
}

TEST(MESH_EXTRACTED_SURFACE_HOST) {
  int cells[3] = {9, 25, 0};
  int edges[3] = {12, 0, 0};
  RunTest<MeshAuditHost,MeshHost>("Left side", cells, edges);
}

TEST(MESH_EXTRACTED_FRACTURE_NETWORK3_HOST) {
  int cells[3] = {108, 200, 0};
  int edges[3] = {72, 0, 0};
  RunTest<MeshAuditHost,MeshHost>("fractures-three", cells, edges);
}


TEST(MESH_EXTRACTED_FRACTURE_NETWORK2) {
  int cells[3] = {108, 200, 0};
  int edges[3] = {48, 0, 0};
  RunTest("fractures-two", cells, edges, 1);
}

TEST(MESH_EXTRACTED_SURFACE) {
  int cells[3] = {9, 25, 0};
  int edges[3] = {12, 0, 0};
  RunTest("Left side", cells, edges, 1);
}

TEST(MESH_EXTRACTED_FRACTURE_NETWORK3) {
  int cells[3] = {108, 200, 0};
  int edges[3] = {72, 0, 0};
  RunTest("fractures-three", cells, edges, 1);
}
