
#include "UnitTest++.h"
#include "Teuchos_RCP.hpp"
#include <Epetra_Comm.h>
#include <Epetra_MpiComm.h>
#include "Epetra_SerialComm.h"

#include "Mesh_maps_base.hh"
#include "Mesh_maps_simple.hh"

#include "State.hpp"

TEST(NUMBERING) {

#ifdef HAVE_MPI
  Epetra_MpiComm *comm = new Epetra_MpiComm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm *comm = new Epetra_SerialComm();
#endif

  // Create a single-cell mesh;
  Teuchos::RCP<Mesh_maps_base> mesh(new Mesh_maps_simple(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1, 1, 1, comm));
  
  State S(1,mesh);
  
  std::string gmvfile = "out.gmv";
  S.write_gmv(gmvfile);
  
  // Write node coordinates
  std::cout << "Node coordinates..." << std::endl;
  double x[3];
  for (unsigned int j = 0; j < 8; ++j) {
    std::cout << j << ":";
    mesh->node_to_coordinates(j, x, x+3);
    for (int i = 0; i < 3; ++i) std::cout << " " << x[i];
    std::cout << std::endl;
  }
  std::cout << std::endl;
  
  // Write face-node connectivity
  unsigned int fnode[8];
  std::cout << "Face node connectivity..." << std::endl;
  for (unsigned int j = 0; j < 6; ++j) {
    mesh->face_to_nodes(j, fnode, fnode+4);
    std::cout << j << ":";
    for (int i = 0; i < 4; ++i) std::cout << " " << fnode[i];
    std::cout << std::endl;
  }
  std::cout << std::endl;
  
  // Write cell-node connectivity
  unsigned int cnode[8];
  std::cout << "Cell node connectivity..." << std::endl;
  for (unsigned int j = 0; j < 1; ++j) {
    mesh->cell_to_nodes(j, cnode, cnode+8);
    std::cout << j << ":";
    for (int i = 0; i < 8; ++i) std::cout << " " << cnode[i];
    std::cout << std::endl;
  }
  std::cout << std::endl;
  
  // Write cell face-node connectivity
  unsigned int cface[6];
  int fdir[6];
  std::cout << "Cell " << 0 << " faces (relative orientation)..." << std::endl;
  mesh->cell_to_faces(0,cface,cface+6);
  mesh->cell_to_face_dirs(0,fdir,fdir+6);
  for (int j = 0; j < 6; ++j) std::cout << j << ": " << cface[j] << "(" << fdir[j] << ")" << std::endl;
  
}
