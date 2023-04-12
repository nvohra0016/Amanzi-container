/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon (ecoon@lanl.gov)
*/

/*
  State

  Utilities for I/O of State data.
*/

#ifndef STATE_IO_HH_
#define STATE_IO_HH_

#include <string>

#include "Mesh.hh"
#include "InputOutputHDF5.hh"

// Amanzi::State
#include "Mesh.hh"
#include "Checkpoint.hh"
#include "ObservationData.hh"
#include "State.hh"
#include "Visualization.hh"

namespace Amanzi {

// Checkpointing
double
ReadCheckpointInitialTime(const Comm_ptr_type& comm, std::string filename);

int
ReadCheckpointPosition(const Comm_ptr_type& comm, std::string filename);

void
ReadCheckpointObservations(const Comm_ptr_type& comm,
                           std::string filename,
                           Amanzi::ObservationData& obs_data);

// Writing mesh info to vis or checkpoint
template<class VisOrChkp>
void
WriteMeshCentroids(const std::string& domain, const AmanziMesh::Mesh& mesh, VisOrChkp& obj)
{
  int dim = mesh.getSpaceDimension();
  MultiVector_type centroids(mesh.getMap(AmanziMesh::Entity_kind::CELL, false), dim);
  auto mesh_on_host = AmanziMesh::onMemSpace<MemSpace_kind::HOST>(mesh);

  {
    auto centroids_hv = centroids.getLocalViewHost(Tpetra::Access::ReadWrite);
    for (int n = 0; n != centroids.getLocalLength(); ++n) {
      const AmanziGeometry::Point& xc = mesh.getCellCentroid(n);
      for (int i = 0; i != dim; ++i) {
        centroids_hv(n,i) = xc[i];
      }
    }
  }

  Teuchos::ParameterList attrs(domain+"_cell_centroids");
  Teuchos::Array<std::string> subfieldnames(dim);
  subfieldnames[0] = "x";
  subfieldnames[1] = "y";
  if (dim == 3) subfieldnames[2] = "z";
  attrs.set("subfieldnames", subfieldnames);
  attrs.set("location", AmanziMesh::Entity_kind::CELL);
  obj.write(attrs, centroids);
}

// // Reading from files
// void
// ReadVariableFromExodusII(Teuchos::ParameterList& plist, CompositeVector& var);

// Statistics
void
WriteStateStatistics(const State& S,
                     const VerboseObject& vo,
                     const Teuchos::EVerbosityLevel vl = Teuchos::VERB_HIGH);
void
WriteStateStatistics(const State& S);



} // namespace Amanzi

#endif
