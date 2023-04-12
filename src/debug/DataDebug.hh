/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors:
*/

#ifndef DATADEBUG_HH_
#define DATADEBUG_HH_

#include "Teuchos_RCP.hpp"
#include "Teuchos_VerboseObject.hpp"

#include "AmanziTypes.hh"
#include "Mesh.hh"

namespace Amanzi {

class DataDebug {
 public:
  explicit DataDebug(Teuchos::RCP<AmanziMesh::Mesh> mesh);
  ~DataDebug() {}

  void
  write_region_data(std::string& region_name, const Vector_type& data, std::string& description);
  void write_region_statistics(std::string& region_name,
                               const Vector_type& data,
                               std::string& description);

 private:
  Teuchos::RCP<AmanziMesh::Mesh> mesh_;
};

} // namespace Amanzi

#endif
