/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors:
*/

/*
  State

  Tests for state as a container of arbitrary data, and serves as documentation
  of how to add custom data.

  NOTE: this test passes if it compiles!
*/

// TPLs
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_ParameterXMLFileReader.hpp"
#include "Teuchos_RCP.hpp"
#include "UnitTest++.h"

// Amanzi
#include "errors.hh"
#include "MeshFactory.hh"

// Amanzi::State
#include "IO.hh"
#include "Data.hh"
#include "Data_Helpers.hh"
#include "State.hh"

struct MyPoint {
  double a;
  double b;
};

using MyPointList = std::vector<MyPoint>;

template<>
bool
Amanzi::Helpers::Initialize<MyPointList>(Teuchos::ParameterList& plist,
                           MyPointList& t)
{
  std::cout << "found it!" << std::endl;
  return true;
}

template<>
void
Amanzi::Helpers::WriteVis<MyPointList>(const Amanzi::Visualization& vis,
                          Teuchos::ParameterList& attrs,
                          const MyPointList& vec)
{}

template<>
void
Amanzi::Helpers::WriteCheckpoint<MyPointList>(const Amanzi::Checkpoint& chkp,
        Teuchos::ParameterList& attrs,
        const MyPointList& vec)
{}

template<>
void
Amanzi::Helpers::ReadCheckpoint<MyPointList>(const Amanzi::Checkpoint& chkp,
        Teuchos::ParameterList& attrs,
        MyPointList& vec)
{}


TEST(STATE_EXTENSIBILITY_CREATION)
{
  using namespace Amanzi;

  auto comm = Amanzi::getDefaultComm();
  Teuchos::ParameterList region_list;
  auto gm = Teuchos::rcp(new Amanzi::AmanziGeometry::GeometricModel(3, region_list, *comm));

  Amanzi::AmanziMesh::Preference pref;
  pref.clear();
  pref.push_back(Amanzi::AmanziMesh::Framework::MSTK);

  Amanzi::AmanziMesh::MeshFactory meshfactory(comm, gm);
  meshfactory.set_preference(pref);
  Teuchos::RCP<Amanzi::AmanziMesh::Mesh> m =
    meshfactory.create(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 8, 1, 1);

  std::string xmlFileName = "test/state_extensibility.xml";
  Teuchos::ParameterXMLFileReader xmlreader(xmlFileName);
  auto plist = Teuchos::parameterList(xmlreader.getParameters());

  State s(*Teuchos::sublist(plist, "state"));
  s.RegisterDomainMesh(m);
  s.Require<MyPointList>("my_points", Tags::DEFAULT, "my_points");
  s.GetRecordW("my_points", "my_points").set_io_vis();
  s.Setup();
  s.InitializeFields();

  Visualization vis(plist->sublist("visualization"));
  vis.createFiles();
  vis.write(s);

  Checkpoint chkp(plist->sublist("checkpoint"), s);
  chkp.write(s);
}
