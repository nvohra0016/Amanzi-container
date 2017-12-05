/*
  Flow PK

  Copyright 2010-201x held jointly by LANS/LANL, LBNL, and PNNL. 
  Amanzi is released under the three-clause BSD License. 
  The terms of use and "as is" disclaimer for this license are 
  provided in the top-level COPYRIGHT file.

  Author: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// TPLs
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "UnitTest++.h"

// Amanzi
#include "GMVMesh.hh"
#include "MeshFactory.hh"
#include "MeshAudit.hh"
//#include "MeshInfo.hh"
#include "State.hh"

// Flow
#include "Darcy_PK.hh"


/* *********************************************************************
* Two tests with different time step controllers.
********************************************************************* */
void RunTestMarshak(std::string controller) {
  using namespace Teuchos;
  using namespace Amanzi;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::AmanziGeometry;
  using namespace Amanzi::Flow;

  Epetra_MpiComm comm(MPI_COMM_WORLD);
  int MyPID = comm.MyPID();
  if (MyPID == 0) std::cout << "\nTest: 2D well model: " << controller << std::endl;

  // read parameter list
  std::string xmlFileName = controller;
  Teuchos::RCP<ParameterList> plist = Teuchos::getParametersFromXmlFile(xmlFileName);

  // create an MSTK mesh framework
  ParameterList regions_list = plist->get<Teuchos::ParameterList>("regions");
  Teuchos::RCP<Amanzi::AmanziGeometry::GeometricModel> gm =
      Teuchos::rcp(new Amanzi::AmanziGeometry::GeometricModel(2, regions_list, &comm));

  FrameworkPreference pref;
  pref.clear();
  pref.push_back(MSTK);
  pref.push_back(STKMESH);

  MeshFactory meshfactory(&comm);
  meshfactory.preference(pref);
  RCP<const Mesh> mesh = meshfactory(-10.0, -5.0, 10.0, 0.0, 101, 50, gm);

  std::string filename;
  // Teuchos::ParameterList mesh_info_list;
  // filename = controller.replace(controller.size()-4, 4, "_mesh");
  // mesh_info_list.set<std::string>("filename", filename);
  // Teuchos::RCP<Amanzi::MeshInfo> mesh_info = Teuchos::rcp(new Amanzi::MeshInfo(mesh_info_list, &comm));
  // mesh_info->WriteMeshCentroids(*mesh);

  // create a simple state and populate it
  Amanzi::VerboseObject::hide_line_prefix = true;

  Teuchos::ParameterList state_list = plist->sublist("state");
  RCP<State> S = rcp(new State(state_list));
  S->RegisterDomainMesh(rcp_const_cast<Mesh>(mesh));

  Teuchos::RCP<Darcy_PK> DPK = Teuchos::rcp(new Darcy_PK(plist, "flow", S));
  DPK->Setup(S.ptr());
  S->Setup();
  S->InitializeFields();

  // modify the default state for the problem at hand
  // -- permeability

  std::string passwd("flow"); 

  Epetra_MultiVector& K = *S->GetFieldData("permeability", passwd)->ViewComponent("cell", false);
  double diff_in_perm = 0.;
  if (!S->GetField("permeability")->initialized()){
    for (int c = 0; c < K.MyLength(); c++) {
      const AmanziGeometry::Point xc = mesh->cell_centroid(c);
      K[0][c] = 0.1 + std::sin(xc[0]) * 0.02;
      K[1][c] = 2.0 + std::cos(xc[1]) * 0.4;
    }
    S->GetField("permeability", "flow")->set_initialized();
  } else{
    for (int c = 0; c < K.MyLength(); c++) {
      const AmanziGeometry::Point xc = mesh->cell_centroid(c);
      diff_in_perm += abs(K[0][c] - (0.1 + std::sin(xc[0]) * 0.02)) + 
        abs(K[1][c] - (2.0 + std::cos(xc[1]) * 0.4));
    }
    std::cout<<"diff_in_perm "<<diff_in_perm<<"\n";
    CHECK(diff_in_perm < 1.0e-12);
  }
    

  // -- fluid density and viscosity
  *S->GetScalarData("fluid_density", passwd) = 1.0;
  S->GetField("fluid_density", "flow")->set_initialized();

  *S->GetScalarData("fluid_viscosity", passwd) = 1.0;
  S->GetField("fluid_viscosity", "flow")->set_initialized();

  // -- gravity
  Epetra_Vector& gravity = *S->GetConstantVectorData("gravity", "state");
  gravity[1] = -1.0;
  S->GetField("gravity", "state")->set_initialized();

  // -- storativity
  S->GetFieldData("specific_storage", passwd)->PutScalar(0.1);
  S->GetField("specific_storage", "flow")->set_initialized();

  // initialize the Darcy process kernel
  DPK->Initialize(S.ptr());

  filename = controller.replace(controller.size()-4, 4, "_flow2D.gmv");

  // transient solution
  double t_old(0.0), t_new, dt(0.5);
  for (int n = 0; n < 10; n++) {
    t_new = t_old + dt;

    DPK->AdvanceStep(t_old, t_new);
    DPK->CommitStep(t_old, t_new, S);

    t_old = t_new;

    if (MyPID == 0) {
      const Epetra_MultiVector& p = *S->GetFieldData("pressure")->ViewComponent("cell");
      //GMV::open_data_file(*mesh, (std::string)"flow2D.gmv");
      GMV::open_data_file(*mesh, filename);
      GMV::start_data();
      GMV::write_cell_data(p, 0, "pressure");
      GMV::close_data_file();
    }

    dt = DPK->get_dt();
  }
}


TEST(FLOW_2D_DARCY_WELL_STANDARD) {
  RunTestMarshak("test/flow_darcy_well.xml");
}

TEST(FLOW_2D_DARCY_WELL_HETE_PERM) {
  RunTestMarshak("test/flow_darcy_well_hete_perm.xml");
}

TEST(FLOW_2D_DARCY_WELL_ADAPRIVE) {
  RunTestMarshak("test/flow_darcy_well_adaptive.xml");
}


/* **************************************************************** */

void Run_3D_DarcyWell(std::string controller) {

  using namespace Teuchos;
  using namespace Amanzi;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::AmanziGeometry;
  using namespace Amanzi::Flow;

  Epetra_MpiComm comm(MPI_COMM_WORLD);
  int MyPID = comm.MyPID();
  if (MyPID == 0) std::cout << "Test: 3D Darcy flow, two wells" << std::endl;

  // read parameter list
  std::string xmlFileName = controller;
  Teuchos::RCP<ParameterList> plist = Teuchos::getParametersFromXmlFile(xmlFileName);

  // create an MSTK mesh framework
  ParameterList regions_list = plist->get<Teuchos::ParameterList>("regions");
  Teuchos::RCP<Amanzi::AmanziGeometry::GeometricModel> gm =
      Teuchos::rcp(new Amanzi::AmanziGeometry::GeometricModel(3, regions_list, &comm));

  FrameworkPreference pref;
  pref.clear();
  pref.push_back(MSTK);
  pref.push_back(STKMESH);

  MeshFactory meshfactory(&comm);
  meshfactory.preference(pref);
  RCP<const Mesh> mesh = meshfactory(-10.0, -1.0, -5.0, 10.0, 1.0, 0.0, 101, 10, 25, gm);

  // create a simple state and populate it
  Amanzi::VerboseObject::hide_line_prefix = true;

  Teuchos::ParameterList state_list = plist->sublist("state");
  RCP<State> S = rcp(new State(state_list));
  S->RegisterDomainMesh(rcp_const_cast<Mesh>(mesh));

  Teuchos::RCP<Darcy_PK> DPK = Teuchos::rcp(new Darcy_PK(plist, "flow", S));
  DPK->Setup(S.ptr());
  S->Setup();
  S->InitializeFields();

  // modify the default state for the problem at hand
  // -- permeability
  std::string passwd("flow"); 
  Epetra_MultiVector& K = *S->GetFieldData("permeability", passwd)->ViewComponent("cell", false);
  
  for (int c = 0; c < K.MyLength(); c++) {
    K[0][c] = 0.1;
    K[1][c] = 2.0;
    K[2][c] = 2.0;
  }
  S->GetField("permeability", "flow")->set_initialized();

  // -- fluid density and viscosity
  *S->GetScalarData("fluid_density", passwd) = 1.0;
  S->GetField("fluid_density", "flow")->set_initialized();

  *S->GetScalarData("fluid_viscosity", passwd) = 1.0;
  S->GetField("fluid_viscosity", "flow")->set_initialized();

  // -- gravity
  Epetra_Vector& gravity = *S->GetConstantVectorData("gravity", "state");
  gravity[2] = -1.0;
  S->GetField("gravity", "state")->set_initialized();

  // -- storativity
  S->GetFieldData("specific_storage", passwd)->PutScalar(0.1);
  S->GetField("specific_storage", "flow")->set_initialized();

  // initialize the Darcy process kernel
  DPK->Initialize(S.ptr());

  std::string filename = controller.replace(controller.size()-4, 4, "_flow3D.gmv");

  // transient solution
  double t_old(0.0), t_new, dt(0.5);
  for (int n = 0; n < 2; n++) {
    t_new = t_old + dt;

    DPK->AdvanceStep(t_old, t_new);
    DPK->CommitStep(t_old, t_new, S);

    t_old = t_new;

    if (MyPID == 0) {
      const Epetra_MultiVector& p = *S->GetFieldData("pressure")->ViewComponent("cell");
      GMV::open_data_file(*mesh, filename);
      GMV::start_data();
      GMV::write_cell_data(p, 0, "pressure");
      GMV::close_data_file();
    }
  }
}


// TEST(FLOW_3D_DARCY_WELL) {
//    Run_3D_DarcyWell("test/flow_darcy_well_3D.xml");
// }


TEST(FLOW_3D_DARCY_PEACEMAN_WELL) {
  // Run_3D_DarcyWell("test/flow_darcy_well_peaceman_3D.xml");
  using namespace Teuchos;
  using namespace Amanzi;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::AmanziGeometry;
  using namespace Amanzi::Flow;

  Epetra_MpiComm comm(MPI_COMM_WORLD);
  int MyPID = comm.MyPID();
  if (MyPID == 0) std::cout << "Test: 3D Darcy flow, one well" << std::endl;

  // read parameter list
  std::string xmlFileName = "test/flow_darcy_1well_peaceman_3D.xml";
  Teuchos::RCP<ParameterList> plist = Teuchos::getParametersFromXmlFile(xmlFileName);

  // create an MSTK mesh framework
  ParameterList regions_list = plist->get<Teuchos::ParameterList>("regions");
  Teuchos::RCP<Amanzi::AmanziGeometry::GeometricModel> gm =
      Teuchos::rcp(new Amanzi::AmanziGeometry::GeometricModel(3, regions_list, &comm));

  FrameworkPreference pref;
  pref.clear();
  pref.push_back(MSTK);
  pref.push_back(STKMESH);

  MeshFactory meshfactory(&comm);
  meshfactory.preference(pref);
  RCP<const Mesh> mesh = meshfactory(-55.0, -55.0, -2., 55.0, 55.0, 0.0, 23, 23, 4, gm);

  // create a simple state and populate it
  Amanzi::VerboseObject::hide_line_prefix = true;

  Teuchos::ParameterList state_list = plist->sublist("state");
  RCP<State> S = rcp(new State(state_list));
  S->RegisterDomainMesh(rcp_const_cast<Mesh>(mesh));

  Teuchos::RCP<Darcy_PK> DPK = Teuchos::rcp(new Darcy_PK(plist, "flow", S));
  DPK->Setup(S.ptr());
  S->Setup();
  S->InitializeFields();

  // modify the default state for the problem at hand
  // -- permeability
  std::string passwd("flow"); 
  Epetra_MultiVector& K = *S->GetFieldData("permeability", passwd)->ViewComponent("cell", false);
  
  for (int c = 0; c < K.MyLength(); c++) {
    K[0][c] = 10.;
    K[1][c] = 10.;
    K[2][c] = 1.;
  }
  S->GetField("permeability", "flow")->set_initialized();

  // -- fluid density and viscosity
  *S->GetScalarData("fluid_density", passwd) = 1.0;
  S->GetField("fluid_density", "flow")->set_initialized();

  *S->GetScalarData("fluid_viscosity", passwd) = 1.0;
  S->GetField("fluid_viscosity", "flow")->set_initialized();

  // -- gravity
  Epetra_Vector& gravity = *S->GetConstantVectorData("gravity", "state");
  gravity[2] = -1.0;
  S->GetField("gravity", "state")->set_initialized();

  // -- storativity
  S->GetFieldData("specific_storage", passwd)->PutScalar(0.);
  S->GetField("specific_storage", "flow")->set_initialized();

  S->GetFieldData("porosity", "porosity")->PutScalar(1.);
  S->GetField("porosity", "porosity")->set_initialized();

  // initialize the Darcy process kernel
  DPK->Initialize(S.ptr());

  //std::string filename = controller.replace(controller.size()-4, 4, "_flow3D.gmv");
  std::string filename = "flow_darcy_well_peaceman_3D.gmv";

  // steady_state solution
  double t_old(0.0), t_new(0.5), dt(0.5);

  DPK->SolveFullySaturatedProblem(*S->GetFieldData("pressure", "flow"));

  t_old = t_new;
  const Epetra_MultiVector& p = *S->GetFieldData("pressure")->ViewComponent("cell");
  Epetra_MultiVector err_p(p), p_exact(p);

  double Q = -100.;
  double rw = 0.1;
  double k = 10.;
  double h = 5.;
  double pw = 10.;
  double depth = 2.5;

  int ncells = mesh->num_entities(AmanziMesh::CELL, AmanziMesh::OWNED);
  double err = 0.;
  double sol = 0.;
  for (int c = 0; c < ncells; c++){
    const AmanziGeometry::Point& xc = mesh->cell_centroid(c);
    double r = sqrt(xc[0]*xc[0] + xc[1]*xc[1]);
    double p_ex;
    p_ex = pw + gravity[2]*(xc[2] + depth);
    if (r > 1e-3) {
      p_ex = p_ex + Q/(2*M_PI*k*h)*(log(r) - log(rw));
    }else{
      p_ex = p[0][c];
    }

    p_exact[0][c] = p_ex;

    double vol = mesh->cell_volume(c);
    err += (p_ex - p[0][c]) * (p_ex - p[0][c])*vol;
    
    err_p[0][c] = abs(p_ex - p[0][c]);

    sol += p[0][c] * p[0][c] * vol;
  }

  // for (double r=55; r < 75; r+=0.25){
  //   double p_ex = pw + Q/(2*M_PI*k*h)*(log(r) - log(rw));
  //   std::cout<<r<<" "<<p_ex<<" "<<Q/(2*M_PI*k*h)<<" "<<log(rw)<<"\n";
  // }

  err = sqrt(err);
  sol = sqrt(sol);
  err = err/sol;
  std::cout<<"Error: "<<err<<"\n";

  CHECK(err < 0.02);

  if (MyPID == 0) {
    GMV::open_data_file(*mesh, filename);
    GMV::start_data();
    GMV::write_cell_data(p, 0, "pressure");
    GMV::write_cell_data(p_exact, 0, "exact");
    GMV::write_cell_data(err_p, 0, "error");
    if (S->HasField("well_index")){
      const Epetra_MultiVector& wi = *S->GetFieldData("well_index")->ViewComponent("cell");
      GMV::write_cell_data(wi, 0, "well_index");
    }
    GMV::close_data_file();
  }
}
