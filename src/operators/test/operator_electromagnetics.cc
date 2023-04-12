/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

/*
  Operators

*/

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// TPLs
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_ParameterXMLFileReader.hpp"
#include "UnitTest++.h"

// Amanzi
#include "GMVMesh.hh"
#include "LinearOperatorPCG.hh"
#include "MeshFactory.hh"
#include "Tensor.hh"

// Amanzi::Operators
#include "OperatorDefs.hh"
#include "PDE_Accumulation.hh"
#include "PDE_Electromagnetics.hh"

#include "AnalyticElectromagnetics01.hh"
#include "AnalyticElectromagnetics02.hh"
#include "AnalyticElectromagnetics03.hh"
#include "Verification.hh"

/* *****************************************************************
 * TBW
 * **************************************************************** */
template <class Analytic>
void
CurlCurl(double c_t,
         int nx,
         double tolerance,
         bool initial_guess,
         const std::string& disc_method = "mfd: default")
{
  using namespace Teuchos;
  using namespace Amanzi;
  using namespace Amanzi::AmanziMesh;
  using namespace Amanzi::AmanziGeometry;
  using namespace Amanzi::Operators;

  auto comm = Amanzi::getDefaultComm();
  int MyPID = comm->getRank();

  if (MyPID == 0)
    std::cout << "\nTest: Curl-curl operator, tol=" << tolerance << "  method=" << disc_method
              << std::endl;

  // read parameter list
  std::string xmlFileName = "test/operator_electromagnetics.xml";
  ParameterXMLFileReader xmlreader(xmlFileName);
  ParameterList plist = xmlreader.getParameters();

  // create a MSTK mesh framework
  ParameterList region_list = plist.sublist("regions");
  Teuchos::RCP<GeometricModel> gm = Teuchos::rcp(new GeometricModel(3, region_list, *comm));

  MeshFactory meshfactory(comm, gm);
  meshfactory.set_preference(Preference({ Framework::MSTK }));

  bool request_faces(true), request_edges(true);
  RCP<const Mesh> mesh;
  if (nx > 0)
    mesh =
      meshfactory.create(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, nx, nx, nx, request_faces, request_edges);
  else
    mesh = meshfactory.create("test/hex_split_faces5.exo", request_faces, request_edges);

  // create resistivity coefficient
  double time = 1.0;
  Analytic ana(mesh);
  WhetStone:Tensor<> Kc(3, 2);

  Teuchos::RCP<std::vector<WhetStone:Tensor<>>> K =
    Teuchos::rcp(new std::vector<WhetStone:Tensor<>>());
  int ncells_owned = mesh->getNumEntities(AmanziMesh::Entity_kind::CELL, AmanziMesh::Parallel_kind::OWNED);

  for (int c = 0; c < ncells_owned; c++) {
    const AmanziGeometry::Point& xc = mesh->getCellCentroid(c)
    Kc = ana.Tensor(xc, time);
    K->push_back(Kc);
  }

  // create boundary data
  int nedges_owned = mesh->getNumEntities(AmanziMesh::Entity_kind::EDGE, AmanziMesh::Parallel_kind::OWNED);
  int nedges_wghost = mesh->getNumEntities(AmanziMesh::Entity_kind::EDGE, AmanziMesh::Parallel_kind::ALL);
  int nfaces_wghost = mesh->getNumEntities(AmanziMesh::Entity_kind::FACE, AmanziMesh::Parallel_kind::ALL);

  Teuchos::RCP<BCs> bc = Teuchos::rcp(new BCs(mesh, AmanziMesh::Entity_kind::EDGE, WhetStone::DOF_Type::SCALAR));
  std::vector<int>& bc_model = bc->bc_model();
  std::vector<double>& bc_value = bc->bc_value();

  std::vector<int> edirs;
  AmanziMesh::Entity_ID_List cells, edges;

  for (int f = 0; f < nfaces_wghost; ++f) {
    const AmanziGeometry::Point& xf = mesh->getFaceCentroid(f)

    if (fabs(xf[0]) < 1e-6 || fabs(xf[0] - 1.0) < 1e-6 || fabs(xf[1]) < 1e-6 ||
        fabs(xf[1] - 1.0) < 1e-6 || fabs(xf[2]) < 1e-6 || fabs(xf[2] - 1.0) < 1e-6) {
      mesh->getFaceEdgesAndDirs(f, edges, &edirs);
      int nedges = edges.size();
      for (int i = 0; i < nedges; ++i) {
        int e = edges[i];
        double len = mesh->getEdgeLength(e)
        const AmanziGeometry::Point& tau = mesh->getEdgeVector(e)
        const AmanziGeometry::Point& xe = mesh.getEdgeCentroid(e);

        bc_model[e] = OPERATOR_BC_DIRICHLET;
        bc_value[e] = (ana.electric_exact(xe, time) * tau) / len;
      }
    }
  }

  // create electromagnetics operator
  Teuchos::ParameterList olist = plist.sublist("PK operator").sublist("electromagnetics operator");
  olist.set<std::string>("discretization primary", disc_method);
  Teuchos::RCP<PDE_Electromagnetics> op_curlcurl =
    Teuchos::rcp(new PDE_Electromagnetics(olist, mesh));
  op_curlcurl->SetBCs(bc, bc);
  const CompositeVectorSpace& cvs = op_curlcurl->global_operator()->DomainMap();

  // create source for a manufactured solution.
  CompositeVector source(cvs);
  Epetra_MultiVector& src = *source.ViewComponent("edge");
  source.PutScalarMasterAndGhosted(0.0);

  for (int c = 0; c < ncells_owned; c++) {
    mesh->getCellEdges(c, edges);
    int nedges = edges.size();
    double vol = 3.0 * mesh->cell_volume(c) / nedges;

    for (int n = 0; n < nedges; ++n) {
      int e = edges[n];
      double len = mesh->getEdgeLength(e)
      const AmanziGeometry::Point& tau = mesh->getEdgeVector(e)
      const AmanziGeometry::Point& xe = mesh.getEdgeCentroid(e);

      src[0][e] += (ana.source_exact(xe, time) * tau) / len * vol;
    }
  }
  source.GatherGhostedToMaster("edge");

  // set up initial guess for a time-dependent problem
  CompositeVector solution(cvs);
  Epetra_MultiVector& sol = *solution.ViewComponent("edge");

  sol.putScalar(0.0);
  if (initial_guess) {
    for (int e = 0; e < nedges_owned; e++) {
      double len = mesh->getEdgeLength(e)
      const AmanziGeometry::Point& tau = mesh->getEdgeVector(e)
      const AmanziGeometry::Point& xe = mesh.getEdgeCentroid(e);

      sol[0][e] = (ana.electric_exact(xe, time) * tau) / len;
    }
  }

  // set up the diffusion operator
  op_curlcurl->SetTensorCoefficient(K);
  op_curlcurl->UpdateMatrices();

  // Add an accumulation term.
  CompositeVector phi(cvs);
  phi.putScalar(c_t);

  Teuchos::RCP<Operator> global_op = op_curlcurl->global_operator();
  Teuchos::RCP<PDE_Accumulation> op_acc =
    Teuchos::rcp(new PDE_Accumulation(AmanziMesh::Entity_kind::EDGE, global_op));

  double dT = 1.0;
  op_acc->AddAccumulationDelta(solution, phi, phi, dT, "edge");

  // BCs, sources, and assemble
  op_curlcurl->ApplyBCs(true, true, true);
  global_op->SymbolicAssembleMatrix();
  global_op->AssembleMatrix();
  global_op->UpdateRHS(source, false);

  ParameterList slist = plist.sublist("preconditioners").sublist("Hypre AMG");
  global_op->InitializePreconditioner(slist);
  global_op->UpdatePreconditioner();

  // Test SPD properties of the matrix and preconditioner.
  VerificationCV ver(global_op);
  ver.CheckMatrixSPD(true, true);
  ver.CheckPreconditionerSPD(1e-12, true, true);

  // Solve the problem.
  ParameterList lop_list = plist.sublist("solvers").sublist("default").sublist("pcg parameters");
  AmanziSolvers::LinearOperatorPCG<Operator, CompositeVector, CompositeVectorSpace> solver(
    global_op, global_op);
  solver.Init(lop_list);

  CompositeVector& rhs = *global_op->rhs();
  int ierr = solver.ApplyInverse(rhs, solution);

  ver.CheckResidual(solution, 1.0e-10);

  int num_itrs = solver.num_itrs();
  CHECK(num_itrs < 100);

  if (MyPID == 0) {
    std::cout << "electric solver (pcg): ||r||=" << solver.residual()
              << " itr=" << solver.num_itrs() << " code=" << solver.returned_code() << std::endl;
  }

  // compute electric error
  Epetra_MultiVector& E = *solution.ViewComponent("edge", true);
  double enorm, el2_err, einf_err;
  ana.ComputeEdgeError(E, time, enorm, el2_err, einf_err);

  if (MyPID == 0) {
    el2_err /= enorm;
    printf("L2(e)=%12.9f  Inf(e)=%9.6f  itr=%3d  size=%d\n",
           el2_err,
           einf_err,
           solver.num_itrs(),
           rhs.GlobalLength());

    CHECK(el2_err < tolerance);
  }
}


TEST(CURL_CURL_LINEAR)
{
  CurlCurl<AnalyticElectromagnetics01>(1.0e-5, 0, 1e-4, false);
  CurlCurl<AnalyticElectromagnetics01>(1.0e-5, 0, 1e-4, false, "mfd: generalized");
}

TEST(CURL_CURL_NONLINEAR)
{
  CurlCurl<AnalyticElectromagnetics02>(1.0e-1, 0, 2e-1, false);
}

TEST(CURL_CURL_TIME_DEPENDENT)
{
  CurlCurl<AnalyticElectromagnetics03>(1.0, 0, 2e-3, true);
  CurlCurl<AnalyticElectromagnetics03>(1.0, 0, 2e-3, true, "mfd: generalized");
}
