#include <iostream>
#include <cstdlib>
#include <cmath>
#include "UnitTest++.h"
#include <vector>

#include "moab_mesh/Mesh_maps_moab.hh"
#include "mpc/State.hpp"
#include "transport/Transport_PK.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"



TEST(ADVANCE_WITH_MOAB) {

  using namespace std;
  using namespace Teuchos;

  /* create a MPC state with one component */
  RCP<Mesh_maps_moab> mesh = rcp( new Mesh_maps_moab( "../moab_mesh/test/hex_4x4x4_ss.exo", MPI_COMM_WORLD ) );

  int num_components = 3;
  State mpc_state( num_components, mesh );

  /* create a transport state from the MPC state */
  RCP<Transport_State>  TS = rcp( new Transport_State(mpc_state) );

  /* read the trasport parameter list */
  string xmlInFileName = "test/transport_input.xml";

  ParameterList parameter_list;
  updateParametersFromXmlFile(xmlInFileName, &parameter_list);

  /* initialize a transport process kernel from a transport state */
  Transport_PK  TPK(parameter_list, TS);

  /* create analytic Darcy flux */
  TS->analytic_total_component_concentration();
  TS->analytic_porosity();
  TS->analytic_darcy_flux();
  TS->analytic_water_saturation();
  TS->analytic_water_density();

  /* advance the state */
  TPK.advance();

  /* printing cell concentration */
  int  i, k;
  double  dT, T;
  RCP<Transport_State> TS_next = TPK.get_transport_state_next();

  RCP<Epetra_MultiVector> tcc      = TS->get_total_component_concentration();
  RCP<Epetra_MultiVector> tcc_next = TS_next->get_total_component_concentration();

  T = 0.0;
  cout << "Original state: T=" << T << endl;
  for( i=0; i<3; i++ ) {
     for( int k=0; k<20; k++ ) printf("%7.4f", (*tcc)[i][k]); 
     cout << endl;
  }
  dT = TPK.get_transport_dT();
  T += dT;

  cout << "New state: T=" << T << endl;
  for( i=0; i<3; i++ ) {
     for( int k=0; k<20; k++ ) printf("%7.4f", (*tcc_next)[i][k]); 
     cout << endl;
  }
 
  cout << "Dynamics of component 0 (3D simulaiton of 1D transport)" << endl;
  for( int i=0; i<20; i++ ) {
     *tcc = *tcc_next;

     TPK.advance();

     dT = TPK.get_transport_dT();
     T += dT;

     printf("T=%6.1f  C_0(x):", T);
     for( int k=0; k<20; k++ ) printf("%7.4f", (*tcc_next)[0][k]); cout << endl;
  }
  cout << "==================================================================" << endl << endl;
}
 


