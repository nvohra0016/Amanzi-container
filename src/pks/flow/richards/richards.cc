/* -*-  mode: c++; c-default-style: "google"; indent-tabs-mode: nil -*- */

/* -------------------------------------------------------------------------
This is the flow component of the Amanzi code.
License: BSD
Authors: Neil Carlson (version 1)
         Konstantin Lipnikov (version 2) (lipnikov@lanl.gov)
         Ethan Coon (ATS version) (ecoon@lanl.gov)
------------------------------------------------------------------------- */

#include "bdf1_time_integrator.hh"
#include "flow_bc_factory.hh"
#include "eos_factory.hh"
#include "wrm_factory.hh"

#include "upwind_cell_centered.hh"
#include "upwind_arithmetic_mean.hh"
#include "upwind_total_flux.hh"
#include "upwind_gravity_flux.hh"

#include "richards.hh"

namespace Amanzi {
namespace Flow {

RegisteredPKFactory<Richards> Richards::reg_("richards flow");


// -------------------------------------------------------------
// Constructor
// -------------------------------------------------------------
Richards::Richards(Teuchos::ParameterList& flow_plist,
                   const Teuchos::RCP<State>& S,
                   const Teuchos::RCP<TreeVector>& solution) :
    flow_plist_(flow_plist) {

  solution_ = solution;

  // require fields

  // -- primary variable: pressure on both cells and faces, ghosted, with 1 dof
  std::vector<AmanziMesh::Entity_kind> locations2(2);
  std::vector<std::string> names2(2);
  std::vector<int> num_dofs2(2,1);
  locations2[0] = AmanziMesh::CELL;
  locations2[1] = AmanziMesh::FACE;
  names2[0] = "cell";
  names2[1] = "face";

  S->RequireField("pressure", "flow")->SetMesh(S->Mesh())->SetGhosted()
                    ->SetComponents(names2, locations2, num_dofs2);

  // -- secondary variables
  //   Nearly all of these will be defined on a single dof on cells.  To make
  //   this process easier, we set up two factories, one owned, one not, and
  //   use those to initialize the data factories.
  CompositeVectorFactory one_cell_owned_factory;
  one_cell_owned_factory.SetMesh(S->Mesh());
  one_cell_owned_factory.SetGhosted();
  one_cell_owned_factory.SetComponent("cell", AmanziMesh::CELL, 1);

  CompositeVectorFactory one_cell_factory;
  one_cell_factory.SetMesh(S->Mesh());
  one_cell_factory.SetGhosted();
  one_cell_factory.AddComponent("cell", AmanziMesh::CELL, 1);

  S->RequireField("darcy_flux", "flow")->SetMesh(S->Mesh())->SetGhosted()
                                ->SetComponent("face", AmanziMesh::FACE, 1);
  S->RequireField("darcy_velocity", "flow")->SetMesh(S->Mesh())->SetGhosted()
                                ->SetComponent("cell", AmanziMesh::CELL, 3);

  S->RequireField("saturation_liquid", "flow")->Update(one_cell_owned_factory);
  S->RequireField("density_liquid", "flow")->Update(one_cell_owned_factory);
  S->RequireField("molar_density_liquid", "flow")->Update(one_cell_owned_factory);
  S->RequireField("viscosity_liquid", "flow")->Update(one_cell_owned_factory);

  S->RequireField("saturation_gas", "flow")->Update(one_cell_owned_factory);
  S->RequireField("density_gas", "flow")->Update(one_cell_owned_factory);
  S->RequireField("molar_density_gas", "flow")->Update(one_cell_owned_factory);
  S->RequireField("mol_frac_gas", "flow")->Update(one_cell_owned_factory);

  // -- For now, we assume scalar permeability.  This will change.
  S->RequireField("permeability", "flow")->Update(one_cell_owned_factory);
  S->RequireScalar("atmospheric_pressure", "flow");

  // -- independent variables not owned by this PK
  S->RequireField("cell_volume")->Update(one_cell_factory);
  S->RequireField("porosity")->Update(one_cell_factory);
  S->RequireField("temperature")->Update(one_cell_factory);
  S->RequireGravity();

  // -- work vectors
  S->RequireField("relative_permeability", "flow")->SetMesh(S->Mesh())->SetGhosted()
                    ->SetComponents(names2, locations2, num_dofs2);
  S->RequireField("numerical_rel_perm", "flow")->SetMesh(S->Mesh())->SetGhosted()
                    ->SetComponents(names2, locations2, num_dofs2);
  S->GetField("numerical_rel_perm","flow")->set_io_vis(false);

  // abs perm tensor
  variable_abs_perm_ = false; // currently not implemented, but may eventually want a model
  int c_owned = S->Mesh()->num_entities(AmanziMesh::CELL, AmanziMesh::OWNED);
  K_ = Teuchos::rcp(new std::vector<WhetStone::Tensor>(c_owned));
  for (int c=0; c!=c_owned; ++c) {
    (*K_)[c].init(S->Mesh()->space_dimension(),1);
  }

  // constitutive relations
  // -- liquid eos
  Teuchos::ParameterList water_eos_plist = flow_plist_.sublist("Water EOS");
  FlowRelations::EOSFactory eos_factory;
  eos_liquid_ = eos_factory.createEOS(water_eos_plist);

  // -- gas eos
  Teuchos::ParameterList eos_gas_plist = flow_plist_.sublist("Vapor and Gas EOS");
  eos_gas_ = Teuchos::rcp(new FlowRelations::EOSVaporInGas(eos_gas_plist));

  // -- water retention models
  Teuchos::ParameterList wrm_plist = flow_plist_.sublist("Water Retention Models");
  // count the number of region-model pairs
  int wrm_count = 0;
  for (Teuchos::ParameterList::ConstIterator i=wrm_plist.begin(); i!=wrm_plist.end(); ++i) {
    if (wrm_plist.isSublist(wrm_plist.name(i))) {
      wrm_count++;
    } else {
      std::string message("Richards: water retention model contains an entry that is not a sublist.");
      Exceptions::amanzi_throw(message);
    }
  }

  // TODO: check and make sure all blocks have a WRM associated with it. --etc

  wrm_.resize(wrm_count);

  // instantiate the region-model pairs
  FlowRelations::WRMFactory wrm_factory;
  int iblock = 0;
  for (Teuchos::ParameterList::ConstIterator i=wrm_plist.begin(); i!=wrm_plist.end(); ++i) {
    Teuchos::ParameterList wrm_region_list = wrm_plist.sublist(wrm_plist.name(i));
    std::string region = wrm_region_list.get<std::string>("Region");
    wrm_[iblock] = Teuchos::rcp(new WRMRegionPair(region, wrm_factory.createWRM(wrm_region_list)));
    iblock++;
  }

  // boundary conditions
  Teuchos::ParameterList bc_plist = flow_plist_.sublist("boundary conditions", true);
  FlowBCFactory bc_factory(S->Mesh(), bc_plist);
  bc_pressure_ = bc_factory.CreatePressure();
  bc_flux_ = bc_factory.CreateMassFlux();

  // Relative permeability method
  string method_name = flow_plist_.get<string>("Relative permeability method", "Upwind with gravity");
  bool symmetric = false;
  if (method_name == "Upwind with gravity") {
    upwinding_ = Teuchos::rcp(new Operators::UpwindGravityFlux("flow",
            "relative_permeability", "numerical_rel_perm", K_));
    Krel_method_ = FLOW_RELATIVE_PERM_UPWIND_GRAVITY;
  } else if (method_name == "Cell centered") {
    upwinding_ = Teuchos::rcp(new Operators::UpwindCellCentered("flow",
            "relative_permeability", "numerical_rel_perm"));
    symmetric = true;
    Krel_method_ = FLOW_RELATIVE_PERM_CENTERED;
  } else if (method_name == "Upwind with Darcy flux") {
    upwinding_ = Teuchos::rcp(new Operators::UpwindTotalFlux("flow",
            "relative_permeability", "numerical_rel_perm", "darcy_flux"));
    Krel_method_ = FLOW_RELATIVE_PERM_UPWIND_DARCY_FLUX;
  } else if (method_name == "Arithmetic mean") {
    upwinding_ = Teuchos::rcp(new Operators::UpwindArithmeticMean("flow",
            "relative_permeability", "numerical_rel_perm"));
    Krel_method_ = FLOW_RELATIVE_PERM_ARITHMETIC_MEAN;
  }

  // operator for the diffusion terms
  Teuchos::ParameterList mfd_plist = flow_plist_.sublist("Diffusion");
  matrix_ = Teuchos::rcp(new Operators::MatrixMFD(mfd_plist, S->Mesh()));
  matrix_->SetSymmetryProperty(symmetric);
  matrix_->SymbolicAssembleGlobalMatrices();

  // preconditioner for the NKA system
  Teuchos::ParameterList mfd_pc_plist = flow_plist_.sublist("Diffusion PC");
  preconditioner_ = Teuchos::rcp(new Operators::MatrixMFD(mfd_pc_plist, S->Mesh()));
  preconditioner_->SetSymmetryProperty(symmetric);
  preconditioner_->SymbolicAssembleGlobalMatrices();
  preconditioner_->InitPreconditioner(mfd_pc_plist);
};


// -------------------------------------------------------------
// Initialize PK
// -------------------------------------------------------------
void Richards::initialize(const Teuchos::RCP<State>& S) {
  // initial timestep size
  dt_ = flow_plist_.get<double>("Initial time step", 1.);

  // initialize boundary conditions
  int nfaces = S->Mesh()->num_entities(AmanziMesh::FACE, AmanziMesh::USED);
  bc_markers_.resize(nfaces, Operators::MFD_BC_NULL);
  bc_values_.resize(nfaces, 0.0);

  // update face pressures as a hint?
  Teuchos::RCP<CompositeVector> pres = S->GetFieldData("pressure", "flow");
  DeriveFaceValuesFromCellValues_(S, pres);

  // declare secondary variables initialized, as they will be done by
  // the commit_state call
  S->GetField("saturation_liquid","flow")->set_initialized();
  S->GetField("density_liquid","flow")->set_initialized();
  S->GetField("molar_density_liquid","flow")->set_initialized();
  S->GetField("viscosity_liquid","flow")->set_initialized();

  S->GetField("saturation_gas","flow")->set_initialized();
  S->GetField("density_gas","flow")->set_initialized();
  S->GetField("molar_density_gas","flow")->set_initialized();
  S->GetField("mol_frac_gas","flow")->set_initialized();

  S->GetField("relative_permeability","flow")->set_initialized();
  S->GetField("darcy_flux", "flow")->set_initialized();
  S->GetField("darcy_velocity", "flow")->set_initialized();

  // rel perm is special -- if the mode is symmetric, it needs to be
  // initialized to 1
  S->GetFieldData("numerical_rel_perm","flow")->PutScalar(1.0);
  S->GetField("numerical_rel_perm","flow")->set_initialized();

  // absolute perm
  SetAbsolutePermeabilityTensor_(S);

  // operators
  matrix_->CreateMFDmassMatrices(K_.ptr());
  preconditioner_->CreateMFDmassMatrices(K_.ptr());

  // initialize the timesteppper
  solution_->set_data(pres);
  atol_ = flow_plist_.get<double>("Absolute error tolerance",1.0);
  rtol_ = flow_plist_.get<double>("Relative error tolerance",1.0);

  if (!flow_plist_.get<bool>("Strongly Coupled PK", false)) {
    // -- instantiate time stepper
    Teuchos::RCP<Teuchos::ParameterList> bdf1_plist_p =
      Teuchos::rcp(new Teuchos::ParameterList(flow_plist_.sublist("Time integrator")));
    time_stepper_ = Teuchos::rcp(new BDF1TimeIntegrator(this, bdf1_plist_p, solution_));
    time_step_reduction_factor_ = bdf1_plist_p->get<double>("time step reduction factor");

    // -- initialize time derivative
    Teuchos::RCP<TreeVector> solution_dot = Teuchos::rcp( new TreeVector(*solution_));
    solution_dot->PutScalar(0.0);

    // -- set initial state
    time_stepper_->set_initial_state(S->time(), solution_, solution_dot);
  }
};


// -----------------------------------------------------------------------------
// Update any secondary (dependent) variables given a solution.
//
//   After a timestep is evaluated (or at ICs), there is no way of knowing if
//   secondary variables have been updated to be consistent with the new
//   solution.
// -----------------------------------------------------------------------------
void Richards::commit_state(double dt, const Teuchos::RCP<State>& S) {
  // update secondary variables for IC
  UpdateSecondaryVariables_(S);

  // update the flux
  UpdatePermeabilityData_(S);
  Teuchos::RCP<const CompositeVector> rel_perm =
    S->GetFieldData("numerical_rel_perm");
  Teuchos::RCP<const CompositeVector> pres =
    S->GetFieldData("pressure");
  Teuchos::RCP<CompositeVector> darcy_flux =
    S->GetFieldData("darcy_flux", "flow");

  matrix_->CreateMFDstiffnessMatrices(*rel_perm);
  matrix_->DeriveFlux(*pres, darcy_flux);
  AddGravityFluxesToVector_(S, darcy_flux);
};


// -----------------------------------------------------------------------------
// Transfer operators -- ONLY COPIES POINTERS
// -----------------------------------------------------------------------------
void Richards::state_to_solution(const Teuchos::RCP<State>& S,
        const Teuchos::RCP<TreeVector>& solution) {
  solution->set_data(S->GetFieldData("pressure", "flow"));
};


// -----------------------------------------------------------------------------
// Transfer operators -- ONLY COPIES POINTERS
// -----------------------------------------------------------------------------
void Richards::solution_to_state(const Teuchos::RCP<TreeVector>& solution,
        const Teuchos::RCP<State>& S) {
  S->SetData("pressure", "flow", solution->data());
};


// -----------------------------------------------------------------------------
// Advance from state S to state S_next at time S.time + dt.
// -----------------------------------------------------------------------------
bool Richards::advance(double dt) {
  state_to_solution(S_next_, solution_);

  // take a bdf timestep
  double h = dt;
  try {
    dt_ = time_stepper_->time_step(h, solution_);
  } catch (Exceptions::Amanzi_exception &error) {
    std::cout << "Timestepper called error: " << error.what() << std::endl;
    if (error.what() == std::string("BDF time step failed")) {
      // try cutting the timestep
      dt_ = dt_*time_step_reduction_factor_;
      return true;
    } else {
      throw error;
    }
  }

  // commit the step as successful
  time_stepper_->commit_solution(h, solution_);
  solution_to_state(solution_, S_next_);
  commit_state(h,S_next_);

  return false;
};


// -----------------------------------------------------------------------------
// Update any diagnostic variables prior to vis (in this case velocity field).
// -----------------------------------------------------------------------------
void Richards::calculate_diagnostics(const Teuchos::RCP<State>& S) {
  // update the cell velocities
  Teuchos::RCP<CompositeVector> velocity = S->GetFieldData("darcy_velocity", "flow");
  Teuchos::RCP<const CompositeVector> flux = S->GetFieldData("darcy_flux");
  matrix_->DeriveCellVelocity(*flux, velocity);
};


// -----------------------------------------------------------------------------
// Use the physical rel perm (on cells) to update a work vector for rel perm.
//
//   This deals with upwinding, etc.
// -----------------------------------------------------------------------------
void Richards::UpdatePermeabilityData_(const Teuchos::RCP<State>& S) {
  // get the upwinding done
  upwinding_->Update(S.ptr());

  // patch up the BCs
  Teuchos::RCP<const CompositeVector> rel_perm = S->GetFieldData("relative_permeability");
  Teuchos::RCP<CompositeVector> num_rel_perm = S->GetFieldData("numerical_rel_perm", "flow");
  for (int f=0; f!=num_rel_perm->size("face"); ++f) {
    if (bc_markers_[f] != Operators::MFD_BC_NULL) {
      (*num_rel_perm)("face",f) = (*rel_perm)("face",f);
    }
  }

  // scale cells by n/visc
  Teuchos::RCP<const CompositeVector> n_liq = S->GetFieldData("molar_density_liquid");
  Teuchos::RCP<const CompositeVector> visc = S->GetFieldData("viscosity_liquid");
  int ncells = num_rel_perm->size("cell");
  for (int c=0; c!=ncells; ++c) {
    (*num_rel_perm)("cell",c) *= (*n_liq)("cell",c) / (*visc)("cell",c);
  }
};


// -----------------------------------------------------------------------------
// Interpolate pressure ICs on cells to ICs for lambda (faces).
// -----------------------------------------------------------------------------
void Richards::DeriveFaceValuesFromCellValues_(const Teuchos::RCP<State>& S,
        const Teuchos::RCP<CompositeVector>& pres) {
  AmanziMesh::Entity_ID_List cells;
  pres->ScatterMasterToGhosted("cell");

  int f_owned = pres->size("face");
  for (int f=0; f!=f_owned; ++f) {
    cells.clear();
    S->Mesh()->face_get_cells(f, AmanziMesh::USED, &cells);
    int ncells = cells.size();

    double face_value = 0.0;
    for (int n=0; n!=ncells; ++n) {
      face_value += (*pres)("cell",cells[n]);
    }
    (*pres)("face",f) = face_value / ncells;
  }
};


// -----------------------------------------------------------------------------
// Evaluate boundary conditions at the current time.
// -----------------------------------------------------------------------------
void Richards::UpdateBoundaryConditions_() {
  for (int n=0; n!=bc_markers_.size(); ++n) {
    bc_markers_[n] = Operators::MFD_BC_NULL;
    bc_values_[n] = 0.0;
  }

  Functions::BoundaryFunction::Iterator bc;
  for (bc=bc_pressure_->begin(); bc!=bc_pressure_->end(); ++bc) {
    int f = bc->first;
    bc_markers_[f] = Operators::MFD_BC_DIRICHLET;
    bc_values_[f] = bc->second;
  }

  for (bc=bc_flux_->begin(); bc!=bc_flux_->end(); ++bc) {
    int f = bc->first;
    bc_markers_[f] = Operators::MFD_BC_FLUX;
    bc_values_[f] = bc->second;
  }
};


// -----------------------------------------------------------------------------
// Add a boundary marker to owned faces.
// -----------------------------------------------------------------------------
void Richards::ApplyBoundaryConditions_(const Teuchos::RCP<State>& S,
        const Teuchos::RCP<CompositeVector>& pres) {
  int nfaces = pres->size("face");
  for (int f=0; f!=nfaces; ++f) {
    if (bc_markers_[f] == Operators::MFD_BC_DIRICHLET) {
      (*pres)("face",f) = bc_values_[f];
    }
  }
};


} // namespace
} // namespace
