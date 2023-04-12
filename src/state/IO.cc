/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon
*/

/*
  State

*/

#include <iostream>
#include <ostream>
#include <string>

#define BOOST_FILESYSTEM_NO_DEPRECATED
#include "boost/filesystem.hpp"
#include <boost/format.hpp>
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "exodusII.h"

// Amanzi
#include "Abbreviations.hh"
#include "CompositeVector.hh"
#include "DomainSet.hh"
#include "errors.hh"
#include "Mesh.hh"
#include "StringExt.hh"

#include "State.hh"
#include "IO.hh"
#include "EvaluatorPrimary.hh"

namespace Amanzi {



// -----------------------------------------------------------------------------
// Non-member function for checkpointing.
// -----------------------------------------------------------------------------
double
ReadCheckpointInitialTime(const Comm_ptr_type& comm, std::string filename)
{
  if (!Keys::ends_with(filename, ".h5")) {
    // new style checkpoint
    boost::filesystem::path filepath = boost::filesystem::path(filename) / "domain.h5";
    filename = filepath.string();
  }

  InputHDF5 checkpoint(comm, filename);

  // load the attributes
  double time(0.);
  checkpoint.read(Teuchos::ParameterList("time"), time);
  return time;
}


// -----------------------------------------------------------------------------
// Non-member function for checkpointing position.
// -----------------------------------------------------------------------------
int
ReadCheckpointPosition(const Comm_ptr_type& comm, std::string filename)
{
  if (!Keys::ends_with(filename, ".h5")) {
    // new style checkpoint
    boost::filesystem::path filepath = boost::filesystem::path(filename) / "domain.h5";
    filename = filepath.string();
  }

  InputHDF5 checkpoint(comm, filename);

  // load the attributes
  int pos = 0;
  checkpoint.read(Teuchos::ParameterList("position"), pos);
  return pos;
}


// // -----------------------------------------------------------------------------
// // Non-member function for checkpointing observations.
// // -----------------------------------------------------------------------------
// void
// ReadCheckpointObservations(const Comm_ptr_type& comm,
//                            std::string filename,
//                            Amanzi::ObservationData& obs_data)
// {
//   if (!Keys::ends_with(filename, ".h5")) {
//     // new style checkpoint
//     boost::filesystem::path filepath = boost::filesystem::path(filename) / "domain.h5";
//     filename = filepath.string();
//   }

//   InputHDF5 checkpoint(comm, filename);

//   // read observations
//   int nlabels, ndata(0), ndata_glb(0);
//   int* nobs;
//   char** tmp_labels;
//   double* tmp_data(NULL);

// az  checkpoint.readDataString(&tmp_labels, &nlabels, "obs_names");
//   if (nlabels > 0) { checkpoint.readAttrInt(&nobs, &nlabels, "obs_numbers"); }
//   for (int i = 0; i < nlabels; ++i) ndata_glb += 2 * nobs[i];
//   ndata = (comm->MyPID() == 0) ? ndata_glb : 0;
//   checkpoint.readDatasetReal(&tmp_data, ndata, "obs_values");

//   checkpoint.close_h5file();

//   // populated observations on root
//   if (comm->MyPID() == 0) {
//     int m(0);
//     Amanzi::ObservationData::DataQuadruple data_quad;

//     for (int i = 0; i < nlabels; ++i) {
//       std::vector<ObservationData::DataQuadruple>& od = obs_data[tmp_labels[i]];
//       for (int k = 0; k < nobs[i]; ++k) {
//         data_quad.time = tmp_data[m++];
//         data_quad.value = tmp_data[m++];
//         data_quad.is_valid = true;
//         od.push_back(data_quad);
//       }
//     }
//   }

//   // clean memory
//   for (int i = 0; i < nlabels; i++) free(tmp_labels[i]);
//   if (nlabels > 0) {
//     free(tmp_labels);
//     free(nobs);
//     if (tmp_data != NULL) free(tmp_data);
//   }
// }


// -----------------------------------------------------------------------------
// Non-member function for deforming the mesh after reading a checkpoint file
// that contains the vertex coordinate field (this is written by deformation pks)
//
// FIX ME: Refactor this to make the name more general.  Should align with a
// mesh name prefix or something, and the coordinates should be written by
// state in WriteCheckpoint if mesh IsDeformableMesh() --ETC
// -----------------------------------------------------------------------------
void
DeformCheckpointMesh(State& S, Key domain)
{
  Key vc_key = Keys::getKey(domain, "vertex_coordinates");
  if (S.HasRecord(vc_key, Tags::DEFAULT)) {
    // only deform mesh if vertex_coordinates field exists
    auto write_access_mesh = S.GetDeformableMesh(domain);

    // get vertex coordinates state
    const CompositeVector& vc = S.Get<CompositeVector>(vc_key, Tags::DEFAULT);
    vc.scatterMasterToGhosted("node");

    Amanzi::AmanziMesh::Entity_ID_List nodeids;
    AmanziGeometry::Point_List new_pos, final_pos;
    {
      auto vc_n = vc.viewComponent<Kokkos::HostSpace>("node", true);
      int dim = write_access_mesh->getSpaceDimension();
      Amanzi::AmanziGeometry::Point new_coords(dim);

      int nV = vc_n.extent(0);
      for (int n = 0; n != nV; ++n) {
        for (int k = 0; k != dim; ++k) new_coords[k] = vc_n(n,k);

        // push back for deform method
        nodeids.emplace_back(n);
        new_pos.emplace_back(new_coords);
      }
    }

    // deform the mesh
    if (Keys::starts_with(domain, "column"))
      AmanziMesh::deform(*write_access_mesh, asView(nodeids), asView(new_pos));
    else
      AmanziMesh::deform(*write_access_mesh, asView(nodeids), asView(new_pos));
  } else {
    Errors::Message msg;
    msg << "DeformCheckpointMesh: unable to deform mesh because field \"" << vc_key
        << "\" does not exist in state.";
    Exceptions::amanzi_throw(msg);
  }
}


// // -----------------------------------------------------------------------------
// // Reads cell-based varibles as attributes.
// // It recongnizes parallel and serial inputs.
// // -----------------------------------------------------------------------------
// void
// ReadVariableFromExodusII(Teuchos::ParameterList& plist, CompositeVector& var)
// {
//   Epetra_MultiVector& var_c = *var.ViewComponent("cell");
//   int nvectors = var_c.NumVectors();

//   std::string file_name = plist.get<std::string>("file");
//   std::vector<std::string> attributes =
//     plist.get<Teuchos::Array<std::string>>("attributes").toVector();

//   // open ExodusII file
//   auto comm = var.Comm();

//   if (comm->NumProc() > 1) {
//     int ndigits = (int)floor(log10(comm->NumProc())) + 1;
//     std::string fmt = boost::str(boost::format("%%s.%%d.%%0%dd") % ndigits);
//     file_name = boost::str(boost::format(fmt) % file_name % comm->NumProc() % comm->MyPID());
//   }

//   int CPU_word_size(8), IO_word_size(0), ierr;
//   float version;
//   int exoid = ex_open(file_name.c_str(), EX_READ, &CPU_word_size, &IO_word_size, &version);
//   if (comm->MyPID() == 0) {
//     printf(
//       "Trying file: %s ws=%d %d  id=%d\n", file_name.c_str(), CPU_word_size, IO_word_size, exoid);
//   }

//   // check if we have to use serial file
//   int fail = (exoid < 0) ? 1 : 0;
//   int fail_tmp(fail);
//   bool distributed_data(true);

//   comm->SumAll(&fail_tmp, &fail, 1);
//   if (fail == comm->NumProc()) {
//     Errors::Message msg("Rao is working on new data layout which we need to proceed.");
//     Exceptions::amanzi_throw(msg);

//     file_name = plist.get<std::string>("file");
//     distributed_data = false;
//     if (comm->MyPID() == 0) {
//       exoid = ex_open(file_name.c_str(), EX_READ, &CPU_word_size, &IO_word_size, &version);
//       printf("Opening file: %s ws=%d %d  id=%d\n",
//              file_name.c_str(),
//              CPU_word_size,
//              IO_word_size,
//              exoid);
//     }
//   } else if (fail > 0) {
//     Errors::Message msg("A few parallel Exodus files are missing, but not all.");
//     Exceptions::amanzi_throw(msg);
//   }

//   // read database parameters
//   if (comm->MyPID() == 0 || distributed_data) {
//     int dim, num_nodes, num_elem, num_elem_blk, num_node_sets, num_side_sets;
//     char title[MAX_LINE_LENGTH + 1];
//     ierr = ex_get_init(
//       exoid, title, &dim, &num_nodes, &num_elem, &num_elem_blk, &num_node_sets, &num_side_sets);

//     int* ids = (int*)calloc(num_elem_blk, sizeof(int));
//     ierr = ex_get_ids(exoid, EX_ELEM_BLOCK, ids);

//     // read number of variables
//     int num_vars;
//     auto obj_type = ex_var_type_to_ex_entity_type('e');
//     ierr = ex_get_variable_param(exoid, obj_type, &num_vars);
//     if (ierr < 0) printf("Exodus file has no variables.\n");

//     char* var_names[num_vars];
//     for (int i = 0; i < num_vars; i++) {
//       var_names[i] = (char*)calloc((MAX_STR_LENGTH + 1), sizeof(char));
//     }

//     obj_type = ex_var_type_to_ex_entity_type('e');
//     ierr = ex_get_variable_names(exoid, obj_type, num_vars, var_names);
//     if (ierr < 0) printf("Exodus file cannot read variable names.\n");

//     int var_index(-1), ncells;
//     for (int k = 0; k < nvectors; ++k) {
//       for (int i = 0; i < num_vars; i++) {
//         std::string tmp(var_names[i]);
//         if (tmp == attributes[k]) var_index = i + 1;
//       }
//       if (var_index < 0) printf("Exodus file has no variable \"%s\".\n", attributes[k].c_str());
//       printf("Variable \"%s\" has index %d.\n", attributes[k].c_str(), var_index);

//       // read variable with the k-th attribute
//       int offset = 0;
//       char elem_type[MAX_LINE_LENGTH + 1];
//       for (int i = 0; i < num_elem_blk; i++) {
//         int num_elem_this_blk, num_attr, num_nodes_elem;
//         ierr = ex_get_block(exoid,
//                             EX_ELEM_BLOCK,
//                             ids[i],
//                             elem_type,
//                             &num_elem_this_blk,
//                             &num_nodes_elem,
//                             0,
//                             0,
//                             &num_attr);

//         double* var_values = (double*)calloc(num_elem_this_blk, sizeof(double));
//         ierr =
//           ex_get_var(exoid, 1, EX_ELEM_BLOCK, var_index, ids[i], num_elem_this_blk, var_values);

//         for (int n = 0; n < num_elem_this_blk; n++) {
//           int c = n + offset;
//           var_c[k][c] = var_values[n];
//         }
//         free(var_values);
//         printf(
//           "MyPID=%d  ierr=%d  id=%d  ncells=%d\n", comm->MyPID(), ierr, ids[i], num_elem_this_blk);

//         offset += num_elem_this_blk;
//       }
//       ncells = offset;
//     }

//     for (int i = 0; i < num_vars; i++) { free(var_names[i]); }

//     ierr = ex_close(exoid);
//     printf("Closing file: %s ncells=%d error=%d\n", file_name.c_str(), ncells, ierr);
//   }
// }


// -----------------------------------------------------------------------------
// prints state statistics
// -----------------------------------------------------------------------------
void
WriteStateStatistics(const State& S, const VerboseObject& vo, const Teuchos::EVerbosityLevel vl)
{
  if (vo.os_OK(vl)) {
    Teuchos::OSTab tab = vo.getOSTab();
    *vo.os() << "\nField                                    Min/Max/Avg" << std::endl;

    for (auto name = S.data_begin(); name!=S.data_end(); ++name) {
      std::string display_name = Keys::abbreviate(name->first, 33);

      for (const auto& r : *name->second) {
        if (r.second->ValidType<CompositeVector>()) {
          auto cv = r.second->Get<CompositeVector>();
          // for (auto comp : cv) {
          //   std::map<std::string, double> vmin, vmax, vavg;
          //   auto vec = cv.getComponent(comp, false);
          //   Teuchos::Array<double> res;
          //   vec->normInf(res);
            
          //   vmax[comp] = vec->normInf();
          //   vmean[comp] = vec->meanValue();
            
          // r.second->Get<CompositeVector>().minValue(vmin);
          // r.second->Get<CompositeVector>().maxValue(vmax);
          // r.second->Get<CompositeVector>().meanValue(vavg);

          // for (auto c_it = vmin.begin(); c_it != vmin.end(); ++c_it) {
          //   std::string namedot(Keys::getKey(display_name, r.first)), name_comp(c_it->first);
          //   if (vmin.size() != 1) namedot.append("." + name_comp);
          //   namedot.resize(40, '.');
          //   *vo.os() << std::defaultfloat << namedot << " " << c_it->second << " / "
          //            << vmax[name_comp] << " / " << vavg[name_comp] << std::endl;
          // }

        } else if (r.second->ValidType<double>()) {
          double vmin = r.second->Get<double>();
          auto namedot = Keys::getKey(display_name, r.first);
          namedot.resize(40, '.');
          *vo.os() << namedot << " " << vmin << std::endl;

        } else if (r.second->ValidType<AmanziGeometry::Point>()) {
          const auto& p = r.second->Get<AmanziGeometry::Point>();
          auto namedot = Keys::getKey(display_name, r.first);
          namedot.resize(40, '.');
          *vo.os() << namedot;
          for (int i = 0; i < p.dim(); ++i) *vo.os() << " " << p[i];
          *vo.os() << std::endl;
        }
      }
    }
  }
}


// -----------------------------------------------------------------------------
// prints state statistics
// -----------------------------------------------------------------------------
void
WriteStateStatistics(const State& S)
{
  Teuchos::ParameterList plist;
  plist.sublist("verbose object").set<std::string>("verbosity level", "high");
  auto vo = Teuchos::rcp(new Amanzi::VerboseObject("Test", plist));
  WriteStateStatistics(S, *vo);
}

} // namespace Amanzi
