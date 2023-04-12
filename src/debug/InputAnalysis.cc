/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors:
*/

#include <fstream>
#include <string>

#include "errors.hh"
#include "exceptions.hh"
#include "dbc.hh"

#include "InputAnalysis.hh"

namespace Amanzi {

/* ******************************************************************
* Initilization.
****************************************************************** */
void
InputAnalysis::Init(Teuchos::ParameterList& plist)
{
  plist_ = &plist;

  if (plist.isSublist("analysis")) {
    Teuchos::ParameterList vo_list = plist.sublist("analysis");
    vo_ = new VerboseObject("InputAnalysis:" + domain_, vo_list);
  }
}


/* ******************************************************************
* Analysis of collected regions
****************************************************************** */
void
InputAnalysis::RegionAnalysis()
{
  if (!plist_->isSublist("analysis")) return;
  Teuchos::ParameterList alist = plist_->sublist("analysis").sublist(domain_);

  Errors::Message msg;
  Teuchos::OSTab tab = vo_->getOSTab();

  if (alist.isParameter("used source regions")) {
    std::vector<std::string> regions =
      alist.get<Teuchos::Array<std::string>>("used source regions").toVector();
    regions.erase(SelectUniqueEntries(regions.begin(), regions.end()), regions.end());

    for (int i = 0; i < regions.size(); i++) {
      int nblock(0), nblock_tmp, nvofs;
      double volume(0.0);

      typename AmanziMesh::Mesh::cEntity_ID_View block;
      typename AmanziMesh::Mesh::cDouble_View vofs;

      try {
        Kokkos::make_pair(block, vofs) = mesh_->getSetEntitiesAndVolumeFractions(
          regions[i], AmanziMesh::Entity_kind::CELL, AmanziMesh::Parallel_kind::OWNED);
        nblock = block.size();
        nvofs = vofs.size();

        Kokkos::parallel_reduce("InputAnalysis", nblock,
                KOKKOS_LAMBDA(const int& n, double& lvolume) {
                  double frac = (nvofs == 0) ? 1.0 : vofs[n];
                  lvolume += mesh_->getCellVolume(block[n]) * frac;
                }, volume);
      } catch (...) {
        nblock = -1;
      }

      // identify if we failed on some cores
      Teuchos::reduceAll(*mesh_->getComm(), Teuchos::REDUCE_MIN, 1, &nblock, &nblock_tmp);
      if (nblock_tmp < 0) {
        Kokkos::make_pair(block, vofs) = mesh_->getSetEntitiesAndVolumeFractions(
          regions[i], AmanziMesh::Entity_kind::FACE, AmanziMesh::Parallel_kind::OWNED);
        nblock = block.size();
        nvofs = vofs.size();

        double volume = 0.0;
        Kokkos::parallel_reduce("InputAnalysis", nblock,
                KOKKOS_LAMBDA(const int& n, double& lvolume) {
                  double frac = (nvofs == 0) ? 1.0 : vofs[n];
                  lvolume += mesh_->getFaceArea(block[n]) * frac;
                }, volume);
      }

      Kokkos::pair<double,double> vof_extrema = {1.0, 0.0};
      if (nvofs == 0) {
        vof_extrema.second = 1.0;
      } else {
        Kokkos::parallel_reduce("InputAnalysis", nvofs,
                KOKKOS_LAMBDA(const int& n, Kokkos::pair<double,double>& extrema) {
                  extrema.first = fmin(extrema.first, vofs[n]);
                  extrema.second = fmax(extrema.second, vofs[n]);
                }, vof_extrema);
      }

      nblock_tmp = nblock;
      int nvofs_tmp(nvofs);
      double volume_tmp(volume), vofs_min, vofs_max;

      Teuchos::reduceAll(*mesh_->getComm(), Teuchos::REDUCE_SUM, 1, &nblock_tmp, &nblock);
      Teuchos::reduceAll(*mesh_->getComm(), Teuchos::REDUCE_SUM, 1, &nvofs_tmp, &nvofs);
      Teuchos::reduceAll(*mesh_->getComm(), Teuchos::REDUCE_SUM, 1, &volume_tmp, &volume);
      Teuchos::reduceAll(*mesh_->getComm(), Teuchos::REDUCE_MIN, 1, &vof_extrema.first, &vofs_min);
      Teuchos::reduceAll(*mesh_->getComm(), Teuchos::REDUCE_MAX, 1, &vof_extrema.second, &vofs_max);
      if (vo_->os_OK(Teuchos::VERB_MEDIUM)) {
        std::string name(regions[i]);
        name.resize(std::min(40, (int)name.size()));
        *vo_->os() << "src: \"" << name << "\" has " << nblock << " cells"
                   << " of " << volume << " [m^3]";
        if (nvofs > 0) *vo_->os() << ", vol.fractions: " << vofs_min << "/" << vofs_max;
        *vo_->os() << std::endl;
      }

      if (nblock == 0) {
        msg << "Used source region is empty.";
        Exceptions::amanzi_throw(msg);
      }
    }
  }

  if (alist.isParameter("used boundary condition regions")) {
    std::vector<std::string> regions =
      alist.get<Teuchos::Array<std::string>>("used boundary condition regions").toVector();
    regions.erase(SelectUniqueEntries(regions.begin(), regions.end()), regions.end());

    for (int i = 0; i < regions.size(); i++) {
      auto [bblock, vvofs] = mesh_->getSetEntitiesAndVolumeFractions(
        regions[i], AmanziMesh::Entity_kind::FACE, AmanziMesh::Parallel_kind::OWNED);
      auto vofs = vvofs; // structured binding compiler/standard bug?
      auto block = bblock;
      int nblock = block.size();
      int nvofs = vofs.size();

      double area(0.0);
      Kokkos::parallel_reduce("InputAnalysis", nblock,
              KOKKOS_LAMBDA(const int& n, double& lvolume) {
                double frac = (nvofs == 0) ? 1.0 : vofs[n];
                lvolume += mesh_->getFaceArea(block[n]) * frac;
              }, area);

      Kokkos::pair<double,double> vof_extrema = {1.0, 0.0};
      if (nvofs == 0) {
        vof_extrema.second = 1.0;
      } else {
        Kokkos::parallel_reduce("InputAnalysis", nvofs,
                KOKKOS_LAMBDA(const int& n, Kokkos::pair<double,double>& extrema) {
                  extrema.first = fmin(extrema.first, vofs[n]);
                  extrema.second = fmax(extrema.second, vofs[n]);
                }, vof_extrema);
      }

      // verify that all faces are boundary faces
      int bc_flag(1);

      for (int n = 0; n < nblock; ++n) {
        auto cells = mesh_->getFaceCells(block[n], AmanziMesh::Parallel_kind::ALL);
        if (cells.size() != 1) bc_flag = 0;
      }

      int nblock_tmp(nblock), nvofs_tmp(nvofs), bc_flag_tmp(bc_flag);
      double area_tmp(area), vofs_min, vofs_max;

      Teuchos::reduceAll(*mesh_->getComm(), Teuchos::REDUCE_SUM, 1, &nblock_tmp, &nblock);
      Teuchos::reduceAll(*mesh_->getComm(), Teuchos::REDUCE_SUM, 1, &nvofs_tmp, &nvofs);
      Teuchos::reduceAll(*mesh_->getComm(), Teuchos::REDUCE_SUM, 1, &area_tmp, &area);
      Teuchos::reduceAll(*mesh_->getComm(), Teuchos::REDUCE_MIN, 1, &vof_extrema.first, &vofs_min);
      Teuchos::reduceAll(*mesh_->getComm(), Teuchos::REDUCE_MAX, 1, &vof_extrema.second, &vofs_max);
      Teuchos::reduceAll(*mesh_->getComm(), Teuchos::REDUCE_MIN, 1, &bc_flag_tmp, &bc_flag);

      if (vo_->os_OK(Teuchos::VERB_MEDIUM)) {
        std::string name(regions[i]);
        name.resize(std::min(40, (int)name.size()));
        *vo_->os() << "bc: \"" << name << "\" has " << nblock << " faces"
                   << " of " << area << " [m^2]";
        if (nvofs > 0) *vo_->os() << ", vol.fractions: " << vofs_min << "/" << vofs_max;
        *vo_->os() << std::endl;
      }

      if (nblock == 0) {
        msg << "Used boundary region is empty.";
        Exceptions::amanzi_throw(msg);
      }
      if (bc_flag == 0) {
        msg << "Used boundary region has non-boundary entries.";
        Exceptions::amanzi_throw(msg);
      }
    }
  }

  if (alist.isParameter("used observation regions")) {
    std::vector<std::string> regions =
      alist.get<Teuchos::Array<std::string>>("used observation regions").toVector();

    int nblock(0), nblock_tmp, nblock_max;
    for (int i = 0; i < regions.size(); i++) {
      double volume(0.0), volume_tmp;
      std::string type;

      // observation region may use either cells of faces
      if (!mesh_->isValidSetName(regions[i], AmanziMesh::Entity_kind::CELL) &&
          !mesh_->isValidSetName(regions[i], AmanziMesh::Entity_kind::FACE)) {
        std::string name(regions[i]);
        name.resize(std::min(40, (int)name.size()));
        *vo_->os() << "Observation region: \"" << name << "\" has unknown type." << std::endl;
      }

      try {
        auto [bblock, vvofs] = mesh_->getSetEntitiesAndVolumeFractions(
          regions[i], AmanziMesh::Entity_kind::CELL, AmanziMesh::Parallel_kind::OWNED);
        auto vofs = vvofs; // structured binding compiler/standard bug?
        auto block = bblock;
        nblock_tmp = nblock = block.size();
        type = "cells";
        Kokkos::parallel_reduce("InputAnalysis", nblock,
                KOKKOS_LAMBDA(const int& n, double& lvolume) {
                  lvolume += mesh_->getCellVolume(block[n]);
                }, volume);

        volume_tmp = volume;
        Teuchos::reduceAll(*mesh_->getComm(), Teuchos::REDUCE_SUM, 1, &nblock_tmp, &nblock);
        Teuchos::reduceAll(*mesh_->getComm(), Teuchos::REDUCE_SUM, 1, &volume_tmp, &volume);
      } catch (...) {
        nblock = -1;
      }

      // identify if we failed on some cores or region is empty
      Teuchos::reduceAll(*mesh_->getComm(), Teuchos::REDUCE_MIN, 1, &nblock, &nblock_tmp);
      Teuchos::reduceAll(*mesh_->getComm(), Teuchos::REDUCE_MAX, 1, &nblock, &nblock_max);

      if (nblock_tmp < 0 || nblock_max == 0) {
        auto [bblock, vvofs] = mesh_->getSetEntitiesAndVolumeFractions(
          regions[i], AmanziMesh::Entity_kind::FACE, AmanziMesh::Parallel_kind::OWNED);
        auto vofs = vvofs; // structured binding compiler/standard bug?
        auto block = bblock;
        nblock_tmp = nblock = block.size();
        type = "faces";
        Kokkos::parallel_reduce("InputAnalysis", nblock,
                KOKKOS_LAMBDA(const int& n, double& lvolume) {
                  lvolume += mesh_->getFaceArea(block[n]);
                }, volume);

        volume_tmp = volume;
        Teuchos::reduceAll(*mesh_->getComm(), Teuchos::REDUCE_SUM, 1, &nblock_tmp, &nblock);
        Teuchos::reduceAll(*mesh_->getComm(), Teuchos::REDUCE_SUM, 1, &volume_tmp, &volume);
      }

      if (vo_->getVerbLevel() >= Teuchos::VERB_MEDIUM) {
        std::string name(regions[i]);
        name.resize(std::min(40, (int)name.size()));
        *vo_->os() << "obs: \"" << name << "\" has " << nblock << " " << type
                   << ", size: " << volume << std::endl;
      }

      if (nblock == 0) {
        msg << "Used observation region is empty.";
        Exceptions::amanzi_throw(msg);
      }
    }
  }
}


/* ******************************************************************
* DEBUG output: boundary conditions.
****************************************************************** */
void
InputAnalysis::OutputBCs()
{
  if (!plist_->isSublist("analysis")) return;
  if (vo_->getVerbLevel() < Teuchos::VERB_EXTREME) return;

  int bc_counter = 0;

  if (plist_->isSublist("flow")) {
    Teuchos::ParameterList& flow_list = plist_->sublist("flow");

    Teuchos::ParameterList bc_list;
    if (flow_list.isSublist("boundary conditions")) {
      bc_list = flow_list.sublist("boundary conditions");
    }

    Teuchos::ParameterList mass_flux_list, pressure_list, seepage_list, head_list;
    if (bc_list.isSublist("mass flux")) {
      mass_flux_list = bc_list.sublist("mass flux");
      for (auto it = mass_flux_list.begin(); it != mass_flux_list.end(); ++it) {
        if (mass_flux_list.isSublist(mass_flux_list.name(it))) {
          Teuchos::ParameterList& bc = mass_flux_list.sublist(mass_flux_list.name(it));

          if ((bc.sublist("outward mass flux")).isSublist("function-tabular")) {
            std::stringstream ss;
            ss << "BCmassflux" << bc_counter++;

            Teuchos::ParameterList& f_tab =
              (bc.sublist("outward mass flux")).sublist("function-tabular");

            Teuchos::Array<double> times = f_tab.get<Teuchos::Array<double>>("x values");
            Teuchos::Array<double> values = f_tab.get<Teuchos::Array<double>>("y values");
            Teuchos::Array<std::string> time_fns = f_tab.get<Teuchos::Array<std::string>>("forms");

            int np = times.size() * 2 - 1;
            Teuchos::Array<double> times_plot(np);
            Teuchos::Array<double> values_plot(np);

            for (int i = 0; i < times.size() - 1; i++) {
              times_plot[2 * i] = times[i];
              values_plot[2 * i] = values[i];
              times_plot[2 * i + 1] = 0.5 * (times[i] + times[i + 1]);
            }
            times_plot[np - 1] = times[times.size() - 1];
            values_plot[np - 1] = values[times.size() - 1];

            for (int i = 0; i < time_fns.size(); i++) {
              if (time_fns[i] == "linear") {
                values_plot[2 * i + 1] = 0.5 * (values[i] + values[i + 1]);
              } else if (time_fns[i] == "constant") {
                values_plot[2 * i + 1] = values[i];
                times_plot[2 * i + 1] = times[i + 1];
              } else {
                Exceptions::amanzi_throw(Errors::Message(
                  "In the definition of BCs: tabular function can only be Linear or Constant"));
              }
            }

            std::string filename = ss.str() + ".dat";
            std::ofstream ofile(filename.c_str());

            ofile << "# "
                  << "time "
                  << "flux" << std::endl;
            for (int i = 0; i < np; i++) {
              ofile << times_plot[i] << " " << values_plot[i] << std::endl;
            }

            ofile.close();
          }
        }
      }
    }
    if (bc_list.isSublist("pressure")) {
      pressure_list = bc_list.sublist("pressure");
      for (auto it = pressure_list.begin(); it != pressure_list.end(); ++it) {
        if (pressure_list.isSublist(pressure_list.name(it))) {
          Teuchos::ParameterList& bc = pressure_list.sublist(pressure_list.name(it));
          if ((bc.sublist("boundary pressure")).isSublist("function-tabular")) {
            std::stringstream ss;
            ss << "BCpressure" << bc_counter++;


            Teuchos::ParameterList& f_tab =
              (bc.sublist("boundary pressure")).sublist("function-tabular");

            Teuchos::Array<double> times = f_tab.get<Teuchos::Array<double>>("x values");
            Teuchos::Array<double> values = f_tab.get<Teuchos::Array<double>>("y values");
            Teuchos::Array<std::string> time_fns = f_tab.get<Teuchos::Array<std::string>>("forms");

            int np = times.size() * 2 - 1;
            Teuchos::Array<double> times_plot(np);
            Teuchos::Array<double> values_plot(np);

            for (int i = 0; i < times.size() - 1; i++) {
              times_plot[2 * i] = times[i];
              values_plot[2 * i] = values[i];
              times_plot[2 * i + 1] = 0.5 * (times[i] + times[i + 1]);
            }
            times_plot[np - 1] = times[times.size() - 1];
            values_plot[np - 1] = values[times.size() - 1];

            for (int i = 0; i < time_fns.size(); i++) {
              if (time_fns[i] == "linear") {
                values_plot[2 * i + 1] = 0.5 * (values[i] + values[i + 1]);
              } else if (time_fns[i] == "constant") {
                values_plot[2 * i + 1] = values[i];
                times_plot[2 * i + 1] = times[i + 1];
              } else {
                Exceptions::amanzi_throw(Errors::Message(
                  "In the definition of BCs: tabular function can only be Linear or Constant"));
              }
            }

            std::string filename = ss.str() + ".dat";
            std::ofstream ofile(filename.c_str());

            ofile << "# time "
                  << "pressure" << std::endl;
            for (int i = 0; i < np; i++) {
              ofile << times_plot[i] << " " << values_plot[i] << std::endl;
            }

            ofile.close();
          }
        }
      }
    }

    if (bc_list.isSublist("seepage face")) {
      seepage_list = bc_list.sublist("seepage face");
      for (auto it = seepage_list.begin(); it != seepage_list.end(); ++it) {
        if (seepage_list.isSublist(seepage_list.name(it))) {
          Teuchos::ParameterList& bc = seepage_list.sublist(seepage_list.name(it));
          if ((bc.sublist("outward mass flux")).isSublist("function-tabular")) {
            std::stringstream ss;
            ss << "BCseepage" << bc_counter++;

            Teuchos::ParameterList& f_tab =
              (bc.sublist("outward mass flux")).sublist("function-tabular");

            Teuchos::Array<double> times = f_tab.get<Teuchos::Array<double>>("x values");
            Teuchos::Array<double> values = f_tab.get<Teuchos::Array<double>>("y values");
            Teuchos::Array<std::string> time_fns = f_tab.get<Teuchos::Array<std::string>>("forms");

            int np = times.size() * 2 - 1;
            Teuchos::Array<double> times_plot(np);
            Teuchos::Array<double> values_plot(np);

            for (int i = 0; i < times.size() - 1; i++) {
              times_plot[2 * i] = times[i];
              values_plot[2 * i] = values[i];
              times_plot[2 * i + 1] = 0.5 * (times[i] + times[i + 1]);
            }
            times_plot[np - 1] = times[times.size() - 1];
            values_plot[np - 1] = values[times.size() - 1];

            for (int i = 0; i < time_fns.size(); i++) {
              if (time_fns[i] == "linear") {
                values_plot[2 * i + 1] = 0.5 * (values[i] + values[i + 1]);
              } else if (time_fns[i] == "constant") {
                values_plot[2 * i + 1] = values[i];
                times_plot[2 * i + 1] = times[i + 1];
              } else {
                Exceptions::amanzi_throw(Errors::Message(
                  "In the definition of BCs: tabular function can only be Linear or Constant"));
              }
            }

            std::string filename = ss.str() + ".dat";
            std::ofstream ofile(filename.c_str());

            ofile << "# time "
                  << "flux" << std::endl;
            for (int i = 0; i < np; i++) {
              ofile << times_plot[i] << " " << values_plot[i] << std::endl;
            }

            ofile.close();
          }
        }
      }
    }

    if (bc_list.isSublist("static head")) {
      head_list = bc_list.sublist("static head");
      for (auto it = head_list.begin(); it != head_list.end(); ++it) {
        if (head_list.isSublist(head_list.name(it))) {
          Teuchos::ParameterList& bc = head_list.sublist(head_list.name(it));
          if ((bc.sublist("water table elevation")).isSublist("function-tabular")) {
            std::stringstream ss;
            ss << "BChead" << bc_counter++;

            Teuchos::ParameterList& f_tab =
              (bc.sublist("water table elevation")).sublist("function-tabular");

            Teuchos::Array<double> times = f_tab.get<Teuchos::Array<double>>("x values");
            Teuchos::Array<double> values = f_tab.get<Teuchos::Array<double>>("y values");
            Teuchos::Array<std::string> time_fns = f_tab.get<Teuchos::Array<std::string>>("forms");

            int np = times.size() * 2 - 1;
            Teuchos::Array<double> times_plot(np);
            Teuchos::Array<double> values_plot(np);

            for (int i = 0; i < times.size() - 1; i++) {
              times_plot[2 * i] = times[i];
              values_plot[2 * i] = values[i];
              times_plot[2 * i + 1] = 0.5 * (times[i] + times[i + 1]);
            }
            times_plot[np - 1] = times[times.size() - 1];
            values_plot[np - 1] = values[times.size() - 1];

            for (int i = 0; i < time_fns.size(); i++) {
              if (time_fns[i] == "linear") {
                values_plot[2 * i + 1] = 0.5 * (values[i] + values[i + 1]);
              } else if (time_fns[i] == "constant") {
                values_plot[2 * i + 1] = values[i];
                times_plot[2 * i + 1] = times[i + 1];
              } else {
                Exceptions::amanzi_throw(Errors::Message(
                  "In the definition of BCs: tabular function can only be Linear or Constant"));
              }
            }

            std::string filename = ss.str() + ".dat";
            std::ofstream ofile(filename.c_str());

            ofile << "# time "
                  << "head" << std::endl;
            for (int i = 0; i < np; i++) {
              ofile << times_plot[i] << " " << values_plot[i] << std::endl;
            }

            ofile.close();
          }
        }
      }
    }
  }
}


/* ******************************************************************
* Selects unique entries and places them in [first, last)
****************************************************************** */
template <class Iterator>
Iterator
InputAnalysis::SelectUniqueEntries(Iterator first, Iterator last)
{
  while (first != last) {
    Iterator next(first);
    last = std::remove(++next, last, *first);
    first = next;
  }
  return last;
}

} // namespace Amanzi
