/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon (coonet@ornl.gov)
*/

#include "Key.hh"
#include "DataStructuresHelpers.hh"
#include "CompositeVectorFunction.hh"

namespace Amanzi {
namespace Functions {

CompositeVectorFunction::CompositeVectorFunction(Teuchos::ParameterList& list,
        std::string function_name,
        AmanziMesh::Entity_kind entity_kind)
  : entity_kind_(entity_kind)
{
  // All are expected to be sublists of identical structure.
  for (auto sublist : list) {
    std::string name = sublist.first;
    if (list.isSublist(name)) {
      Teuchos::ParameterList& spec_plist = list.sublist(name);

      try {
        readSpec_(spec_plist, function_name, false);
      } catch (Errors::Message& msg) {
        Errors::Message m;
        m << "in sublist " << name << ": " << msg.what();
        throw(m);
      }

    } else { // ERROR -- parameter is not a sublist
      Errors::Message m;
      m << "parameter " << name << " is not a sublist";
      throw(m);
    }
  }
}


void
CompositeVectorFunction::addSpec(const std::string& compname,
        AmanziMesh::Entity_kind entity_kind,
        int num_vectors,
        const std::string& region,
        const Teuchos::RCP<const MultiFunction>& func)
{
  MeshFunction<MultiFunction>::addSpec(Spec(compname,
               PatchSpace(mesh_, false, region, entity_kind, num_vectors, -1),
               func,
               false));
}


//
// process a list for regions, components, and functions
//
void
CompositeVectorFunction::readSpec_(Teuchos::ParameterList& list,
        const std::string& function_name,
        bool ghosted)
{
  Teuchos::Array<std::string> regions;
  if (list.isParameter("regions")) {
    regions = list.get<Teuchos::Array<std::string>>("regions");
  } else {
    regions.push_back(list.get<std::string>("region", Keys::cleanPListName(list.name())));
  }

  Teuchos::Array<std::string> components;
  if (list.isParameter("component")) {
    components.push_back(list.get<std::string>("component"));
  } else {
    std::vector<std::string> defs{ "cell", "face", "node" };
    Teuchos::Array<std::string> def(defs);
    components = list.get<Teuchos::Array<std::string>>("components", def);
  }

  Teuchos::Array<AmanziMesh::Entity_kind> entity_kinds;

  if (entity_kind_ == AmanziMesh::Entity_kind::UNKNOWN) {
    if (list.isParameter("entity kind")) {
      entity_kinds.push_back(AmanziMesh::createEntityKind(list.get<std::string>("entity kind")));
    } else if (list.isParameter("entity kinds")) {
      auto ekinds = list.get<Teuchos::Array<std::string>>("entity kinds");
      for (auto ekind : ekinds) entity_kinds.push_back(AmanziMesh::createEntityKind(ekind));
    } else {
      for (auto compname : components) {
        AmanziMesh::Entity_kind ekind;
        try {
          ekind = AmanziMesh::createEntityKind(compname);
        } catch (std::exception& msg) {
          Errors::Message m;
          m << "error in sublist " << function_name << ": " << msg.what();
          throw(m);
        }
        entity_kinds.push_back(ekind);
      }
    }

    if (entity_kinds.size() != components.size()) {
      Errors::Message m;
      m << "error in sublist " << function_name << ": \"components\" and \"entity kinds\" contain lists of differing lengths.";
      throw(m);
    }

  } else {
    entity_kinds.resize(components.size(), entity_kind_);
  }

  Teuchos::ParameterList f_list = list.sublist(function_name, true);

  // Make the function.
  Teuchos::RCP<Function> f;
  FunctionFactory f_fact;
  try {
    f = Teuchos::rcp(f_fact.Create(f_list));
  } catch (std::exception& msg) {
    Errors::Message m;
    m << "error in sublist " << function_name << ": " << msg.what();
    throw(m);
  }

  // TODO: Currently this does not support multiple-DoF functions.   --ETC
  Teuchos::RCP<MultiFunction> func = Teuchos::rcp(new MultiFunction(f));

  for (int i=0; i!=components.size(); ++i) {
    auto comp = components[i];
    auto ekind = entity_kinds[i];
    for (auto region : regions) {
      addSpec(comp, ekind, 1, region, func);
    }
  }
}


void
CompositeVectorFunction::Compute(double time, CompositeVector& vec)
{
  for (auto [compname, ps, functor, marker] : *this) {
    if (vec.hasComponent(compname)) {
      Patch<double> p(ps);
      Impl::computeFunction(*functor, time, p);
      patchToCompositeVector(p, compname, vec);
    }
  }
}


} // namespace Functions
} // namespace Amanzi
