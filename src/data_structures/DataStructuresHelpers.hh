/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon
*/

//!
#ifndef DATA_STRUCTURE_HELPERS_HH_
#define DATA_STRUCTURE_HELPERS_HH_

#include "AmanziTypes.hh"
#include "AmanziMap.hh"
#include "Patch.hh"
#include "CompositeVector.hh"

namespace Amanzi {

void
patchToCompositeVector(const Patch& p, const std::string& component, CompositeVector& cv);

void
patchToCompositeVector(const Patch& p,
                       const std::string& component,
                       CompositeVector& cv,
                       CompositeVector_<int>& flag_cv);

//
// Copies values from a set of patches into a vector.
//
void
multiPatchToCompositeVector(const MultiPatch& mp, const std::string& component, CompositeVector& cv);

//
// Copies values and flag from a set of patches into a vector and a flag vector.
//
void
multiPatchToCompositeVector(const MultiPatch& mp,
        const std::string& component,
        CompositeVector& cv,
        CompositeVector_<int>& flag);


void
DeriveFaceValuesFromCellValues(CompositeVector&);


// Create a BFS-ordered list of TreeVector(Space) nodes.
template <class T>
void
recurseTreeVectorBFS(T& tv, std::vector<Teuchos::RCP<T>>& list)
{
  for (typename T::iterator it = tv.begin(); it != tv.end(); ++it) { list.push_back(*it); }

  for (typename T::iterator it = tv.begin(); it != tv.end(); ++it) {
    recurseTreeVectorBFS<T>(**it, list);
  }
}

template <class T>
void
recurseTreeVectorBFS_const(const T& tv, std::vector<Teuchos::RCP<const T>>& list)
{
  for (typename T::const_iterator it = tv.begin(); it != tv.end(); ++it) { list.push_back(*it); }

  for (typename T::const_iterator it = tv.begin(); it != tv.end(); ++it) {
    recurseTreeVectorBFS_const<T>(**it, list);
  }
}

// Create a list of leaf nodes of the TreeVector(Space)
template <class T>
std::vector<Teuchos::RCP<T>>
collectTreeVectorLeaves(T& tv)
{
  std::vector<Teuchos::RCP<T>> list;
  list.push_back(Teuchos::rcpFromRef(tv));
  recurseTreeVectorBFS<T>(tv, list);

  std::vector<Teuchos::RCP<T>> leaves;
  for (typename std::vector<Teuchos::RCP<T>>::iterator it = list.begin(); it != list.end(); ++it) {
    if ((*it)->getData() != Teuchos::null) { leaves.push_back(*it); }
  }

  return leaves;
}

template <class T>
std::vector<Teuchos::RCP<const T>>
collectTreeVectorLeaves_const(const T& tv)
{
  std::vector<Teuchos::RCP<const T>> list;
  list.push_back(Teuchos::rcpFromRef(tv));
  recurseTreeVectorBFS_const<T>(tv, list);

  std::vector<Teuchos::RCP<const T>> leaves;
  for (typename std::vector<Teuchos::RCP<const T>>::const_iterator it = list.begin();
       it != list.end();
       ++it) {
    if ((*it)->getData() != Teuchos::null) { leaves.push_back(*it); }
  }

  return leaves;
}

template <class T>
int
getNumTreeVectorLeaves(const T& tv)
{
  return collectTreeVectorLeaves_const(tv).size();
}


} // namespace Amanzi

#endif
