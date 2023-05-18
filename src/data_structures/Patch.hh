/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon
*/

//! A patch is data on a collection of meshed entities, as defined by a mesh and
//! a region.
#pragma once

#include <vector>

#include "Teuchos_RCP.hpp"
#include "Kokkos_Core.hpp"

#include "Mesh.hh"

namespace Amanzi {


//
// A set of entity IDs, defined by a region, mesh, and entity kind.
//
// Note this also has an arbitrary flag_type, which can be a boundary condition
// type or other helper flag.  It is NOT used by this class, but is kept here
// to make using these objects with State easier.
//
struct PatchSpace {
  Teuchos::RCP<const AmanziMesh::Mesh> mesh;
  bool ghosted;
  std::string region;
  AmanziMesh::Entity_kind entity_kind;
  int num_vectors;
  int flag_type;

  PatchSpace() : ghosted(false) {}
  PatchSpace(const Teuchos::RCP<const AmanziMesh::Mesh>& mesh_,
             bool ghosted_,
             const std::string& region_,
             const AmanziMesh::Entity_kind& entity_kind_,
             const int& num_vectors_,
             const int& flag_type_)
    : mesh(mesh_),
      ghosted(ghosted_),
      region(region_),
      entity_kind(entity_kind_),
      num_vectors(num_vectors_),
      flag_type(flag_type_)
  {}

  int size() const
  {
    if (entity_kind == AmanziMesh::BOUNDARY_FACE) {
      Errors::Message msg("Patch cannot handle BOUNDARY_FACE entities, because "
                          "Mesh does not support sets on these types of "
                          "entities.  Instead use FACE and filter as needed.");
    }

    return mesh->getSetSize(region,
                            entity_kind,
                            ghosted ? AmanziMesh::Parallel_kind::ALL :
                                      AmanziMesh::Parallel_kind::OWNED);
  }
};


//
// A collection of independent patch spaces.  Conceptually these should share
// the same entity_kind, ghosted, mesh, and num_vectors.
template<typename T> struct MultiPatch;

struct MultiPatchSpace {
  Teuchos::RCP<const AmanziMesh::Mesh> mesh;
  bool ghosted;
  int flag_type;
  AmanziMesh::Entity_kind flag_entity;

  MultiPatchSpace() : ghosted(false) {}
  MultiPatchSpace(bool ghosted_) : ghosted(ghosted_) {}
  MultiPatchSpace(const Teuchos::RCP<const AmanziMesh::Mesh>& mesh_,
                  bool ghosted_,
                  int flag_type_ = -1)
    : mesh(mesh_), ghosted(ghosted_), flag_type(flag_type_), flag_entity(AmanziMesh::UNKNOWN)
  {}

  template<typename T>
  Teuchos::RCP<MultiPatch<T>> Create() const;

  const PatchSpace& operator[](const int& i) const { return subspaces_[i]; }

  void set_mesh(const Teuchos::RCP<const AmanziMesh::Mesh>& mesh_)
  {
    mesh = mesh_;
    for (auto& p : *this) { p.mesh = mesh_; }
  }

  using const_iterator = std::vector<PatchSpace>::const_iterator;
  const_iterator begin() const { return subspaces_.begin(); }
  const_iterator end() const { return subspaces_.end(); }
  std::size_t size() const { return subspaces_.size(); }

  using iterator = std::vector<PatchSpace>::iterator;
  iterator begin() { return subspaces_.begin(); }
  iterator end() { return subspaces_.end(); }

  void addPatch(const std::string& region, AmanziMesh::Entity_kind entity_kind, int num_vectors)
  {
    subspaces_.emplace_back(PatchSpace{ mesh, ghosted, region, entity_kind, num_vectors, flag_type });
  }

  void addPatch(const PatchSpace& ps) { subspaces_.emplace_back(ps); }

 private:
  std::vector<PatchSpace> subspaces_;
};


//
// A set of entity IDs and data on those entities.
//
template<typename T>
struct Patch {
  using ViewType = Kokkos::View<T**, Kokkos::LayoutLeft>;

  Patch(const PatchSpace& space_) : space(space_)
  {
    Kokkos::resize(data, space.size(), space.num_vectors);
  }

  KOKKOS_INLINE_FUNCTION ~Patch(){};

  PatchSpace space;
  // note, this layout is required to ensure that function is slowest-varying,
  // and so can be used with MultiFunction::apply(). See note in
  // MultiFunction.hh
  ViewType data;

  std::size_t size() const { return data.extent(0); }
  std::size_t getNumVectors() const { return data.extent(1); }
};

//
// A collection of Patches that share contiguous memory.
//
template<typename T>
struct MultiPatch {
  explicit MultiPatch(const MultiPatchSpace& space_) : space(space_)
  {
    for (const auto& subspace : space) patches_.emplace_back(Patch<T>{ subspace });
  }

  using iterator = typename std::vector<Patch<T>>::iterator;
  iterator begin() { return patches_.begin(); }
  iterator end() { return patches_.end(); }
  std::size_t size() const { return patches_.size(); }

  using const_iterator = typename std::vector<Patch<T>>::const_iterator;
  const_iterator begin() const { return patches_.begin(); }
  const_iterator end() const { return patches_.end(); }

  Patch<T>& operator[](const int& i) { return patches_[i]; }

  const Patch<T>& operator[](const int& i) const { return patches_[i]; }

  MultiPatchSpace space;

 protected:
  std::vector<Patch<T>> patches_;
};


template<typename T>
Teuchos::RCP<MultiPatch<T>>
MultiPatchSpace::Create() const
{
  return Teuchos::rcp(new MultiPatch<T>(*this));
}

} // namespace Amanzi
