/*
  Copyright 2010-201x held jointly by LANL, ORNL, LBNL, and PNNL.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Rao Garimella, others
*/

// Default imlementations of MeshFramework.

#include "VerboseObject.hh"
#include "errors.hh"
#include "exceptions.hh"
#include "MeshFramework.hh"
#include "Mesh_Helpers_impl.hh"

namespace Amanzi {
namespace AmanziMesh {

MeshFramework::MeshFramework(const Comm_ptr_type& comm,
                             const Teuchos::RCP<const AmanziGeometry::GeometricModel>& gm,
                             const Teuchos::RCP<Teuchos::ParameterList>& plist)
  : comm_(comm),
    gm_(gm),
    plist_(plist),
    space_dim_(-1),
    manifold_dim_(-1),
    vis_mesh_(Teuchos::null)
{
  if (!plist_.get()) plist_ = Teuchos::rcp(new Teuchos::ParameterList("mesh"));
  vo_ = Teuchos::rcp(new VerboseObject(comm, "MeshFramework", *plist_));
}


Parallel_kind
MeshFramework::getEntityPtype(const Entity_kind kind, const Entity_ID entid) const
{
  return entid >= getNumEntities(kind, Parallel_kind::OWNED) ? Parallel_kind::OWNED : Parallel_kind::GHOST;
}

Entity_GID
MeshFramework::getEntityGID(const Entity_kind kind, const Entity_ID lid) const
{
  AMANZI_ASSERT(comm_->getSize() == 1);
  return lid;
}

View_type<const Entity_GID,MemSpace_kind::HOST>
MeshFramework::getEntityGIDs(const Entity_kind kind, bool ghosted) const
{
  auto count = getNumEntities(kind, ghosted ? Parallel_kind::ALL : Parallel_kind::OWNED);
  View_type<Entity_GID,MemSpace_kind::HOST> gids("gids", count);
  for (int lid=0; lid!=gids.size(); ++lid) {
    auto gid = getEntityGID(kind, lid);
    if (gid < 0) {
      auto gid2 = getEntityGID(kind, lid);
    }
    AMANZI_ASSERT(gid >= 0);
    gids(lid) = gid;
  }
  return gids;
}

Entity_ID MeshFramework::getEntityParent(const Entity_kind kind, const Entity_ID entid) const
{
  Errors::Message msg("Parent entities not implemented.");
  Exceptions::amanzi_throw(msg);
  return -1;
}

Cell_kind
MeshFramework::getCellType(const Entity_ID c) const
{
  cEntity_ID_View faces;
  getCellFaces(c, faces);
  return getCellType_(c, faces);
}

Cell_kind
MeshFramework::getCellType_(const Entity_ID c,
                           const cEntity_ID_View& faces) const
{
  if (getManifoldDimension() == 2) {
    switch (faces.size()) {
    case 3:
      return Cell_kind::TRI;
      break;
    case 4:
      return Cell_kind::QUAD;
      break;
    default:
      return Cell_kind::POLYGON;
    }
  } else if (getManifoldDimension() == 3) {
    int nquads = 0;
    for (const auto& f : faces) {
      cEntity_ID_View fnodes;
      getFaceNodes(f, fnodes);
      if (fnodes.size() == 4) nquads++;
    }

    switch (faces.size()) {
      case 4:
        if (nquads == 0) return Cell_kind::TET;
        else return Cell_kind::POLYHED;
        break;
      case 5:
        if (nquads == 1) return Cell_kind::PYRAMID;
        else if (nquads == 3) return Cell_kind::PRISM;
        else return Cell_kind::POLYHED;
        break;
      case 6:
        if (nquads == 6) return Cell_kind::HEX;
        else return Cell_kind::POLYHED;
        break;
      default:
        return Cell_kind::POLYHED;
    }
  } else {
    Errors::Message msg;
    msg << "Mesh of manifold_dimension = " << getManifoldDimension() << " not supported";
    Exceptions::amanzi_throw(msg);
  }
  return Cell_kind::UNKNOWN;
}


void
MeshFramework::setNodeCoordinate(const Entity_ID nodeid, const AmanziGeometry::Point& ncoord)
{
  Errors::Message msg("MeshFramework is not deformable.");
  Exceptions::amanzi_throw(msg);
}


View_type<const AmanziGeometry::Point,MemSpace_kind::HOST>
MeshFramework::getCellCoordinates(const Entity_ID c) const
{
  return Impl::computeCellCoordinates(*this, c);
}

View_type<const AmanziGeometry::Point,MemSpace_kind::HOST>
MeshFramework::getFaceCoordinates(const Entity_ID f) const
{
  return Impl::computeFaceCoordinates(*this, f);
}

View_type<const AmanziGeometry::Point,MemSpace_kind::HOST>
MeshFramework::getEdgeCoordinates(const Entity_ID e) const
{
  return Impl::computeEdgeCoordinates(*this, e);
}

//
// topology
//
void
MeshFramework::getCellEdges(const Entity_ID c, cEntity_ID_View& edges) const
{
  hasEdgesOrThrow();
  edges = Impl::computeCellEdges(*this, c);
}

void
MeshFramework::getCellNodes(const Entity_ID c, cEntity_ID_View& nodes) const
{
  nodes = Impl::computeCellNodes(*this, c);
}

void
MeshFramework::getFaceEdgesAndDirs(const Entity_ID f,
        cEntity_ID_View& edges,
        cDirection_View * const dirs) const {
  hasEdgesOrThrow();
  throwNotImplemented_("getFaceEdgesAndDirs");
}


void
MeshFramework::getEdgeNodes(const Entity_ID e, cEntity_ID_View& nodes) const
{
  hasEdgesOrThrow();
  throwNotImplemented_("getEdgeNodes");
}


void
MeshFramework::getEdgeCells(const Entity_ID e, const Parallel_kind ptype, cEntity_ID_View& cells) const
{
  hasEdgesOrThrow();
  throwNotImplemented_("getEdgeCells");
}

void
MeshFramework::getEdgeFaces(const Entity_ID e, const Parallel_kind ptype, cEntity_ID_View& faces) const
{
  hasEdgesOrThrow();
  throwNotImplemented_("getEdgeFaces");
}

void
MeshFramework::getNodeCells(const Entity_ID n, const Parallel_kind ptype, cEntity_ID_View& cells) const
{
  cells = Impl::computeNodeCells(*this, n, ptype);
}

void
MeshFramework::getNodeEdges(const Entity_ID nodeid, const Parallel_kind ptype, cEntity_ID_View& edgeids) const
{
  hasEdgesOrThrow();
  throwNotImplemented_("getNodeEdges");
}

void
MeshFramework::getSetEntities(const AmanziGeometry::RegionLabeledSet& region,
        const Entity_kind kind, const Parallel_kind ptype, cEntity_ID_View& entids) const
{
  throwNotImplemented_("getSetEntities");
}


void
MeshFramework::hasEdgesOrThrow() const
{
  if (!hasEdges()) {
    Errors::Message msg("MeshFramework does not include edges.");
    Exceptions::amanzi_throw(msg);
  }
}

void
MeshFramework::throwNotImplemented_(const std::string& methodname) const
{
  Errors::FrameworkNotImplemented msg;
  msg << "MeshFramework does not implement " << methodname << "()";
  Exceptions::amanzi_throw(msg);
}


} // namespace AmanziMesh
} // namespace Amanzi
