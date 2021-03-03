/*
  Copyright 2010-201x held jointly by LANL, ORNL, LBNL, and PNNL.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Rao Garimella, others
*/

//! The interface for meshes provided by external frameworks.

/*!

*/

#pragma once

#include <string>

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "MeshDefs.hh"
#include "AmanziComm.hh"

namespace Amanzi {

class VerboseObject;
namespace AmanziGeometry {
class Point;
class RegionLabeledSet;
class GeometricModel;
}

namespace AmanziMesh {

class MeshFramework  {
 protected:
  MeshFramework(const Comm_ptr_type& comm,
                const Teuchos::RCP<const AmanziGeometry::GeometricModel>& gm,
                const Teuchos::RCP<Teuchos::ParameterList>& plist);

 public:
  virtual ~MeshFramework() = default;

  // ----------------------
  // Accessors and Mutators
  // ----------------------
  Comm_ptr_type get_comm() const { return comm_; }
  void set_comm(const Comm_ptr_type& comm) { comm_ = comm; }

  Teuchos::RCP<Teuchos::ParameterList> get_parameter_list() const { return plist_; }
  void set_parameter_list(const Teuchos::RCP<Teuchos::ParameterList>& plist) { plist_ = plist; }

  Teuchos::RCP<const VerboseObject> get_verbose_object() const { return vo_; }
  void set_verbose_object(const Teuchos::RCP<const VerboseObject>& vo) { vo_ = vo; }

  Teuchos::RCP<const AmanziGeometry::GeometricModel> get_geometric_model() const { return gm_; }
  void set_geometric_model(const Teuchos::RCP<const AmanziGeometry::GeometricModel>& gm) { gm_ = gm; }

  // space dimension describes the dimension of coordinates in space
  std::size_t get_space_dimension() const { return space_dim_; }
  void set_space_dimension(unsigned int dim) { space_dim_ = dim; }

  // manifold dimension describes the dimensionality of the corresponding R^n
  // manifold onto which this mesh can be projected.
  std::size_t get_manifold_dimension() const { return manifold_dim_; }
  void set_manifold_dimension(const unsigned int dim) { manifold_dim_ = dim; }

  // Some meshes are subsets of or derived from a parent mesh.
  // Usually this is null, but some meshes may provide it.
  virtual Teuchos::RCP<const MeshFramework> get_parent() const { return Teuchos::null; }

  // Some meshes have a corresponding mesh that is better for visualization.
  const MeshFramework& get_vis_mesh() const {
    if (vis_mesh_.get()) return *vis_mesh_;
    return *this;
  }
  void set_vis_mesh(const Teuchos::RCP<const MeshFramework>& vis_mesh) { vis_mesh_ = vis_mesh; }

  // Some meshes have edges
  //
  // DEVELOPER NOTE: frameworks that do not implement edges need not provide
  // any edge method -- defaults here all throw errors.
  virtual bool has_edges() const { return false; }

  // Some meshes can be deformed.
  virtual bool is_deformable() const { return false; }

  // ----------------
  // Entity meta-data
  // ----------------
  virtual std::size_t getNumEntities(const Entity_kind kind, const Parallel_type ptype) const = 0;

  // Parallel type of the entity.
  //
  // DEVELOPER NOTE: meshes which order entities by OWNED, GHOSTED need not
  // implement this method.
  virtual Parallel_type getEntityPtype(const Entity_kind kind, const Entity_ID entid) const;

  // Global ID of any entity
  //
  // DEVELOPER NOTE: serial meshes need not provide this method -- the default
  // returns lid.
  virtual Entity_GID getEntityGID(const Entity_kind kind, const Entity_ID lid) const;
  virtual Entity_GID_List getEntityGIDs(const Entity_kind kind, const Parallel_type ptype) const;

  // corresponding entity in the parent mesh
  virtual Entity_ID getEntityParent(const Entity_kind kind, const Entity_ID entid) const;

  // Cell types: UNKNOWN, TRI, QUAD, etc. See MeshDefs.hh
  //
  // DEVELOPER NOTE: Default implementation guesses based on topology.
  virtual Cell_type getCellType(const Entity_ID cellid) const;


  //---------------------
  // Geometry
  //---------------------
  // locations
  virtual AmanziGeometry::Point getNodeCoordinate(const Entity_ID node) const = 0;
  virtual void setNodeCoordinate(const Entity_ID nodeid, const AmanziGeometry::Point& ncoord);
  virtual AmanziGeometry::Point getCellCentroid(const Entity_ID c) const;
  virtual AmanziGeometry::Point getFaceCentroid(const Entity_ID f) const;
  virtual AmanziGeometry::Point getEdgeCentroid(const Entity_ID e) const;

  virtual Point_List cell_coordinates(const Entity_ID c) const;
  virtual Point_List face_coordinates(const Entity_ID f) const;
  virtual Point_List edge_coordinates(const Entity_ID e) const;

  // extent
  virtual double getCellVolume(const Entity_ID c) const;
  virtual double getFaceArea(const Entity_ID f) const;
  virtual double getEdgeLength(const Entity_ID e) const;

  // lumped things for more efficient calculation
  virtual std::pair<double, AmanziGeometry::Point>
  computeCellGeometry(const Entity_ID c) const;

  virtual std::tuple<double, AmanziGeometry::Point, Point_List>
  computeFaceGeometry(const Entity_ID f) const;

  virtual std::pair<AmanziGeometry::Point, AmanziGeometry::Point>
  computeEdgeGeometry(const Entity_ID e) const;

  // Normal vector of a face
  //
  // The vector is normalized and then weighted by the area of the face.
  //
  // Orientation is the natural orientation, e.g. that it points from cell 0 to
  // cell 1 with respect to face_cell adjacency information.
  inline
  AmanziGeometry::Point getFaceNormal(const Entity_ID f) const {
    return getFaceNormal(f, -1, nullptr);
  }

  // Normal vector of a face, outward with respect to a cell.
  //
  // The vector is normalized and then weighted by the area of the face.
  //
  // Orientation, if provided, returns the direction of
  // the natural normal (1 if outward, -1 if inward).
  virtual AmanziGeometry::Point getFaceNormal(const Entity_ID f,
          const Entity_ID c, int * const orientation=nullptr) const;

  // Vector describing the edge, where the length is the edge length.
  //
  // Orientation is the natural orientation, e.g. that it points from node 0 to
  // node 1 with respect to edge_node adjacency information.
  inline
  AmanziGeometry::Point getEdgeVector(const Entity_ID e) const {
    return getEdgeVector(e, -1, nullptr);
  }

  // Vector describing the edge, where the length is the edge length, with respect to node.
  //
  // Orientation, if provided, returns the direction of the natural orientation
  // (1 if this direction, -1 if the opposite).
  virtual AmanziGeometry::Point getEdgeVector(const Entity_ID e,
          const Entity_ID n, int * const orientation=nullptr) const;


  //---------------------
  // Downward adjacencies
  //---------------------
  // Get faces of a cell
  //
  // On a distributed mesh, this will return all the faces of the
  // cell, OWNED or GHOST. If the framework supports it, the faces will be
  // returned in a standard order according to Exodus II convention
  // for standard cells; in all other situations (not supported or
  // non-standard cells), the list of faces will be in arbitrary order
  //
  // EXTENSIONS: MSTK FRAMEWORK: by the way the parallel partitioning,
  // send-receive protocols and mesh query operators are designed, a side 
  // effect of this is that master and ghost entities will have the same
  // hierarchical topology.
  void getCellFaces(
       const Entity_ID c,
       Entity_ID_List& faces) const {
    getCellFacesAndDirs(c, faces, nullptr);
  }

  // Get faces of a cell and directions in which the cell uses the face
  //
  // On a distributed mesh, this will return all the faces of the
  // cell, OWNED or GHOST. If the framework supports it, the faces will be
  // returned in a standard order according to Exodus II convention
  // for standard cells.
  //
  // In 3D, direction is 1 if face normal points out of cell
  // and -1 if face normal points into cell
  // In 2D, direction is 1 if face/edge is defined in the same
  // direction as the cell polygon, and -1 otherwise
  virtual void getCellFacesAndDirs(
    const Entity_ID c,
    Entity_ID_List& faces,
    Entity_Direction_List * const dirs) const = 0;

  // Get the bisectors, i.e. vectors from cell centroid to face centroids.
  virtual void getCellFacesAndBisectors(
          const Entity_ID cellid,
          Entity_ID_List& faceids,
          Point_List * const bisectors) const;

  virtual void getCellEdges(const Entity_ID c, Entity_ID_List& edges) const;
  virtual void getCellNodes(const Entity_ID c, Entity_ID_List& nodes) const;

  // Get edges and dirs of a 2D cell.
  //
  // This is to make the code cleaner for integrating over the cell in 2D
  // where faces and edges are identical but integrating over the cells using
  // face information is more cumbersome (one would have to take the face
  // normals, rotate them and then get a consistent edge vector)
  virtual void getCell2DEdgesAndDirs(const Entity_ID cellid,
          Entity_ID_List& edgeids,
          Entity_Direction_List * const edge_dirs) const;

  void getFaceEdges(const Entity_ID f,
                  Entity_ID_List& edges) const {
    getFaceEdgesAndDirs(f, edges);
  }

  // Get edges of a face and directions in which the face uses the edges.
  //
  // In 3D, edge direction is 1 when it is oriented counter clockwise
  // with respect to the face natural normal.
  //
  // On a distributed mesh, this will return all the edges of the
  // face, OWNED or GHOST. If the framework supports it, the edges will be
  // returned in a ccw order around the face as it is naturally defined.
  //
  // IMPORTANT NOTE IN 2D: In meshes where the cells are two
  // dimensional, faces and edges are identical. For such cells, this
  // operator will return a single edge and a direction of 1. However,
  // this direction cannot be relied upon to compute, say, a contour
  // integral around the 2D cell.
  virtual void getFaceEdgesAndDirs(const Entity_ID f,
          Entity_ID_List& edges,
          Entity_Direction_List * const dirs=nullptr) const = 0;

  // Get nodes of face
  //
  // In 3D, the nodes of the face are returned in ccw order consistent
  // with the face normal.
  virtual void getFaceNodes(const Entity_ID f, Entity_ID_List& nodes) const = 0;

  virtual void getEdgeNodes(const Entity_ID e, Entity_ID_List& nodes) const;

  //-------------------
  // Upward adjacencies
  //-------------------
  // The cells are returned in no particular order. Also, the order of cells
  // is not guaranteed to be the same for corresponding faces on different
  // processors
  virtual void getFaceCells(const Entity_ID f,
                          const Parallel_type ptype,
                          Entity_ID_List& cells) const = 0;

  // Cells of a given Parallel_type connected to an edge
  //
  // The order of cells is not guaranteed to be the same for corresponding
  // edges on different processors
  virtual void getEdgeCells(const Entity_ID edgeid,
                          const Parallel_type ptype,
                          Entity_ID_List& cellids) const = 0;

  // Faces of type 'ptype' connected to an edge
  // NOTE: The order of faces is not guaranteed to be the same for
  // corresponding edges on different processors
  virtual void getEdgeFaces(const Entity_ID edgeid,
                          const Parallel_type ptype,
                          Entity_ID_List& faceids) const = 0;

  // Cells of type 'ptype' connected to a node
  // NOTE: The order of cells is not guaranteed to be the same for
  // corresponding nodes on different processors
  virtual void getNodeCells(const Entity_ID nodeid,
                          const Parallel_type ptype,
                          Entity_ID_List& cellids) const;

  // Faces of type parallel 'ptype' connected to a node
  // NOTE: The order of faces is not guarnateed to be the same for
  // corresponding nodes on different processors
  virtual void getNodeFaces(const Entity_ID nodeid,
                          const Parallel_type ptype,
                          Entity_ID_List& faceids) const = 0;

  // Edges of type 'ptype' connected to a node
  //
  // The order of edges is not guaranteed to be the same for corresponding
  // node on different processors
  virtual void getNodeEdges(const Entity_ID nodeid,
                          const Parallel_type ptype,
                          Entity_ID_List& edgeids) const;

  //--------------------------------------------------------------
  // Mesh Sets for ICs, BCs, Material Properties and whatever else
  //--------------------------------------------------------------
  // Get list of entities of type kind from a framework set.
  //
  // Default implementation does not support framework sets.
  virtual void getSetEntities(const AmanziGeometry::RegionLabeledSet& region,
          const Entity_kind kind,
          const Parallel_type ptype,
          Entity_ID_List& entids) const;

 protected:
  void hasEdgesOrThrow_() const;
  void throwNotImplemented_(const std::string& fname) const;

 protected:
  Comm_ptr_type comm_;
  Teuchos::RCP<Teuchos::ParameterList> plist_;
  Teuchos::RCP<const VerboseObject> vo_;
  Teuchos::RCP<const AmanziGeometry::GeometricModel> gm_;
  Teuchos::RCP<const MeshFramework> vis_mesh_;

  std::size_t space_dim_;
  std::size_t manifold_dim_;
};

}  // namespace AmanziMesh
}  // namespace Amanzi

