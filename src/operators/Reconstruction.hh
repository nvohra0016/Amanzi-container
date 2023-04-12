/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Konstantin Lipnikov (lipnikov@lanl.gov)
*/

/*!

A reconstruction of discrete fields is used to increase accuracy of discrete models.
The reconstruction can be either unconstrained or limited. 
Amanzi supports a variety of state-of-the-art reconstruction and limiting algorithms 
and their extensions for various PKs.

* `"reconstruction`" [list] describes parameters used by reconstruction algorithms.

 * `"method`" [string] specifies a reconstruction method. Available option is
   `"cell-based`" (default).

 * `"polynomial order`" [int] defines the polynomial order of the reconstructed function. 
   Default is 1.

 * `"weight`" [string] defined weight fr reconstruction. Available options are
   `"constant`" (default) and `"inverse distance`".

 * `"limiter`" [string] specifies limiting method. Available options are 
   `"Barth-Jespersen`" (default), `"Michalak-Gooch`", `"tensorial`", and `"Kuzmin`". 

 * `"limiter stencil`" [string] specifies stencil for calculating local bounds. Available 
   options are `"face to cells`", `"cell to closets cells`", `"cell to all cells`",
   and `"node to cells`".
   For a square mesh, the above options define stencils of size 2, 5, 9, and 4,
   respectively.
   Option `"face to cells`" is default for `"Barth-Jespersen`", `"Michalak-Gooch`", 
   and `"tensorial`".  Option `"node to cells`" is default for `"Kuzmin`".

 * `"limiter points`" [int] specifies the number of integration points (Gauss points in 2D) 
   on face where limiting occurs. Default is 1. Limited to 2D.

 * `"limiter location`" [string] defines geometry entity where the *limiter points*
   are located. Available options are `"node`", `"face`", and `"cell`".
   Option `"node`" is default for `"node to cells`" stencil.
   Option `"face`" is default for other stencils.

 * `"limiter cfl`" [double] is a safety factor (less than 1) applied to the limiter.
   Default value is 1.

 * `"use external bounds`" [bool] specifies if bounds for limiters are provided by 
   the hosting application. Default is `"false`".`

 * `"limiter extension for transport`" [bool] adds additional corrections to 
   limiters required by the transport PK. Default value is *false*.

.. code-block:: xml

  <ParameterList name="reconstruction">
    <Parameter name="method" type="string" value="cell-based"/>
    <Parameter name="polynomial order" type="int" value="1"/>
    <Parameter name="weight" type="string" value="inverse distance"/>
    <Parameter name="limiter" type="string" value="tensorial"/>
    <Parameter name="limiter extension for transport" type="bool" value="false"/>
    <Parameter name="limiter stencil" type="string" value="face to cells"/>
    <Parameter name="limiter points" type="int" value="0"/>
  </ParameterList>

*/

#ifndef AMANZI_RECONSTRUCTION_HH_
#define AMANZI_RECONSTRUCTION_HH_

#include <vector>

#include "Teuchos_RCP.hpp"

#include "MeshFramework.hh"
#include "Point.hh"

namespace Amanzi {
namespace Operators {

class Reconstruction {
 public:
  Reconstruction(){};
  Reconstruction(Teuchos::RCP<const Amanzi::AmanziMesh::Mesh> mesh)
    : mesh_(mesh), field_(Teuchos::null), component_(0){};
  virtual ~Reconstruction() = default;

  // main members
  virtual void Init(Teuchos::RCP<const Epetra_MultiVector> field, Teuchos::ParameterList& plist)
  {
    field_ = field;
  }
  virtual void
  Init(Teuchos::RCP<const Epetra_MultiVector> field, Teuchos::ParameterList& plist, int component)
  {
    field_ = field;
    component_ = component;
  }
  virtual void ComputeGradient() = 0;

 protected:
  Teuchos::RCP<const AmanziMesh::Mesh> mesh_;
  Teuchos::RCP<const Epetra_MultiVector> field_;
  int component_;
};

} // namespace Operators
} // namespace Amanzi

#endif
