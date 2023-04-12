/*
  Copyright 2010-202x held jointly by participating institutions.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors:
*/

#include "errors.hh"
#include "exceptions.hh"
#include <memory>
#include "HDF5Reader.hh"
#include "FunctionBilinearAndTime.hh"

namespace Amanzi {

FunctionBilinearAndTime::FunctionBilinearAndTime(const std::string& filename,
                                                 const std::string& time_header,
                                                 const std::string& row_header,
                                                 const std::string& row_coordinate,
                                                 const std::string& col_header,
                                                 const std::string& col_coordinate,
                                                 const std::string& val_header)
  : row_header_(row_header),
    col_header_(col_header),
    val_header_(val_header),
    filename_(filename),
    t_before_(-1.0),
    t_after_(-1.0),
    current_interval_(-2)
{
  HDF5Reader reader(filename_);
  reader.ReadData(time_header, times_);

  if (row_coordinate == "x")
    row_index_ = 1;
  else if (row_coordinate == "y")
    row_index_ = 2;
  else if (row_coordinate == "z")
    row_index_ = 3;

  if (col_coordinate == "x")
    col_index_ = 1;
  else if (col_coordinate == "y")
    col_index_ = 2;
  else if (col_coordinate == "z")
    col_index_ = 3;
}


FunctionBilinearAndTime::FunctionBilinearAndTime(const FunctionBilinearAndTime& other)
  : row_header_(other.row_header_),
    col_header_(other.col_header_),
    val_header_(other.val_header_),
    row_index_(other.row_index_),
    col_index_(other.col_index_),
    times_(other.times_),
    filename_(other.filename_),
    t_before_(other.t_before_),
    t_after_(other.t_after_),
    current_interval_(other.current_interval_),
    val_before_(other.val_before_->Clone()),
    val_after_(other.val_after_->Clone())
{}


double
FunctionBilinearAndTime::operator()(const Kokkos::View<double*, Kokkos::HostSpace>& x) const
{
  int interval; double s;
  std::tie(interval, s) = ComputeAndLoadInterval_(x(0));

  if (interval == -1) {
    return (*val_after_)(x);
  } else if (interval == times_.size() - 1) {
    return (*val_before_)(x);
  } else {
    return (*val_after_)(x)*s + (*val_before_)(x) * (1.0 - s);
  }
}


void
FunctionBilinearAndTime::apply(const Kokkos::View<double**>& in, Kokkos::View<double*>& out) const
{
  int interval; double s;
  std::tie(interval, s) = ComputeAndLoadInterval_(in(0,0));

  if (interval == -1) {
    return val_after_->apply(in, out);
  } else if (interval == times_.size() - 1) {
    return val_before_->apply(in, out);
  } else {
    Kokkos::View<double*> out2("FunctionBilinearAndTime workspace", out.extent(0));
    val_before_->apply(in, out);
    val_after_->apply(in, out2);

    Kokkos::parallel_for("FunctionBilinearAndTime::apply", out.extent(0),
                         KOKKOS_LAMBDA(const int& i) {
                           out(i) = (1 - s) * out(i) + s * out2(i);
                         });
  }
}


std::pair<int,double>
FunctionBilinearAndTime::ComputeAndLoadInterval_(double time) const
{
  // get the interval of the current time
  int interval = std::lower_bound(times_.begin(), times_.end(), time) - times_.begin() - 1;
  if (interval != current_interval_) {
    if ((current_interval_ == -2) || (interval != current_interval_ + 1)) {
      // completely uninitialized, or not an increment, load both
      if (interval == -1) {
        val_before_ = nullptr;
        val_after_ = Load_(0);
      } else if (interval == times_.size() - 1) {
        val_before_ = Load_(times_.size() - 1);
        val_after_ = nullptr;
      } else {
        val_before_ = Load_(interval);
        val_after_ = Load_(interval + 1);
      }

    } else {
      // an increment, we can move after --> before
      val_before_ = std::move(val_after_);
      val_after_ = Load_(interval + 1);
    }
    current_interval_ = interval;
  }

  double s = 0.;
  if (interval == -1) s = -1;
  else if (interval == times_.size() - 1) s = -1;
  else s = (time - times_[interval]) / (times_[interval + 1] - times_[interval]);
  return std::make_pair(interval, s);
}


std::unique_ptr<FunctionBilinear>
FunctionBilinearAndTime::Load_(const int time_index) const
{
  HDF5Reader reader(filename_);

  Kokkos::View<double*, Kokkos::HostSpace> row, col;
  reader.ReadData(row_header_, row);
  reader.ReadData(col_header_, col);

  Kokkos::View<double**, Kokkos::HostSpace> values;
  reader.ReadMatData(val_header_ + "/" + std::to_string(time_index), values);
  return std::make_unique<FunctionBilinear>(row, col, values, row_index_, col_index_);
}

} // namespace Amanzi
