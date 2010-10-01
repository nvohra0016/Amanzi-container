/* -*-  mode: c++; c-default-style: "google"; indent-tabs-mode: nil -*- */
#ifndef __ACTIVITY_MODEL_HPP__
#define __ACTIVITY_MODEL_HPP__

// Base class for activity calculations

#include <vector>

#include "Species.hpp"
#include "AqueousEquilibriumComplex.hpp"

class ActivityModel {
 public:
  ActivityModel();
  virtual ~ActivityModel();

  void calculateIonicStrength(std::vector<Species> primarySpecies,
                              std::vector<AqueousEquilibriumComplex> secondarySpecies);
  void calculateActivityCoefficients(std::vector<Species> &primarySpecies,
                                     std::vector<AqueousEquilibriumComplex> &secondarySpecies);
  virtual double evaluate(const Species& species) = 0;

  double ionic_strength(void) const { return this->I_; }

  virtual void display(void) const = 0;

 protected:
  double log_to_ln(double d) { return d*2.30258509299; }
  double ln_to_log(double d) { return d*0.434294481904; }

  void ionic_strength(double d) { this->I_ = d; }

  double I_;  // ionic strength

 private:
};

#endif  // __ACTIVITY_MODEL_HPP__

