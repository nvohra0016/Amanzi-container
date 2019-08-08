/* -*-  mode: c++; indent-tabs-mode: nil -*- */
/*
  License: see $ATS_DIR/COPYRIGHT
  Author: Ahmad Jan (jana@ornl.gov)
*/

//! Evaluates snow melt 
//! Source: USDA - Natural Resources Conservation Service, 
//! National Engineering Handbook (NEH) part 630, Chapter 11
/*!

Requires the following dependencies:

* `"air temperature key`" ``[string]`` **DOMAIN-air_temperature**
* `"precipitation snow key`" ``[string]`` **DOMAIN-precipitation_snow**
*/

#include "Key.hh"
#include "snow_meltrate_evaluator.hh"

namespace Amanzi {
namespace Flow {
namespace Relations {


SnowMeltRateEvaluator::SnowMeltRateEvaluator(Teuchos::ParameterList& plist) :
    SecondaryVariableFieldEvaluator(plist),
    compatibility_checked_(false)
{
  melt_rate_ = plist.get<double>("snow melt rate [mm/(day C)]", 2.74);
  snow_transition_depth_ = plist.get<double>("snow-ground transition depth [m]", 0.02);

  domain_ = Keys::getDomain(my_key_);
  if (domain_ == "snow") {
    domain_surf_ = "surface";
  } else if (boost::starts_with(domain_, "snow_")) {
    domain_surf_ = std::string("surface_") + domain_.substr(5,domain_.size());
  } else {
    Errors::Message msg("SnowMeltRateEvaluator: not sure how to interpret domains.");
    Exceptions::amanzi_throw(msg);    
  }

  at_key_ = Keys::readKey(plist, domain_surf_, "air temperature", "air_temperature");
  dependencies_.insert(at_key_);

  snow_key_ = Keys::readKey(plist, domain_, "snow water equivalent", "water_equivalent");
  dependencies_.insert(snow_key_);
}

// Required methods from SecondaryVariableFieldEvaluator
void
SnowMeltRateEvaluator::EvaluateField_(const Teuchos::Ptr<State>& S,
        const Teuchos::Ptr<CompositeVector>& result)
{
  const auto& air_temp = *S->GetFieldData(at_key_)->ViewComponent("cell", false);
  const auto& swe = *S->GetFieldData(snow_key_)->ViewComponent("cell", false);
  auto& res = *result->ViewComponent("cell", false);
  
  for (int c=0; c!=res.MyLength(); ++c) {
    double melt_rate = melt_rate_ * 0.001 / 86400.; // convert mm/day to m/s 
    if (air_temp[0][c] > 273.15) {
      res[0][c] = melt_rate * (air_temp[0][c] - 273.15);

      if (swe[0][c] < snow_transition_depth_) {
        res[0][c] *= swe[0][c] / snow_transition_depth_;
      }
      
    } else {
      res[0][c] = 0.0;
    }
  }
}



} //namespace
} //namespace
} //namespace

