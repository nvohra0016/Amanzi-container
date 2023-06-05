#define REGISTER(c)  \
  template<> Utils::RegisteredFactory<Evaluator, EvaluatorModelCV<c,DefaultDevice>> EvaluatorModelCV<c,DefaultDevice>::reg_(c<cView_type,View_type>::name)


#define REGISTER_BY_MATERIAL(c) \
  template<> Utils::RegisteredFactory<Evaluator, EvaluatorModelCVByMaterial<c,DefaultDevice>> EvaluatorModelCVByMaterial<c,DefaultDevice>::reg_(c<cView_type,View_type>::name + " by material")

