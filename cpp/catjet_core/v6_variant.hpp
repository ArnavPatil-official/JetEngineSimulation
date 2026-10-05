// P8.1 benchmark-only engine for P8-A1.2 variant c (products-only HP
// equilibrium). The G0 engine (v6_engine.*) is unchanged; this class reuses
// its public components and copies only run_full_cycle/run_at_thrust so the
// combustor call can be swapped. Mode Full must reproduce V6Engine at the G0
// tolerance (scripts/phase8/verify_native_variants.py); mode ProductsOnly is
// the variant-c approximation with its own registered tolerance.
#pragma once

#include "v6_engine.hpp"

#include <memory>
#include <string>
#include <vector>

namespace catjet {

// The 12 registered P8-A1 product species, in the Python variant's order.
const std::vector<std::string>& products_only_species();

class V6VariantEngine {
public:
    enum class Equilibrium { Full, ProductsOnly };

    V6VariantEngine(const std::string& mechanism, Equilibrium mode);

    V6Config config;
    long cycle_calls = 0;  // run_full_cycle invocations (counts probe attempts too)

    CombustorResult combustor_run(double T_in, double p_in, const std::string& fuel, double phi,
                                  double efficiency, double heat_loss_fraction);
    CycleResult run_full_cycle(const std::string& fuel, const std::vector<std::string>& fuel_species,
                               double phi, double combustor_efficiency);
    AtThrustResult run_at_thrust(double target_kN, const std::string& fuel,
                                 const std::vector<std::string>& fuel_species, double combustor_efficiency,
                                 double phi_lo = 0.05, double phi_hi = 1.0,
                                 double t4_max_K = 3800.0 * 5.0 / 9.0, double phi_xtol = 1e-12,
                                 double phi_guess = -1.0);

private:
    V6Engine base_;
    Equilibrium mode_;
    std::shared_ptr<Cantera::Solution> mix_sol_, eq_sol_, out_sol_;
    std::shared_ptr<Cantera::ThermoPhase> mix_, eq_, out_, products_;
    std::vector<double> eq_pristine_, out_pristine_, products_pristine_;
    std::vector<size_t> full_index_;
};

}  // namespace catjet
