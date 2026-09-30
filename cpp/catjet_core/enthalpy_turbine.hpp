// P8.2: composition-carrying state, conservative mixing and frozen-Y
// enthalpy-polytropic expansion. See docs/phase8_p82_registration.md.
#pragma once

#include "gas_state.hpp"
#include "v6_engine.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace catjet {

struct GasProperties {
    double h = 0.0;
    double s = 0.0;
    double cp = 0.0;
    double R = 0.0;
    double gamma = 0.0;
    double rho = 0.0;
    std::vector<double> elements;  // element mass fractions in mechanism order
};

struct MassStream {
    GasState state;
    double mass_flow = 0.0;
};

struct MixResult {
    MassStream stream;
    double mass_relative = 0.0;
    double energy_relative = 0.0;
    double element_relative = 0.0;
};

struct ExpansionResult {
    MassStream stream;
    double requested_work = 0.0;
    double actual_work = 0.0;
    double energy_relative = 0.0;
    double element_relative = 0.0;
    int steps = 0;
};

class GasThermo {
public:
    explicit GasThermo(const std::string& mechanism);
    GasState from_moles(double T, double P, const std::string& composition);
    GasState at_enthalpy(double h, double P, const std::vector<double>& Y);
    GasState compress(const GasState& in, double pressure_ratio, double eta);
    GasProperties properties(const GasState& state);
    MixResult mix(const MassStream& a, const MassStream& b, double P);
    ExpansionResult expand_for_work(const MassStream& in, double work_W,
                                    double eta_poly, int steps = 50,
                                    double constant_cp = 0.0, double constant_R = 0.0);
    size_t n_species() const;

private:
    std::shared_ptr<Cantera::Solution> solution_;
    std::shared_ptr<Cantera::ThermoPhase> gas_;
    void set(const GasState& state);
    std::vector<double> element_mass_fractions();
};

struct P82Config {
    V6Config base;
    double vaporization_J_kg = 360000.0;
    double ngv_fraction = 0.0641;
    double rotor_fraction = 0.0275;
    int pressure_steps = 50;
};

struct P82CycleResult {
    CycleResult cycle;
    std::map<std::string, MassStream> stations;
    std::map<std::string, ExpansionResult> stages;
    double burner_heat_rejection_W = 0.0;
    double max_mass_relative = 0.0;
    double max_energy_relative = 0.0;
    double max_element_relative = 0.0;
};

struct P82AtThrustResult {
    P82CycleResult result;
    ThrustMatch match;
};

class P82Engine {
public:
    explicit P82Engine(const std::string& mechanism);
    P82Config config;
    P82CycleResult run_full_cycle(const std::string& fuel,
                                  const std::vector<std::string>& fuel_species,
                                  double phi, double combustor_efficiency,
                                  int ablation_level = 2);
    P82AtThrustResult run_at_thrust(double target_kN, const std::string& fuel,
                                    const std::vector<std::string>& fuel_species,
                                    double combustor_efficiency, int ablation_level = 2,
                                    double phi_lo = 0.05, double phi_hi = 1.0,
                                    double t4_max_K = 3800.0 * 5.0 / 9.0,
                                    double phi_xtol = 1e-12, double phi_guess = -1.0);
    GasThermo& thermo() { return thermo_; }

private:
    V6Engine v6_;
    GasThermo thermo_;
};

}  // namespace catjet
