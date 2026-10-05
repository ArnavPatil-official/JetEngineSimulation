// P8.1: the v6 thrust-matched cycle, ported from integrated_engine.py
// (IntegratedTurbofanEngine.run_full_cycle / run_at_thrust with the analytic
// turbine and nozzle), simulation/compressor/compressor.py, simulation/fan.py,
// simulation/combustor/combustor.py and EmissionsEstimator.estimate_nox.
// Operation order follows the Python source so results agree to rounding;
// G0 (docs/phase8_registration.md) checks 1e-9 against the frozen v6 rows.
#pragma once

#include "cantera/core.h"

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace catjet {

// integrated_engine.CycleDoesNotClose
struct CycleDoesNotClose : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// integrated_engine.ThrustTargetUnreachable (reason + diagnostics)
struct ThrustTargetUnreachable : std::runtime_error {
    std::string reason;
    double target_kN;
    std::map<std::string, double> info;
    ThrustTargetUnreachable(const std::string& why, double target, std::map<std::string, double> inf);
};

// design_point + component settings read by the Python cycle
struct V6Config {
    double mass_flow_core = 79.9;
    double bypass_ratio = 9.1;
    double fpr = 1.45;
    double eta_fan = 0.90;
    double pi_c = 43.2;
    double combustor_pressure_loss = 0.0;
    double combustor_heat_loss_fraction = 0.0;
    double combustor_air_fraction = 1.0;
    double A_combustor_exit = 0.207;
    double A_nozzle_exit = 0.340;
    double P_ambient = 101325.0;
    double T_ambient = 288.15;
    double eta_c = 0.86;          // compressor.eta_c
    double eta_polytropic = 0.9;  // turbine_design['eta_polytropic']
    // EmissionsEstimator NOx correlation (fitted in Python, passed in)
    double nox_A = 0.0, nox_B = 0.0, nox_C = 0.0;
};

struct CompressorResult { double T_out, p_out, h_in, h_out, work_specific; };
struct FanResult { double T_exit, p_exit, dT, work_total, u_bypass_exit, thrust_bypass; };
struct CombustorResult { double T_out, p_out, h_out, cp_out, R_out, gamma_out; std::vector<double> Y_out; };
struct TurbineResult { double rho, u, p, T, work_specific, work_total, cp, R, gamma; };
struct NozzleResult { double rho, u, p, T, thrust_total, thrust_momentum, thrust_pressure, A_exit_effective; };

struct CycleResult {
    CompressorResult compressor;
    CombustorResult combustor;  // after the dilution mix when beta < 1
    TurbineResult turbine;
    NozzleResult nozzle;
    bool has_fan = false;
    FanResult fan{};
    double fuel_air_ratio, fuel_mass_flow, total_mass_flow, bypass_mass_flow, total_air_mass_flow;
    double thrust_N, thrust_kN, thrust_core_kN, thrust_bypass_kN;
    double tsfc_SI, tsfc_mg_per_Ns, specific_thrust_Ns_kg, fan_work_W, thermal_efficiency;
    double nox_g_s;
};

struct ThrustMatch {
    double phi, target_kN, residual_kN, t4_K;
    int n_cycle_evaluations;
    std::map<std::string, double> info;
};

struct AtThrustResult {
    CycleResult cycle;
    ThrustMatch match;
};

class V6Engine {
public:
    // mechanism: path to the CRECK YAML (data/creck_c1c16_full.yaml)
    explicit V6Engine(const std::string& mechanism);

    V6Config config;

    // components (each mirrors the Python method of the same name)
    CompressorResult run_compressor(double T_in, double p_in);
    FanResult run_fan(double T0, double p0, double m_dot_bypass) const;
    double fuel_air_ratio(const std::string& fuel, const std::vector<std::string>& fuel_species, double phi);
    CombustorResult combustor_run(double T_in, double p_in, const std::string& fuel, double phi,
                                  double efficiency, double heat_loss_fraction);
    TurbineResult run_turbine_analytic(const CombustorResult& in, double m_dot, double target_work_total) const;
    NozzleResult run_nozzle(const TurbineResult& in, double m_dot) const;
    double estimate_nox(double OPR, double m_dot_fuel) const;

    CycleResult run_full_cycle(const std::string& fuel, const std::vector<std::string>& fuel_species,
                               double phi, double combustor_efficiency);

    // run_at_thrust; phi_guess < 0 means None
    AtThrustResult run_at_thrust(double target_kN, const std::string& fuel,
                                 const std::vector<std::string>& fuel_species, double combustor_efficiency,
                                 double phi_lo = 0.05, double phi_hi = 1.0,
                                 double t4_max_K = 3800.0 * 5.0 / 9.0, double phi_xtol = 1e-12,
                                 double phi_guess = -1.0);

private:
    std::shared_ptr<Cantera::Solution> gas_sol_, eq_sol_, out_sol_, far_sol_;
    std::shared_ptr<Cantera::ThermoPhase> gas_, eq_, out_, far_;
    std::vector<double> eq_pristine_, out_pristine_, far_pristine_;
    // Python ThermoPhase.__composition_to_array(comp, 'mole')
    std::vector<double> composition_to_array(Cantera::ThermoPhase& th, const std::string& comp);
};

}  // namespace catjet
