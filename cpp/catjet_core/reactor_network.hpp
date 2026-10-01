// P8.5 combustor reactor network (docs/phase8_p85_registration.md and
// amendment P8.5-A1): K parallel Gaussian-phi primary PSRs -> quick-quench
// PSR -> lean PSR chain -> frozen dilution, solved zone by zone.
#pragma once

#include "gas_state.hpp"

#include "cantera/core.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace catjet {

// Gauss-Hermite nodes x_k and probability weights w_k = omega_k / sqrt(pi).
void gauss_hermite(int K, std::vector<double>& x, std::vector<double>& w);

struct NetworkParams {
    double phi_pz_design = 1.8;   // primary-zone phi at the engine design point
    double sigma_rel = 0.10;      // sigma_phi / phi_pz
    double alpha_qq = 0.45;       // quick-quench air fraction of burner air
    double volume_scale = 1.0;    // s
    int K = 7;                    // primary PSRs (Gauss-Hermite order)
    int n_lean = 10;              // lean-zone PSR chain length
    bool no_dilution = false;     // G1 limit: alpha_qq := 1 - alpha_pz
};

struct NetworkDesign {
    double alpha_pz = 0.0;  // hardware split, phi_global(design) / phi_pz_design
    double V_ref = 0.0;     // m^3, 4.0 ms reference residence time at design, s = 1
    double far_st = 0.0;    // stoichiometric fuel/air mass ratio
    double rho_mean = 0.0;
};

struct PsrState {
    GasState inlet;
    GasState outlet;
    double mass_flow = 0.0;
    double volume = 0.0;
    double residence_time = 0.0;
    int steady_iterations = 0;
    double final_residual = 0.0;
    bool converged = false;
    bool extinguished = false;
};

struct NetworkResult {
    GasState exit;
    double mass_flow_exit = 0.0;
    double eta_b = 0.0;
    double EI_NOx_g_kg = 0.0;
    double EI_CO_g_kg = 0.0;
    double EI_UHC_g_kg = 0.0;
    double phi_pz = 0.0;
    double alpha_pz = 0.0;
    double alpha_qq = 0.0;
    double alpha_dil = 0.0;
    std::vector<double> phi_k;
    std::vector<PsrState> primary;
    PsrState quench;
    std::vector<PsrState> lean;
    GasState lean_exit;
    double energy_relative = 0.0;     // whole network
    double element_relative = 0.0;    // whole network
    double max_mixer_energy_relative = 0.0;
    double max_mixer_element_relative = 0.0;
    bool all_converged = true;
    bool any_extinguished = false;
};

class ReactorNetwork {
public:
    // fuel: a species name (POSF10325 for A2NOx) or a molar composition string.
    // n_solutions: Cantera Solutions owned (one per primary-PSR thread).
    ReactorNetwork(const std::string& mechanism, const std::string& fuel, int n_solutions = 9);

    NetworkDesign design(double T3, double P3, double m_air, double m_fuel,
                         const NetworkParams& p, double pressure_loss = 0.045);
    NetworkResult run(double T3, double P3, double m_air, double m_fuel,
                      const NetworkParams& p, const NetworkDesign& d,
                      double pressure_loss = 0.045);
    // HP equilibrium of the total inlet (air + liquid-basis fuel) at P.
    GasState equilibrium(double T3, double P, double m_air, double m_fuel);
    double lhv_mass(const std::string& species);  // J/kg at 298.15 K, gas basis
    size_t n_species() const;
    std::vector<std::string> species_names() const;

    static constexpr double kVaporization = 360000.0;  // J/kg, v7 value
    static constexpr double kTauRef = 4.0e-3;          // s
    static constexpr double kFracPZ = 0.30, kFracQQ = 0.10, kFracLean = 0.60;

private:
    std::string mechanism_, fuel_;
    std::vector<std::shared_ptr<Cantera::Solution>> sols_;
    std::vector<double> fuel_Y_;  // fuel mass fractions (mechanism order)
    std::vector<double> lhv_;   // per species, J/kg (0 if not counted)
    std::vector<bool> counted_, uhc_;
    GasState air(double T, double P);
    GasState fuel_gas(double T, double P);
    double h_of(Cantera::ThermoPhase& th, const GasState& s);
    GasState mix(Cantera::ThermoPhase& th, const std::vector<std::pair<GasState, double>>& streams,
                 double P, double& energy_rel, double& element_rel);
    GasState feed(Cantera::ThermoPhase& th, double T3, double P, double m_air, double m_fuel);
    PsrState psr(std::shared_ptr<Cantera::Solution> sol, const GasState& inlet,
                 double mass_flow, double volume);
};

}  // namespace catjet
