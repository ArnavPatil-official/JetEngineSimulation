// P8.4 two-shaft high-bypass turbofan, design point and off-design
// matching (docs/phase8_p84_registration.md, amendment P8.4-A1). Element
// equations mirror the pinned pyCycle 4.4.0 HBTF example; SI internally,
// pyCycle English units only where corrected quantities and maps need them.
#pragma once

#include "maps.hpp"

#include "cantera/core.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace catjet {

enum class ThermoMode { Matched, Production };

struct Flow {
    double W = 0.0;   // kg/s
    double Tt = 0.0;  // K
    double Pt = 0.0;  // Pa
    double ht = 0.0;  // J/kg
    double St = 0.0;  // J/kg/K
    std::vector<double> Y;
};

struct ThroatState {
    double Ps = 0.0, Ts = 0.0, V = 0.0, rho = 0.0, area = 0.0, MN = 0.0;
};

struct Bleed {
    std::string name;
    double frac_W = 0.0, frac_P = 0.0, frac_work = 0.0;
};

struct CompressorSpec {
    std::string name;
    ComponentMap map;
    double PR_des = 0.0, eff_des = 0.0;
    std::vector<Bleed> bleeds;   // fractions of inlet flow
};

struct TurbineSpec {
    std::string name;
    ComponentMap map;
    double eff_des = 0.0;
};

struct HbtfSpec {
    // flight / design targets (pyCycle units converted at the binding)
    double alt_m = 0.0, MN = 0.0, dTs_K = 0.0;
    double Fn_des_N = 0.0, T4_max_K = 0.0;
    double N_lp_des = 0.0, N_hp_des = 0.0;   // rpm
    double BPR_des = 0.0;
    double ram_recovery = 0.999;
    double dPqP_duct4 = 0.0, dPqP_duct6 = 0.0, dPqP_burner = 0.0, dPqP_duct11 = 0.0,
           dPqP_duct13 = 0.0, dPqP_duct15 = 0.0;
    double Cv_core = 1.0, Cv_byp = 1.0;
    double frac_byp_bleed = 0.0;
    double cool3_frac_W = 0.0, cool4_frac_W = 0.0;            // bld3, of HPC exit flow
    double cool3_frac_P = 1.0, cool4_frac_P = 0.0;            // HPT re-entry
    double cool1_frac_P_lpt = 1.0, cool2_frac_P_lpt = 0.0;    // LPT re-entry
    double HPX_W = 0.0;
    CompressorSpec fan, lpc, hpc;   // hpc bleeds: cool1, cool2, cust
    TurbineSpec hpt, lpt;
    // pyCycle US 1976 table (ft, degR, psi), Akima-interpolated as pyCycle does
    std::vector<double> atm_alt_ft, atm_T_R, atm_P_psi;
};

struct CompressorScalars { double s_Nc = 1, s_PR = 1, s_eff = 1, s_Wc = 1, Wc_des = 1; };
struct TurbineScalars { double s_Np = 1, s_PR = 1, s_eff = 1, s_Wp = 1, Wp_des = 1; };

struct DesignData {
    CompressorScalars fan, lpc, hpc;
    TurbineScalars hpt, lpt;
    double A_core = 0.0, A_byp = 0.0;   // nozzle throat areas, m^2
    double P_hpt = 0.0, P_lpt = 0.0;    // design shaft powers, W (normalisation)
    double W_des = 0.0;
    bool valid = false;
};

struct CycleOutputs {
    std::map<std::string, double> scalars;   // W, FAR, OPR, Fn_N, Fg_N, TSFC, BPR, Tt3, Tt4, ...
    std::map<std::string, Flow> stations;
    double mass_closure = 0.0, energy_closure = 0.0, element_closure = 0.0;
    bool extrapolated = false;
    std::vector<std::string> extrapolated_maps;
};

struct SolveResult {
    bool converged = false;
    int iterations = 0;
    std::vector<double> x;          // unknowns
    std::vector<double> residuals;  // scaled
    std::vector<double> norm_history;
    std::string reason;
    CycleOutputs out;
};

class Hbtf {
public:
    // mechanism: data/thermo/pycycle_janaf.yaml (Matched) or CRECK (Production).
    // fuel: Matched -> element formula "C:12,H:23" with pyCycle element weights;
    //       Production -> molar composition of the surrogate.
    Hbtf(const std::string& mechanism, ThermoMode mode, const std::string& air,
         const std::string& fuel, std::map<std::string, double> fuel_element_weights = {});
    HbtfSpec spec;
    DesignData design;

    // Design: unknowns W [kg/s], FAR, PR_hpt, PR_lpt.
    SolveResult solve_design(std::vector<double> guess);
    // Off-design: unknowns W, FAR, BPR, N_lp, N_hp, R_fan, R_lpc, R_hpc, PR_hpt, PR_lpt.
    // throttle "T4" (target spec.T4_max_K) or "Fn" (target Fn_target_N).
    SolveResult solve_offdesign(std::vector<double> guess, double alt_m, double MN, double dTs_K,
                                const std::string& throttle, double target);
    // One evaluation (for tests): residual vector for the given unknowns.
    std::vector<double> residuals(const std::vector<double>& x, bool design_mode,
                                  CycleOutputs* out = nullptr);

    double us1976_T(double alt_m) const;   // K (Akima on pyCycle's table)
    double us1976_P(double alt_m) const;   // Pa

private:
    ThermoMode mode_;
    std::shared_ptr<Cantera::Solution> sol_;
    std::shared_ptr<Cantera::ThermoPhase> gas_;
    std::vector<double> Y_air_;
    std::string fuel_;
    std::vector<double> fuel_element_moles_per_kg_;   // Matched mode, per element index
    std::vector<double> Y_fuel_;                      // Production mode
    // current off-design operating condition
    double alt_ = 0.0, MN_ = 0.0, dTs_ = 0.0, throttle_target_ = 0.0;
    std::string throttle_ = "T4";

    void set_Y(const std::vector<double>& Y);
    Flow state_TP(const std::vector<double>& Y, double T, double P, double W);
    Flow state_hP(const std::vector<double>& Y, double h, double P, double W);
    Flow state_SP(const std::vector<double>& Y, double s, double P, double W);
    double sound_speed(const std::vector<double>& Y, double s, double P);
    ThroatState static_at_MN(const Flow& f, double MN);
    ThroatState static_at_Ps(const Flow& f, double Ps);
    Flow mix(const std::vector<Flow>& flows, double P);
    std::vector<double> burner_Y(const Flow& air, double W_fuel);
    double fuel_enthalpy(const Flow& air) ;
    SolveResult newton(std::vector<double> x, const std::vector<double>& lo,
                       const std::vector<double>& hi, const std::vector<double>& scale,
                       bool design_mode);
};

}  // namespace catjet
