// P8.3: fixed-area, frozen-composition convergent nozzles. The thermodynamic
// state is supplied by P8.2 GasState. This module is separate from the frozen
// v6 nozzle so G0 remains independently reproducible.
#pragma once

#include "gas_state.hpp"

#include "cantera/core.h"

#include <memory>
#include <string>

namespace catjet {

struct ChokingNozzleResult {
    GasState exit;
    double critical_pressure = 0.0;
    double critical_mass_flux = 0.0;
    double area = 0.0;
    double Cd = 0.0;
    double Cv = 0.0;
    double mass_flow = 0.0;
    double velocity_ideal = 0.0;
    double velocity = 0.0;
    double thrust_momentum = 0.0;
    double thrust_pressure = 0.0;
    double thrust_total = 0.0;
    double energy_relative = 0.0;
    bool choked = false;
};

struct DualNozzleResult {
    ChokingNozzleResult core;
    ChokingNozzleResult bypass;
    double thrust_total = 0.0;
    double mass_flow_total = 0.0;
};

class ChokingNozzle {
public:
    explicit ChokingNozzle(const std::string& mechanism);

    // Critical pressure is the maximum of rho * sqrt(2(h0-h)) along the
    // frozen-composition isentrope. It is located with the equivalent sonic
    // condition u^2=a^2, then checked against neighbouring mass fluxes.
    double critical_pressure(const GasState& stagnation);
    double mass_flux(const GasState& stagnation, double pressure);

    ChokingNozzleResult run(const GasState& stagnation, double ambient_pressure,
                            double fixed_area, double Cd, double Cv);
    DualNozzleResult run_dual(const GasState& core, const GasState& bypass,
                              double ambient_pressure, double core_area,
                              double bypass_area, double core_Cd, double core_Cv,
                              double bypass_Cd, double bypass_Cv);

private:
    std::shared_ptr<Cantera::Solution> solution_;
    std::shared_ptr<Cantera::ThermoPhase> gas_;

    struct Isentrope {
        double h0;
        double s0;
        double p0;
    };
    Isentrope prepare(const GasState& stagnation);
    struct Point {
        double T;
        double rho;
        double h;
        double u;
        double flux;
        double sound_speed;
    };
    Point point(const Isentrope& start, double pressure);
    double critical_pressure(const Isentrope& start);
};

}  // namespace catjet
