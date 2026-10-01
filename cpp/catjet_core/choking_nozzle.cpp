#include "choking_nozzle.hpp"

#include "brentq.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace catjet {

ChokingNozzle::ChokingNozzle(const std::string& mechanism)
    : solution_(Cantera::newSolution(mechanism)), gas_(solution_->thermo())
{
}

ChokingNozzle::Isentrope ChokingNozzle::prepare(const GasState& state)
{
    if (!(std::isfinite(state.T) && state.T > 0.0 && std::isfinite(state.P) && state.P > 0.0)) {
        throw std::invalid_argument("stagnation T and P must be positive and finite");
    }
    if (state.Y.size() != gas_->nSpecies()) {
        throw std::invalid_argument("GasState Y does not match the nozzle mechanism");
    }
    gas_->setMassFractions(state.Y.data());
    gas_->setState_TP(state.T, state.P);
    return {gas_->enthalpy_mass(), gas_->entropy_mass(), state.P};
}

ChokingNozzle::Point ChokingNozzle::point(const Isentrope& start, double pressure)
{
    if (!(pressure > 0.0 && pressure <= start.p0)) {
        throw std::invalid_argument("isentrope pressure must be in (0, P0]");
    }
    gas_->setState_SP(start.s0, pressure, 1e-13); // frozen Y; P8.3-A1 tolerance
    const double h = gas_->enthalpy_mass();
    const double dh = std::max(0.0, start.h0 - h);
    const double u = std::sqrt(2.0 * dh);
    const double rho = gas_->density();
    return {gas_->temperature(), rho, h, u, rho * u, gas_->soundSpeed()};
}

double ChokingNozzle::critical_pressure(const Isentrope& start)
{
    // For one-dimensional isentropic flow, d(rho*u)/dp=0 at u=a. Find the
    // sign change by pressure halving, without assuming a constant gamma.
    auto sonic = [&](double p) {
        const Point q = point(start, p);
        return q.u * q.u - q.sound_speed * q.sound_speed;
    };
    double lower = start.p0;
    bool bracketed = false;
    for (int n = 0; n < 60; ++n) {
        lower *= 0.5;
        if (sonic(lower) > 0.0) {
            bracketed = true;
            break;
        }
    }
    if (!bracketed) {
        throw std::runtime_error("no mass-flux maximum found on the isentrope");
    }
    const double pstar = brentq(sonic, lower, start.p0, 1e-13 * start.p0);
    const double flux = point(start, pstar).flux;
    // Verify that the sonic root is actually a local maximum of rho*u. This
    // guards against an EOS with multiple branches or a failed sound speed.
    const double step = 1e-5;
    if (point(start, pstar * (1.0 - step)).flux > flux * (1.0 + 1e-10) ||
        point(start, pstar * (1.0 + step)).flux > flux * (1.0 + 1e-10)) {
        throw std::runtime_error("sonic pressure is not a mass-flux maximum");
    }
    return pstar;
}

double ChokingNozzle::critical_pressure(const GasState& stagnation)
{
    return critical_pressure(prepare(stagnation));
}

double ChokingNozzle::mass_flux(const GasState& stagnation, double pressure)
{
    return point(prepare(stagnation), pressure).flux;
}

ChokingNozzleResult ChokingNozzle::run(const GasState& stagnation, double ambient_pressure,
                                      double fixed_area, double Cd, double Cv)
{
    if (!(std::isfinite(ambient_pressure) && ambient_pressure > 0.0)) {
        throw std::invalid_argument("ambient pressure must be positive and finite");
    }
    if (!(std::isfinite(fixed_area) && fixed_area > 0.0)) {
        throw std::invalid_argument("nozzle area must be positive and finite");
    }
    if (!(std::isfinite(Cd) && Cd > 0.0 && std::isfinite(Cv) && Cv > 0.0)) {
        throw std::invalid_argument("Cd and Cv must be positive and finite");
    }
    const Isentrope start = prepare(stagnation);
    const double pstar = critical_pressure(start);
    const Point star = point(start, pstar);
    ChokingNozzleResult r;
    r.critical_pressure = pstar;
    r.critical_mass_flux = star.flux;
    r.area = fixed_area;
    r.Cd = Cd;
    r.Cv = Cv;
    if (ambient_pressure >= start.p0) {
        r.exit = stagnation;
        return r;
    }
    r.choked = ambient_pressure < pstar;
    const double pexit = r.choked ? pstar : ambient_pressure;
    const Point q = point(start, pexit);
    r.mass_flow = Cd * fixed_area * q.flux;
    r.velocity_ideal = q.u;
    r.velocity = Cv * q.u;
    // Velocity loss remains as thermal enthalpy in the actual exit state.
    // The isentropic state q only defines the ideal mass-flux capacity.
    const double target_h = start.h0 - 0.5 * r.velocity * r.velocity;
    gas_->setState_HP(target_h, pexit, 1e-13);
    r.exit = {gas_->temperature(), pexit, stagnation.Y};
    const double energy_residual =
        start.h0 - gas_->enthalpy_mass() - 0.5 * r.velocity * r.velocity;
    r.energy_relative = std::fabs(energy_residual) /
        std::max({1.0, std::fabs(start.h0), 0.5 * r.velocity * r.velocity});
    r.thrust_momentum = r.mass_flow * r.velocity;
    r.thrust_pressure = (pexit - ambient_pressure) * fixed_area;
    r.thrust_total = r.thrust_momentum + r.thrust_pressure;
    return r;
}

DualNozzleResult ChokingNozzle::run_dual(const GasState& core, const GasState& bypass,
                                        double ambient_pressure, double core_area,
                                        double bypass_area, double core_Cd, double core_Cv,
                                        double bypass_Cd, double bypass_Cv)
{
    DualNozzleResult r;
    r.core = run(core, ambient_pressure, core_area, core_Cd, core_Cv);
    r.bypass = run(bypass, ambient_pressure, bypass_area, bypass_Cd, bypass_Cv);
    r.thrust_total = r.core.thrust_total + r.bypass.thrust_total;
    r.mass_flow_total = r.core.mass_flow + r.bypass.mass_flow;
    return r;
}

}  // namespace catjet
