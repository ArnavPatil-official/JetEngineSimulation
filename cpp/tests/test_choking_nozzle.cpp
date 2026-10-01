#include "catjet_core/choking_nozzle.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

bool near_rel(double x, double y, double tol)
{
    return std::fabs(x - y) <= tol * std::max(std::fabs(x), std::fabs(y));
}

}  // namespace

TEST_CASE("P8.3 constant-cp critical ratio is the analytic convergent-nozzle limit")
{
    catjet::ChokingNozzle nozzle(CATJET_PERFECT_GAS);
    catjet::GasState inlet{1100.0, 400000.0, {1.0}};
    auto gas = Cantera::newSolution(CATJET_PERFECT_GAS)->thermo();
    gas->setMassFractions(inlet.Y.data());
    gas->setState_TP(inlet.T, inlet.P);
    const double cp = gas->cp_mass();
    const double R = Cantera::GasConstant / gas->meanMolecularWeight();
    const double gamma = cp / (cp - R);
    const double expected = std::pow(2.0 / (gamma + 1.0), gamma / (gamma - 1.0));
    const double pstar = nozzle.critical_pressure(inlet);
    CHECK(near_rel(pstar / inlet.P, expected, 1e-10));
    const double peak = nozzle.mass_flux(inlet, pstar);
    CHECK(nozzle.mass_flux(inlet, pstar * (1.0 - 1e-4)) < peak);
    CHECK(nozzle.mass_flux(inlet, pstar * (1.0 + 1e-4)) < peak);
}

TEST_CASE("P8.3 mass flow and force are continuous across choking")
{
    catjet::ChokingNozzle nozzle(CATJET_PERFECT_GAS);
    catjet::GasState inlet{1100.0, 400000.0, {1.0}};
    const double pstar = nozzle.critical_pressure(inlet);
    const auto lo = nozzle.run(inlet, pstar * (1.0 - 1e-9), 0.08, 0.96, 0.95);
    const auto hi = nozzle.run(inlet, pstar * (1.0 + 1e-9), 0.08, 0.96, 0.95);
    CHECK(lo.choked);
    CHECK_FALSE(hi.choked);
    CHECK(near_rel(lo.mass_flow, hi.mass_flow, 1e-8));
    CHECK(near_rel(lo.thrust_total, hi.thrust_total, 1e-8));
    CHECK(lo.energy_relative < 1e-10);
    CHECK(hi.energy_relative < 1e-10);
    CHECK(lo.exit.Y == inlet.Y);
    CHECK(hi.exit.Y == inlet.Y);
}

TEST_CASE("P8.3 unchoked ideal limit reproduces v6 fully expanded thrust")
{
    catjet::ChokingNozzle nozzle(CATJET_PERFECT_GAS);
    catjet::GasState inlet{1100.0, 400000.0, {1.0}};
    const double pamb = 300000.0;
    const double target_mass_flow = 50.0;
    const double flux = nozzle.mass_flux(inlet, pamb);
    const double area = target_mass_flow / flux; // v6 effective area at this point
    const auto r = nozzle.run(inlet, pamb, area, 1.0, 1.0);
    auto gas = Cantera::newSolution(CATJET_PERFECT_GAS)->thermo();
    gas->setMassFractions(inlet.Y.data());
    gas->setState_TP(inlet.T, inlet.P);
    const double cp = gas->cp_mass();
    const double R = Cantera::GasConstant / gas->meanMolecularWeight();
    const double gamma = cp / (cp - R);
    const double exponent = (gamma - 1.0) / gamma;
    const double v6_u = std::sqrt(2.0 * cp * inlet.T *
                                (1.0 - std::pow(pamb / inlet.P, exponent)));
    CHECK_FALSE(r.choked);
    CHECK(near_rel(r.mass_flow, target_mass_flow, 1e-12));
    CHECK(near_rel(r.thrust_total, target_mass_flow * v6_u, 1e-10));
    CHECK(r.thrust_pressure == 0.0);
    CHECK(r.energy_relative < 1e-10);
}

TEST_CASE("P8.3 core and bypass nozzles use separate fixed areas")
{
    catjet::ChokingNozzle nozzle(CATJET_PERFECT_GAS);
    catjet::GasState core{1100.0, 400000.0, {1.0}};
    catjet::GasState bypass{500.0, 160000.0, {1.0}};
    const auto r = nozzle.run_dual(core, bypass, 101325.0, 0.08, 0.5, 0.96, 0.98, 0.97, 0.99);
    CHECK(r.core.choked);
    CHECK_FALSE(r.bypass.choked);
    CHECK(near_rel(r.mass_flow_total, r.core.mass_flow + r.bypass.mass_flow, 1e-14));
    CHECK(near_rel(r.thrust_total, r.core.thrust_total + r.bypass.thrust_total, 1e-14));
    CHECK(r.core.energy_relative < 1e-10);
    CHECK(r.bypass.energy_relative < 1e-10);
}

TEST_CASE("P8.3 real-gas air mixture retains all species and closes energy")
{
    auto gas = Cantera::newSolution(CATJET_MECHANISM)->thermo();
    gas->setState_TPX(1100.0, 400000.0, "O2:1,N2:3.76");
    std::vector<double> Y(gas->nSpecies());
    gas->getMassFractions(Y.data());
    catjet::GasState inlet{1100.0, 400000.0, Y};
    catjet::ChokingNozzle nozzle(CATJET_MECHANISM);
    const auto r = nozzle.run(inlet, 101325.0, 0.08, 0.96, 0.95);
    CHECK(r.choked);
    CHECK(r.exit.Y == Y);
    CHECK(r.energy_relative < 1e-10);
}
