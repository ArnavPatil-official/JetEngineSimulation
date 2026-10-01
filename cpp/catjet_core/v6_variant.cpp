// See v6_variant.hpp. run_full_cycle and run_at_thrust are copies of the G0
// V6Engine methods (v6_engine.cpp at 74f53c8) with the combustor call routed
// through this class; everything else calls the unchanged V6Engine.
#include "v6_variant.hpp"

#include "brentq.hpp"

#include "cantera/thermo/IdealGasPhase.h"
#include "cantera/thermo/Species.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <variant>

namespace catjet {

namespace {

// Copies of the v6_engine.cpp helpers (diagnostic strings only).
std::string py_repr(double x)
{
    if (std::isnan(x)) return "nan";
    if (std::isinf(x)) return x > 0 ? "inf" : "-inf";
    char buf[64];
    auto res = std::to_chars(buf, buf + sizeof(buf), x, std::chars_format::scientific);
    std::string sci(buf, res.ptr);
    bool neg = sci[0] == '-';
    std::string body = neg ? sci.substr(1) : sci;
    auto epos = body.find('e');
    std::string mant = body.substr(0, epos);
    int exp10 = std::stoi(body.substr(epos + 1));
    std::string digits;
    for (char c : mant) {
        if (c != '.') digits += c;
    }
    std::string out;
    if (exp10 < -4 || exp10 >= 16) {
        out = digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        char e[16];
        std::snprintf(e, sizeof(e), "e%c%02d", exp10 < 0 ? '-' : '+', std::abs(exp10));
        out += e;
    } else if (exp10 < 0) {
        out = "0." + std::string(-exp10 - 1, '0') + digits;
    } else {
        int intlen = exp10 + 1;
        if ((int)digits.size() <= intlen) {
            out = digits + std::string(intlen - digits.size(), '0') + ".0";
        } else {
            out = digits.substr(0, intlen) + "." + digits.substr(intlen);
        }
    }
    return neg ? "-" + out : out;
}

std::string fmt(const char* f, double x)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), f, x);
    return buf;
}

std::vector<double> composition_to_array(Cantera::ThermoPhase& th, const std::string& comp)
{
    std::vector<double> original(th.stateSize());
    th.saveState(original);
    double T = th.temperature();
    double P = th.pressure();
    th.setMoleFractionsByName(comp);
    th.setState_TP(T, P);
    std::vector<double> X(th.nSpecies());
    th.getMoleFractions(X.data());
    th.restoreState(original);
    return X;
}

}  // namespace

const std::vector<std::string>& products_only_species()
{
    static const std::vector<std::string> names{"N2", "O2", "AR", "CO2", "H2O", "CO",
                                                "H2", "OH", "H", "O", "HO2", "H2O2"};
    return names;
}

V6VariantEngine::V6VariantEngine(const std::string& mechanism, Equilibrium mode)
    : base_(mechanism), mode_(mode)
{
    mix_sol_ = Cantera::newSolution(mechanism);
    eq_sol_ = Cantera::newSolution(mechanism);
    out_sol_ = Cantera::newSolution(mechanism);
    mix_ = mix_sol_->thermo();
    eq_ = eq_sol_->thermo();
    out_ = out_sol_->thermo();
    eq_pristine_.resize(eq_->stateSize());
    out_pristine_.resize(out_->stateSize());
    eq_->saveState(eq_pristine_);
    out_->saveState(out_pristine_);

    auto products = std::make_shared<Cantera::IdealGasPhase>();
    for (const auto& name : products_only_species()) {
        // Shared Species objects, as Python's ct.Solution(species=[full.species(n)...]).
        std::shared_ptr<Cantera::Species> sp = eq_->species(name);
        for (const auto& [element, count] : sp->composition) {
            if (products->elementIndex(element, false) == Cantera::npos) products->addElement(element);
        }
        products->addSpecies(sp);
        full_index_.push_back(eq_->speciesIndex(name));
    }
    products->initThermo();
    products->setState_TP(300.0, Cantera::OneAtm);
    products_ = products;
    products_pristine_.resize(products_->stateSize());
    products_->saveState(products_pristine_);
}

CombustorResult V6VariantEngine::combustor_run(double T_in, double p_in, const std::string& fuel,
                                               double phi, double efficiency, double heat_loss_fraction)
{
    if (mode_ == Equilibrium::Full) {
        base_.config = config;
        return base_.combustor_run(T_in, p_in, fuel, phi, efficiency, heat_loss_fraction);
    }
    // scripts/phase8/v6_optimized.py ProductsOnlyCombustor.run
    if (!(0.0 <= heat_loss_fraction && heat_loss_fraction < 1.0)) {
        throw std::invalid_argument("heat_loss_fraction must be in [0, 1)");
    }
    Cantera::ThermoPhase& in = *eq_;
    Cantera::ThermoPhase& out = *out_;
    Cantera::ThermoPhase& prod = *products_;
    in.restoreState(eq_pristine_);
    out.restoreState(out_pristine_);
    in.setState_TP(T_in, p_in);
    std::vector<double> f = composition_to_array(in, fuel);
    std::vector<double> o = composition_to_array(in, "O2:1.0, N2:3.76");
    in.setEquivalenceRatio(phi, f.data(), o.data(), Cantera::ThermoBasis::molar);
    const double h_in = in.enthalpy_mass();
    auto z = [&](const char* element) {
        const size_t m = in.elementIndex(element, false);
        if (m == Cantera::npos) throw std::invalid_argument(std::string("missing element ") + element);
        return in.elementalMoleFraction(m);
    };
    const double co2 = z("C"), h2o = z("H") / 2.0;
    const double o2 = (z("O") - 2.0 * co2 - h2o) / 2.0;
    if (o2 < -1e-12) throw std::invalid_argument("products-only initial mixture has negative O2");
    Cantera::Composition initial{{"CO2", co2}, {"H2O", h2o}, {"O2", std::max(o2, 0.0)},
                                 {"N2", z("N") / 2.0}, {"AR", z("Ar")}};
    prod.restoreState(products_pristine_);
    prod.setMoleFractionsByName(initial);
    prod.setState_TP(T_in, p_in);
    prod.setState_HP(h_in, p_in);
    prod.equilibrate("HP", "auto", 1e-9, 1000, 100, 0, 0);
    const double T_ideal = prod.temperature();
    const double T_out = T_in + efficiency * (1.0 - heat_loss_fraction) * (T_ideal - T_in);
    std::vector<double> y_products(prod.nSpecies());
    prod.getMassFractions(y_products.data());
    std::vector<double> full_y(out.nSpecies(), 0.0);
    for (size_t k = 0; k < full_index_.size(); ++k) full_y[full_index_[k]] = y_products[k];
    out.setMassFractions(full_y.data());
    out.setState_TP(T_out, p_in);
    const double cp_out = out.cp_mass();
    const double R_out = Cantera::GasConstant / out.meanMolecularWeight();
    std::vector<double> Y_out(out.nSpecies());
    out.getMassFractions(Y_out.data());
    return {out.temperature(), out.pressure(), out.enthalpy_mass(), cp_out, R_out,
            cp_out / (cp_out - R_out), Y_out};
}

// Copy of V6Engine::run_full_cycle; only the combustor call and the mixing
// phase (an equivalent never-reset Solution) differ in identity.
CycleResult V6VariantEngine::run_full_cycle(const std::string& fuel,
                                            const std::vector<std::string>& fuel_species,
                                            double phi, double combustor_efficiency)
{
    ++cycle_calls;
    base_.config = config;
    CycleResult r;
    const V6Config& c = config;
    double T_ambient = c.T_ambient, P_ambient = c.P_ambient, m_dot_core = c.mass_flow_core;
    double bpr = c.bypass_ratio;
    double m_dot_bypass = 0.0, fan_work_total = 0.0;
    if (bpr > 0) {
        m_dot_bypass = bpr * m_dot_core;
        r.fan = base_.run_fan(T_ambient, P_ambient, m_dot_bypass);
        r.has_fan = true;
        fan_work_total = r.fan.work_total;
    }
    r.compressor = base_.run_compressor(T_ambient, P_ambient);
    double p_loss = c.combustor_pressure_loss;
    if (!(0.0 <= p_loss && p_loss < 1.0)) throw std::invalid_argument("combustor_pressure_loss must be in [0, 1)");
    double p_comb_in = r.compressor.p_out * (1.0 - p_loss);
    double beta = c.combustor_air_fraction;
    if (!(0.0 < beta && beta <= 1.0)) throw std::invalid_argument("combustor_air_fraction must be in (0, 1]");
    double m_dot_burn = beta * m_dot_core;

    double f = base_.fuel_air_ratio(fuel, fuel_species, phi);
    r.combustor = combustor_run(r.compressor.T_out, p_comb_in, fuel, phi, combustor_efficiency,
                                c.combustor_heat_loss_fraction);
    double m_dot_fuel = f * m_dot_burn;
    double m_dot_total = m_dot_core + m_dot_fuel;
    double comp_work_total = r.compressor.work_specific * m_dot_core;

    if (beta < 1.0) {
        double m_prod = m_dot_burn + m_dot_fuel;
        double m_byp_air = (1.0 - beta) * m_dot_core;
        double cp_p = r.combustor.cp_out, R_p = r.combustor.R_out;
        mix_->setMoleFractionsByName("O2:0.21, N2:0.79");
        mix_->setState_TP(r.compressor.T_out, p_comb_in);
        double cp_air = mix_->cp_mass();
        double R_air = Cantera::GasConstant / mix_->meanMolecularWeight();
        double T4_mix = ((m_prod * cp_p * r.combustor.T_out + m_byp_air * cp_air * r.compressor.T_out) /
                         (m_prod * cp_p + m_byp_air * cp_air));
        double cp_mix = (m_prod * cp_p + m_byp_air * cp_air) / m_dot_total;
        double R_mix = (m_prod * R_p + m_byp_air * R_air) / m_dot_total;
        r.combustor.T_out = T4_mix;
        r.combustor.cp_out = cp_mix;
        r.combustor.R_out = R_mix;
        r.combustor.gamma_out = cp_mix / (cp_mix - R_mix);
    }

    r.turbine = base_.run_turbine_analytic(r.combustor, m_dot_total, comp_work_total + fan_work_total);
    r.nozzle = base_.run_nozzle(r.turbine, m_dot_total);

    double thrust_core = r.nozzle.thrust_total;
    double thrust_bypass = r.has_fan ? r.fan.thrust_bypass : 0.0;
    double thrust = thrust_core + thrust_bypass;
    double m_dot_air_total = m_dot_core + m_dot_bypass;
    const double inf = std::numeric_limits<double>::infinity();
    if (thrust <= 0) {
        r.tsfc_SI = inf;
        r.tsfc_mg_per_Ns = inf;
    } else {
        r.tsfc_SI = m_dot_fuel / thrust;
        r.tsfc_mg_per_Ns = r.tsfc_SI * 1.0e6;
    }
    double LHV = 43e6;
    double fuel_power = m_dot_fuel * LHV;
    if (thrust <= 0 || fuel_power <= 0) {
        r.thermal_efficiency = 0.0;
    } else {
        double ke_flux = 0.5 * m_dot_total * std::pow(r.nozzle.u, 2.0);
        if (r.has_fan) ke_flux += 0.5 * m_dot_bypass * std::pow(r.fan.u_bypass_exit, 2.0);
        r.thermal_efficiency = ke_flux / fuel_power;
    }
    double OPR = r.compressor.p_out / c.P_ambient;
    r.nox_g_s = base_.estimate_nox(OPR, m_dot_fuel);

    r.fuel_air_ratio = f;
    r.fuel_mass_flow = m_dot_fuel;
    r.total_mass_flow = m_dot_total;
    r.bypass_mass_flow = m_dot_bypass;
    r.total_air_mass_flow = m_dot_air_total;
    r.thrust_N = thrust;
    r.thrust_kN = thrust / 1e3;
    r.thrust_core_kN = thrust_core / 1e3;
    r.thrust_bypass_kN = thrust_bypass / 1e3;
    r.specific_thrust_Ns_kg = m_dot_air_total > 0 ? thrust / m_dot_air_total : inf;
    r.fan_work_W = fan_work_total;
    return r;
}

// Copy of V6Engine::run_at_thrust (same bracket, guard and Brent sequence).
AtThrustResult V6VariantEngine::run_at_thrust(double target_kN, const std::string& fuel,
                                              const std::vector<std::string>& fuel_species,
                                              double combustor_efficiency, double phi_lo, double phi_hi,
                                              double t4_max_K, double phi_xtol, double phi_guess)
{
    double lo = phi_lo, hi = phi_hi;
    const std::string bounds_repr = "(" + py_repr(phi_lo) + ", " + py_repr(phi_hi) + ")";
    if (!(std::isfinite(lo) && std::isfinite(hi) && 0.0 < lo && lo < hi)) {
        throw std::invalid_argument("phi_bounds must satisfy 0 < lo < hi");
    }
    double target = target_kN;
    if (!(std::isfinite(target) && target > 0.0)) throw std::invalid_argument("target_kN must be finite and positive");
    if (!(std::isfinite(t4_max_K) && t4_max_K > 0.0)) throw std::invalid_argument("t4_max_K must be finite and positive");
    bool have_guess = phi_guess >= 0.0;
    if (have_guess && !std::isfinite(phi_guess)) throw std::invalid_argument("phi_guess must be finite or None");
    if (!(std::isfinite(combustor_efficiency) && 0.0 < combustor_efficiency && combustor_efficiency <= 1.0)) {
        throw std::invalid_argument("combustor_efficiency must be in (0, 1]");
    }

    std::map<double, std::variant<CycleResult, std::string>> cache;
    auto cycle = [&](double phi) -> const CycleResult& {
        auto it = cache.find(phi);
        if (it == cache.end()) {
            try {
                it = cache.emplace(phi, run_full_cycle(fuel, fuel_species, phi, combustor_efficiency)).first;
            } catch (const CycleDoesNotClose& exc) {
                it = cache.emplace(phi, std::string(exc.what())).first;
            }
        }
        if (std::holds_alternative<std::string>(it->second)) {
            throw CycleDoesNotClose(std::get<std::string>(it->second));
        }
        return std::get<CycleResult>(it->second);
    };
    auto runs = [&](double phi) {
        try {
            cycle(phi);
            return true;
        } catch (const CycleDoesNotClose&) {
            return false;
        }
    };
    auto t4 = [&](double phi) { return cycle(phi).combustor.T_out; };
    auto residual = [&](double phi) { return cycle(phi).thrust_kN - target; };
    const double rtol = 4 * std::numeric_limits<double>::epsilon();

    std::map<std::string, double> info{{"phi_bounds_lo", phi_lo}, {"phi_bounds_hi", phi_hi},
                                       {"t4_max_K", t4_max_K}};
    auto guard_phi = [&](double a) {
        return brentq([&](double p) { return t4(p) - t4_max_K; }, a, hi, phi_xtol);
    };
    auto fail_high = [&](double a) {
        std::string where;
        if (t4(hi) > t4_max_K) {
            double g = guard_phi(a);
            info["phi_upper"] = g;
            info["t4_guard_active"] = 1.0;
            info["thrust_at_upper_kN"] = residual(g) + target;
            where = "the T4 guard (" + fmt("%.1f", t4_max_K) + " K, phi=" + fmt("%.4f", g) + ")";
        } else {
            info["phi_upper"] = hi;
            info["t4_guard_active"] = 0.0;
            info["thrust_at_upper_kN"] = residual(hi) + target;
            where = "phi=" + py_repr(hi);
        }
        throw ThrustTargetUnreachable(
            "above the maximum thrust " + fmt("%.3f", info["thrust_at_upper_kN"]) + " kN at " + where,
            target_kN, info);
    };

    bool bracketed = false;
    double a = 0.0, b = 0.0, phi = 0.0;
    if (have_guess && lo < phi_guess && phi_guess < hi) {
        double p0 = phi_guess, step = 1.02;
        if (runs(p0)) {
            double r0 = residual(p0);
            double p1 = p0;
            for (int i = 0; i < 12; i++) {
                p1 = (r0 < 0) ? std::min(p1 * step, hi) : std::max(p1 / step, lo);
                if (!runs(p1)) break;
                if ((residual(p1) > 0) != (r0 > 0)) {
                    if (p0 < p1) {
                        a = p0;
                        b = p1;
                    } else {
                        a = p1;
                        b = p0;
                    }
                    bracketed = true;
                    break;
                }
                if (p1 == lo || p1 == hi) break;
            }
        }
    }
    if (bracketed) {
        phi = brentq(residual, a, b, phi_xtol, rtol);
        if (t4(phi) > t4_max_K) bracketed = false;
    }
    if (!bracketed) {
        if (!runs(lo)) {
            if (!runs(hi)) {
                throw ThrustTargetUnreachable("cycle does not close anywhere in phi " + bounds_repr, target_kN, info);
            }
            double c0 = lo, c1 = hi;
            while (c1 - c0 > 1e-4 * c1) {
                double m = 0.5 * (c0 + c1);
                if (runs(m)) c1 = m; else c0 = m;
            }
            if (residual(c1) > 0.0) {
                while (c1 - c0 > 1e-9 * c1) {
                    double m = 0.5 * (c0 + c1);
                    if (runs(m)) c1 = m; else c0 = m;
                }
            }
            lo = c1;
            info["phi_lower_cycle_closure"] = lo;
        }
        info["thrust_at_lower_kN"] = residual(lo) + target;
        if (t4(lo) > t4_max_K) {
            throw ThrustTargetUnreachable("T4 " + fmt("%.1f", t4(lo)) + " K at phi=" + py_repr(lo) +
                                              " already exceeds the guard", target_kN, info);
        }
        if (residual(lo) > 0.0) {
            throw ThrustTargetUnreachable("below the minimum thrust " + fmt("%.3f", info["thrust_at_lower_kN"]) +
                                              " kN at phi=" + py_repr(lo), target_kN, info);
        }
        if (residual(hi) < 0.0) fail_high(lo);
        phi = brentq(residual, lo, hi, phi_xtol, rtol);
        if (t4(phi) > t4_max_K) fail_high(lo);
    }
    AtThrustResult out;
    out.cycle = cycle(phi);
    out.match.phi = phi;
    out.match.target_kN = target;
    out.match.residual_kN = residual(phi);
    out.match.t4_K = t4(phi);
    out.match.n_cycle_evaluations = static_cast<int>(cache.size());
    out.match.info = info;
    return out;
}

}  // namespace catjet
