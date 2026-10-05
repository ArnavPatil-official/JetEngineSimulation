// P8.1 port of the v6 cycle; see v6_engine.hpp. Line references are to the
// Python source at the Phase 7 freeze (protected in Phase 8).
#include "v6_engine.hpp"

#include "brentq.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <limits>
#include <variant>

namespace catjet {

namespace {

// Python repr(float): shortest round-trip digits, scientific iff exp < -4 or >= 16
std::string py_repr(double x)
{
    if (std::isnan(x)) return "nan";
    if (std::isinf(x)) return x > 0 ? "inf" : "-inf";
    char buf[64];
    auto res = std::to_chars(buf, buf + sizeof(buf), x, std::chars_format::scientific);
    std::string sci(buf, res.ptr);
    // sci = [-]d[.ddd]e[+-]XX
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

}  // namespace

ThrustTargetUnreachable::ThrustTargetUnreachable(const std::string& why, double target,
                                                 std::map<std::string, double> inf)
    : std::runtime_error("thrust target " + fmt("%.3f", target) + " kN unreachable: " + why),
      reason(why), target_kN(target), info(std::move(inf))
{
}

V6Engine::V6Engine(const std::string& mechanism)
{
    // IntegratedTurbofanEngine.__init__: self.gas (compressor/mix gas, never reset)
    gas_sol_ = Cantera::newSolution(mechanism);
    gas_ = gas_sol_->thermo();
    // Combustor._fresh_solutions: two Solutions reset to their as-constructed state per call
    eq_sol_ = Cantera::newSolution(mechanism);
    out_sol_ = Cantera::newSolution(mechanism);
    eq_ = eq_sol_->thermo();
    out_ = out_sol_->thermo();
    eq_pristine_.resize(eq_->stateSize());
    out_pristine_.resize(out_->stateSize());
    eq_->saveState(eq_pristine_);
    out_->saveState(out_pristine_);
    // _calculate_fuel_air_ratio cache: one Solution reset per call
    far_sol_ = Cantera::newSolution(mechanism);
    far_ = far_sol_->thermo();
    far_pristine_.resize(far_->stateSize());
    far_->saveState(far_pristine_);
}

std::vector<double> V6Engine::composition_to_array(Cantera::ThermoPhase& th, const std::string& comp)
{
    // thermo.pyx __composition_to_array (basis 'mole'): TPX = None, None, comp; X; restore
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

// run_compressor + Compressor.compute_outlet_state
CompressorResult V6Engine::run_compressor(double T_in, double p_in)
{
    Cantera::ThermoPhase& g = *gas_;
    // self.gas.TPX = T_in, p_in, "O2:0.21, N2:0.79"
    {
        double T = T_in, P = p_in;
        g.setMoleFractionsByName("O2:0.21, N2:0.79");
        g.setState_TP(T, P);
    }
    g.setState_TP(T_in, p_in);
    double s_in = g.entropy_mass();
    double p_out = p_in * config.pi_c;
    g.setState_SP(s_in, p_out);
    double T_out_ideal = g.temperature();
    double T_out = T_in + (T_out_ideal - T_in) / config.eta_c;
    g.setState_TP(T_in, p_in);
    double h_in = g.enthalpy_mass();
    g.setState_TP(T_out, p_out);
    double h_out = g.enthalpy_mass();
    return {T_out, p_out, h_in, h_out, h_out - h_in};
}

// simulation/fan.py Fan(fpr, eta_fan, cp=1005.0, gamma=1.4).run
FanResult V6Engine::run_fan(double T0, double p0, double m_dot_bypass) const
{
    const double fpr = config.fpr, eta_fan = config.eta_fan, cp = 1005.0, gamma = 1.4;
    if (fpr < 1.0) throw std::invalid_argument("Fan pressure ratio must be >= 1");
    if (!(0.0 < eta_fan && eta_fan <= 1.0)) throw std::invalid_argument("Fan efficiency must be in (0, 1]");
    if (m_dot_bypass < 0) throw std::invalid_argument("Bypass mass flow must be non-negative");
    double exponent = (gamma - 1.0) / gamma;
    double dT_ideal = T0 * (std::pow(fpr, exponent) - 1.0);
    double dT = dT_ideal / eta_fan;
    double T_exit = T0 + dT;
    double p_exit = p0 * fpr;
    double work_total = m_dot_bypass * cp * dT;
    double expansion_factor = std::max(0.0, 1.0 - std::pow(p0 / p_exit, exponent));
    double u_exit = std::sqrt(2.0 * cp * T_exit * expansion_factor);
    double thrust_bypass = m_dot_bypass * u_exit;
    return {T_exit, p_exit, dT, work_total, u_exit, thrust_bypass};
}

// IntegratedTurbofanEngine._calculate_fuel_air_ratio
double V6Engine::fuel_air_ratio(const std::string& fuel, const std::vector<std::string>& fuel_species,
                                double phi)
{
    Cantera::ThermoPhase& g = *far_;
    g.restoreState(far_pristine_);
    g.setState_TP(300.0, 101325.0);
    std::vector<double> f = composition_to_array(g, fuel);
    std::vector<double> o = composition_to_array(g, "O2:1.0, N2:3.76");
    g.setEquivalenceRatio(phi, f.data(), o.data(), Cantera::ThermoBasis::molar);
    std::vector<double> Y(g.nSpecies());
    g.getMassFractions(Y.data());
    double Y_fuel = 0.0;  // Python sum() starts from int 0
    for (const auto& sp : fuel_species) {
        size_t k = g.speciesIndex(sp);
        if (k != Cantera::npos) Y_fuel += Y[k];
    }
    double Y_air = 0.0;
    for (const char* sp : {"O2", "N2"}) {
        size_t k = g.speciesIndex(sp);
        if (k != Cantera::npos) Y_air += Y[k];
    }
    if (Y_air < 1e-10) throw std::invalid_argument("Air mass fraction is zero - check mixture definition");
    return Y_fuel / Y_air;
}

// Combustor.run
CombustorResult V6Engine::combustor_run(double T_in, double p_in, const std::string& fuel, double phi,
                                        double efficiency, double heat_loss_fraction)
{
    Cantera::ThermoPhase& eq = *eq_;
    Cantera::ThermoPhase& out = *out_;
    eq.restoreState(eq_pristine_);
    out.restoreState(out_pristine_);
    eq.setState_TP(T_in, p_in);
    std::vector<double> f = composition_to_array(eq, fuel);
    std::vector<double> o = composition_to_array(eq, "O2:1.0, N2:3.76");
    eq.setEquivalenceRatio(phi, f.data(), o.data(), Cantera::ThermoBasis::molar);
    // Python defaults: solver 'auto', rtol 1e-9, max_steps 1000, max_iter 100
    eq.equilibrate("HP", "auto", 1e-9, 1000, 100, 0, 0);
    double T_ideal = eq.temperature();
    std::vector<double> Y_ideal(eq.nSpecies());
    eq.getMassFractions(Y_ideal.data());
    if (!(0.0 <= heat_loss_fraction && heat_loss_fraction < 1.0)) {
        throw std::invalid_argument("heat_loss_fraction must be in [0, 1)");
    }
    double T_out = T_in + efficiency * (1.0 - heat_loss_fraction) * (T_ideal - T_in);
    // gas_out.TPY = T_out, p_in, Y_ideal
    out.setMassFractions(Y_ideal.data());
    out.setState_TP(T_out, p_in);
    double cp_out = out.cp_mass();
    double R_out = Cantera::GasConstant / out.meanMolecularWeight();
    double cv_out = cp_out - R_out;
    double gamma_out = cp_out / cv_out;
    std::vector<double> Y_out(out.nSpecies());
    out.getMassFractions(Y_out.data());
    return {out.temperature(), out.pressure(), out.enthalpy_mass(), cp_out, R_out, gamma_out, Y_out};
}

// run_turbine_analytic (flow state from _cantera_to_flow_state: same T, p, cp, R, gamma)
TurbineResult V6Engine::run_turbine_analytic(const CombustorResult& in, double m_dot,
                                             double target_work_total) const
{
    double cp = in.cp_out, R = in.R_out, gamma = in.gamma_out;
    double T_in = in.T_out, p_in = in.p_out;
    double eta_t = config.eta_polytropic;
    double T_out = T_in - target_work_total / (m_dot * cp);
    if (T_out <= 0) {
        throw CycleDoesNotClose("Analytic turbine: target work " + fmt("%.1f", target_work_total / 1e6) +
                                " MW exceeds available enthalpy flux");
    }
    double p_out = p_in * std::pow(T_out / T_in, gamma / (eta_t * (gamma - 1.0)));
    double A_outlet = config.A_combustor_exit * 1.82;
    double rho_out = p_out / (R * T_out);
    double u_out = m_dot / (rho_out * A_outlet);
    return {rho_out, u_out, p_out, T_out, target_work_total / m_dot, target_work_total, cp, R, gamma};
}

// run_nozzle (analytic, always fully expanded)
NozzleResult V6Engine::run_nozzle(const TurbineResult& in, double m_dot) const
{
    if (m_dot <= 0) throw std::invalid_argument("Mass flow rate must be positive for nozzle computation");
    double T_in = in.T, p_in = in.p, cp = in.cp, R = in.R, gamma = in.gamma;
    double P_amb = config.P_ambient;
    double A_exit = config.A_nozzle_exit;
    double pressure_ratio = (p_in < P_amb) ? 1.0 : P_amb / p_in;
    double exponent = (gamma - 1) / gamma;
    double expansion_factor = std::max(0.0, 1.0 - std::pow(pressure_ratio, exponent));
    double u = std::sqrt(2 * cp * T_in * expansion_factor);
    if (u <= 0) throw CycleDoesNotClose("Computed non-positive nozzle exit velocity");
    double T_exit = T_in * std::pow(pressure_ratio, exponent);
    double rho_exit = P_amb / (R * T_exit);
    double F_momentum = m_dot * u;
    double p_exit = p_in * pressure_ratio;
    double delta_p = p_exit - P_amb;
    double F_pressure = (std::fabs(delta_p) < 1.0) ? 0.0 : delta_p * A_exit;
    double F_total = F_momentum + F_pressure;
    return {rho_exit, u, P_amb, T_exit, F_total, F_momentum, F_pressure, m_dot / (rho_exit * u)};
}

// EmissionsEstimator.estimate_nox
double V6Engine::estimate_nox(double OPR, double m_dot_fuel) const
{
    if (OPR <= 1.0 || m_dot_fuel <= 0) return 0.0;
    double nox_ei = config.nox_A * std::pow(OPR, config.nox_B) * std::pow(m_dot_fuel, config.nox_C);
    return nox_ei * m_dot_fuel;
}

// run_full_cycle(turbine_model='analytic', nozzle_model='analytic'), eta_b given
CycleResult V6Engine::run_full_cycle(const std::string& fuel, const std::vector<std::string>& fuel_species,
                                     double phi, double combustor_efficiency)
{
    CycleResult r;
    const V6Config& c = config;
    double T_ambient = c.T_ambient, P_ambient = c.P_ambient, m_dot_core = c.mass_flow_core;
    double bpr = c.bypass_ratio;
    double m_dot_bypass = 0.0, fan_work_total = 0.0;
    if (bpr > 0) {
        m_dot_bypass = bpr * m_dot_core;
        r.fan = run_fan(T_ambient, P_ambient, m_dot_bypass);
        r.has_fan = true;
        fan_work_total = r.fan.work_total;
    }
    r.compressor = run_compressor(T_ambient, P_ambient);
    double p_loss = c.combustor_pressure_loss;
    if (!(0.0 <= p_loss && p_loss < 1.0)) throw std::invalid_argument("combustor_pressure_loss must be in [0, 1)");
    double p_comb_in = r.compressor.p_out * (1.0 - p_loss);
    double beta = c.combustor_air_fraction;
    if (!(0.0 < beta && beta <= 1.0)) throw std::invalid_argument("combustor_air_fraction must be in (0, 1]");
    double m_dot_burn = beta * m_dot_core;

    double f = fuel_air_ratio(fuel, fuel_species, phi);
    r.combustor = combustor_run(r.compressor.T_out, p_comb_in, fuel, phi, combustor_efficiency,
                                c.combustor_heat_loss_fraction);
    double m_dot_fuel = f * m_dot_burn;
    double m_dot_total = m_dot_core + m_dot_fuel;
    double comp_work_total = r.compressor.work_specific * m_dot_core;

    if (beta < 1.0) {
        double m_prod = m_dot_burn + m_dot_fuel;
        double m_byp_air = (1.0 - beta) * m_dot_core;
        double cp_p = r.combustor.cp_out, R_p = r.combustor.R_out;
        gas_->setMoleFractionsByName("O2:0.21, N2:0.79");
        gas_->setState_TP(r.compressor.T_out, p_comb_in);
        double cp_air = gas_->cp_mass();
        double R_air = Cantera::GasConstant / gas_->meanMolecularWeight();
        double T4_mix = ((m_prod * cp_p * r.combustor.T_out + m_byp_air * cp_air * r.compressor.T_out) /
                         (m_prod * cp_p + m_byp_air * cp_air));
        double cp_mix = (m_prod * cp_p + m_byp_air * cp_air) / m_dot_total;
        double R_mix = (m_prod * R_p + m_byp_air * R_air) / m_dot_total;
        r.combustor.T_out = T4_mix;
        r.combustor.cp_out = cp_mix;
        r.combustor.R_out = R_mix;
        r.combustor.gamma_out = cp_mix / (cp_mix - R_mix);
    }

    r.turbine = run_turbine_analytic(r.combustor, m_dot_total, comp_work_total + fan_work_total);
    r.nozzle = run_nozzle(r.turbine, m_dot_total);

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
    r.nox_g_s = estimate_nox(OPR, m_dot_fuel);

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

// run_at_thrust
AtThrustResult V6Engine::run_at_thrust(double target_kN, const std::string& fuel,
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
