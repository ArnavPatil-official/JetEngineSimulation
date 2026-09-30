#include "enthalpy_turbine.hpp"

#include "brentq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <variant>

namespace catjet {

namespace {

double relative_error(double lhs, double rhs, double denominator_floor = 1.0)
{
    return std::abs(lhs - rhs) / std::max({std::abs(lhs), std::abs(rhs), denominator_floor});
}

void check_positive(double value, const char* name)
{
    if (!(std::isfinite(value) && value > 0.0)) {
        throw std::invalid_argument(std::string(name) + " must be finite and positive");
    }
}

void check_fraction(double value, const char* name)
{
    if (!(std::isfinite(value) && value >= 0.0 && value <= 1.0)) {
        throw std::invalid_argument(std::string(name) + " must be in [0,1]");
    }
}

}  // namespace

GasThermo::GasThermo(const std::string& mechanism)
{
    solution_ = Cantera::newSolution(mechanism);
    gas_ = solution_->thermo();
}

size_t GasThermo::n_species() const
{
    return gas_->nSpecies();
}

void GasThermo::set(const GasState& state)
{
    check_positive(state.T, "GasState.T");
    check_positive(state.P, "GasState.P");
    if (state.Y.size() != gas_->nSpecies()) {
        throw std::invalid_argument("GasState.Y must have one entry per mechanism species");
    }
    double total = 0.0;
    for (double y : state.Y) {
        if (!std::isfinite(y) || y < -1e-15) {
            throw std::invalid_argument("GasState.Y has a negative or non-finite entry");
        }
        total += y;
    }
    if (std::abs(total - 1.0) > 1e-10) {
        throw std::invalid_argument("GasState.Y must sum to one within 1e-10");
    }
    gas_->setMassFractions(state.Y.data());
    gas_->setState_TP(state.T, state.P);
}

GasState GasThermo::from_moles(double T, double P, const std::string& composition)
{
    check_positive(T, "temperature");
    check_positive(P, "pressure");
    gas_->setMoleFractionsByName(composition);
    gas_->setState_TP(T, P);
    GasState out{T, P, std::vector<double>(gas_->nSpecies())};
    gas_->getMassFractions(out.Y.data());
    return out;
}

GasState GasThermo::at_enthalpy(double h, double P, const std::vector<double>& Y)
{
    GasState state{300.0, P, Y};
    set(state);
    // Cantera's default relative tolerance is 1e-9, too loose for the
    // registered 1e-10 enthalpy-flux closure across unequal streams.
    gas_->setState_HP(h, P, 1e-13);
    state.T = gas_->temperature();
    gas_->getMassFractions(state.Y.data());
    return state;
}

GasState GasThermo::compress(const GasState& in, double pressure_ratio, double eta)
{
    check_positive(pressure_ratio, "compressor pressure ratio");
    if (pressure_ratio < 1.0 || !(eta > 0.0 && eta <= 1.0)) {
        throw std::invalid_argument("compressor ratio must be >=1 and efficiency in (0,1]");
    }
    set(in);
    const double h_in = gas_->enthalpy_mass();
    const double s_in = gas_->entropy_mass();
    const double P_out = in.P * pressure_ratio;
    gas_->setState_SP(s_in, P_out);
    const double h_ideal = gas_->enthalpy_mass();
    return at_enthalpy(h_in + (h_ideal - h_in) / eta, P_out, in.Y);
}

std::vector<double> GasThermo::element_mass_fractions()
{
    std::vector<double> result(gas_->nElements());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = gas_->elementalMassFraction(i);
    }
    return result;
}

GasProperties GasThermo::properties(const GasState& state)
{
    set(state);
    GasProperties p;
    p.h = gas_->enthalpy_mass();
    p.s = gas_->entropy_mass();
    p.cp = gas_->cp_mass();
    p.R = Cantera::GasConstant / gas_->meanMolecularWeight();
    p.gamma = p.cp / (p.cp - p.R);
    p.rho = gas_->density();
    p.elements = element_mass_fractions();
    return p;
}

MixResult GasThermo::mix(const MassStream& a, const MassStream& b, double P)
{
    check_positive(P, "mix pressure");
    if (!(std::isfinite(a.mass_flow) && a.mass_flow >= 0.0 &&
          std::isfinite(b.mass_flow) && b.mass_flow >= 0.0)) {
        throw std::invalid_argument("mix mass flows must be finite and nonnegative");
    }
    const double m = a.mass_flow + b.mass_flow;
    check_positive(m, "total mix mass flow");
    if (a.state.Y.size() != gas_->nSpecies() || b.state.Y.size() != gas_->nSpecies()) {
        throw std::invalid_argument("mix streams must use the same mechanism species");
    }
    const GasProperties pa = properties(a.state);
    const GasProperties pb = properties(b.state);
    std::vector<double> Y(gas_->nSpecies());
    for (size_t k = 0; k < Y.size(); ++k) {
        Y[k] = (a.mass_flow * a.state.Y[k] + b.mass_flow * b.state.Y[k]) / m;
    }
    const double H_in = a.mass_flow * pa.h + b.mass_flow * pb.h;
    GasState out = at_enthalpy(H_in / m, P, Y);
    const GasProperties po = properties(out);
    MixResult r;
    r.stream = {std::move(out), m};
    r.mass_relative = relative_error(r.stream.mass_flow, m, 1e-12);
    r.energy_relative = std::abs(m * po.h - H_in) /
                        std::max(std::abs(a.mass_flow * pa.h) + std::abs(b.mass_flow * pb.h), 1.0);
    for (size_t e = 0; e < po.elements.size(); ++e) {
        const double element_in = a.mass_flow * pa.elements[e] + b.mass_flow * pb.elements[e];
        const double element_out = m * po.elements[e];
        r.element_relative = std::max(r.element_relative,
                                      std::abs(element_out - element_in) /
                                      std::max(std::abs(element_in), 1e-12));
    }
    return r;
}

ExpansionResult GasThermo::expand_for_work(const MassStream& in, double work_W,
                                           double eta_poly, int steps,
                                           double constant_cp, double constant_R)
{
    check_positive(in.mass_flow, "turbine mass flow");
    if (!(std::isfinite(work_W) && work_W >= 0.0)) {
        throw std::invalid_argument("turbine work must be finite and nonnegative");
    }
    if (!(std::isfinite(eta_poly) && eta_poly > 0.0 && eta_poly <= 1.0)) {
        throw std::invalid_argument("turbine polytropic efficiency must be in (0,1]");
    }
    if (steps < 1) throw std::invalid_argument("turbine pressure steps must be >=1");
    const GasProperties pin = properties(in.state);
    ExpansionResult r;
    r.stream.mass_flow = in.mass_flow;
    r.requested_work = work_W;
    r.steps = steps;
    if (work_W == 0.0) {
        r.stream.state = in.state;
        return r;
    }

    if (constant_cp > 0.0 || constant_R > 0.0) {
        check_positive(constant_cp, "constant cp");
        check_positive(constant_R, "constant R");
        const double T_out = in.state.T - work_W / (in.mass_flow * constant_cp);
        if (T_out <= 0.0) throw CycleDoesNotClose("constant-cp turbine work exceeds available enthalpy");
        const double P_out = in.state.P *
            std::pow(T_out / in.state.T, constant_cp / (eta_poly * constant_R));
        r.stream.state = {T_out, P_out, in.state.Y};
        r.actual_work = in.mass_flow * constant_cp * (in.state.T - T_out);
        r.energy_relative = std::abs(r.actual_work - work_W) /
                            std::max(std::abs(in.mass_flow * constant_cp * in.state.T) + work_W, 1.0);
        return r;
    }

    const double h_target = pin.h - work_W / in.mass_flow;
    // P8.2-A1: the outlet pressure is found from a genuinely integrated
    // polytropic path, so 50-to-100 step convergence measures discretisation.
    auto integrate = [&](double log_ratio) {
        const double dx = log_ratio / steps;
        double T = in.state.T;
        set(in.state);  // Y is frozen throughout this pressure path
        auto slope = [&](double temperature, double x) {
            const double P = in.state.P * std::exp(x);
            if (!(std::isfinite(temperature) && temperature > 100.0 &&
                  std::isfinite(P) && P > 0.0)) {
                throw CycleDoesNotClose("polytropic path left the positive thermodynamic domain");
            }
            gas_->setState_TP(temperature, P);
            const double cp = gas_->cp_mass();
            if (!(std::isfinite(cp) && cp > 0.0)) {
                throw CycleDoesNotClose("polytropic path has invalid cp");
            }
            return eta_poly * pin.R * temperature / cp;
        };
        for (int i = 0; i < steps; ++i) {
            const double x = i * dx;
            const double k1 = slope(T, x);
            const double k2 = slope(T + 0.5 * dx * k1, x + 0.5 * dx);
            const double k3 = slope(T + 0.5 * dx * k2, x + 0.5 * dx);
            const double k4 = slope(T + dx * k3, x + dx);
            T += dx * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        }
        if (!(std::isfinite(T) && T > 100.0)) {
            throw CycleDoesNotClose("polytropic outlet left the positive thermodynamic domain");
        }
        return GasState{T, in.state.P * std::exp(log_ratio), in.state.Y};
    };
    auto residual = [&](double log_ratio) {
        GasState trial = integrate(log_ratio);
        return properties(trial).h - h_target;
    };
    double left = -1.0;
    try {
        while (left >= -20.0 && residual(left) > 0.0) {
            left *= 2.0;
        }
    } catch (const Cantera::CanteraError&) {
        throw CycleDoesNotClose("polytropic pressure bracket left Cantera temperature range");
    }
    if (left < -20.0) {
        throw CycleDoesNotClose("polytropic work target has no physical pressure bracket");
    }
    double log_pressure_ratio;
    try {
        log_pressure_ratio = brentq(residual, left, 0.0, 1e-12);
    } catch (const Cantera::CanteraError&) {
        throw CycleDoesNotClose("polytropic root left Cantera temperature range");
    }
    r.stream.state = integrate(log_pressure_ratio);
    const GasProperties pout = properties(r.stream.state);
    r.actual_work = in.mass_flow * (pin.h - pout.h);
    r.energy_relative = std::abs(r.actual_work - work_W) /
                        std::max(std::abs(in.mass_flow * pin.h) + work_W, 1.0);
    for (size_t e = 0; e < pout.elements.size(); ++e) {
        const double element_in = in.mass_flow * pin.elements[e];
        const double element_out = in.mass_flow * pout.elements[e];
        r.element_relative = std::max(r.element_relative,
                                      std::abs(element_out - element_in) /
                                      std::max(std::abs(element_in), 1e-12));
    }
    return r;
}

P82Engine::P82Engine(const std::string& mechanism) : v6_(mechanism), thermo_(mechanism)
{
}

P82CycleResult P82Engine::run_full_cycle(const std::string& fuel,
                                          const std::vector<std::string>& fuel_species,
                                          double phi, double combustor_efficiency,
                                          int ablation_level)
{
    if (ablation_level != 1 && ablation_level != 2) {
        throw std::invalid_argument("P8.2 ablation level must be 1 or 2");
    }
    v6_.config = config.base;
    const V6Config& c = config.base;
    const double m_core = c.mass_flow_core;
    check_positive(m_core, "core mass flow");
    check_fraction(c.combustor_air_fraction, "combustor air fraction");
    check_fraction(c.combustor_pressure_loss, "combustor pressure loss");
    check_fraction(c.combustor_heat_loss_fraction, "combustor heat loss");
    const double f_ngv = ablation_level == 2 ? config.ngv_fraction : 0.0;
    const double f_rotor = ablation_level == 2 ? config.rotor_fraction : 0.0;
    check_fraction(f_ngv, "NGV cooling fraction");
    check_fraction(f_rotor, "rotor cooling fraction");
    if (!(c.combustor_air_fraction > f_ngv + f_rotor)) {
        throw CycleDoesNotClose("cooling fractions exhaust burner air");
    }
    const double m_burn_air = (c.combustor_air_fraction - f_ngv - f_rotor) * m_core;
    const double m_dilution = (1.0 - c.combustor_air_fraction) * m_core;
    const double m_ngv = f_ngv * m_core;
    const double m_rotor = f_rotor * m_core;

    P82CycleResult out;
    CycleResult& r = out.cycle;
    const auto save = [&](const std::string& name, const MassStream& stream) {
        out.stations[name] = stream;
    };
    const auto audit = [&](const MixResult& mix) {
        out.max_mass_relative = std::max(out.max_mass_relative, mix.mass_relative);
        out.max_energy_relative = std::max(out.max_energy_relative, mix.energy_relative);
        out.max_element_relative = std::max(out.max_element_relative, mix.element_relative);
    };

    GasState air = thermo_.from_moles(c.T_ambient, c.P_ambient, "O2:1, N2:3.76");
    save("ambient", {air, m_core});
    GasState compressor_exit = thermo_.compress(air, c.pi_c, c.eta_c);
    GasProperties p_air_in = thermo_.properties(air);
    GasProperties p_air_out = thermo_.properties(compressor_exit);
    r.compressor = {compressor_exit.T, compressor_exit.P, p_air_in.h,
                    p_air_out.h, p_air_out.h - p_air_in.h};
    save("compressor_exit", {compressor_exit, m_core});

    const double m_bypass = std::max(c.bypass_ratio, 0.0) * m_core;
    double fan_work = 0.0;
    if (m_bypass > 0.0) {
        r.fan = v6_.run_fan(c.T_ambient, c.P_ambient, m_bypass);
        r.has_fan = true;
        fan_work = r.fan.work_total;
        save("fan_inlet", {air, m_bypass});
        save("fan_exit", {{r.fan.T_exit, r.fan.p_exit, air.Y}, m_bypass});
    }

    const double P_burn = compressor_exit.P * (1.0 - c.combustor_pressure_loss);
    GasState burn_air{compressor_exit.T, P_burn, air.Y};
    const double f = v6_.fuel_air_ratio(fuel, fuel_species, phi);
    const double m_fuel = f * m_burn_air;
    const double m_prod = m_burn_air + m_fuel;
    save("burner_air_in", {burn_air, m_burn_air});
    GasState fuel_gas = thermo_.from_moles(burn_air.T, P_burn, fuel);
    save("fuel_gas_in", {fuel_gas, m_fuel});
    CombustorResult burner = v6_.combustor_run(burn_air.T, P_burn, fuel, phi,
                                                combustor_efficiency,
                                                c.combustor_heat_loss_fraction);
    GasState products_gas{burner.T_out, burner.p_out, burner.Y_out};
    const double H_gas = m_prod * thermo_.properties(products_gas).h;
    const double H_liquid = H_gas - m_fuel * config.vaporization_J_kg;
    GasState products = thermo_.at_enthalpy(H_liquid / m_prod, P_burn, products_gas.Y);
    save("burner_exit", {products, m_prod});
    const GasProperties p_burn_air = thermo_.properties(burn_air);
    const GasProperties p_fuel_gas = thermo_.properties(fuel_gas);
    const GasProperties p_products = thermo_.properties(products);
    const double H_reactant = m_burn_air * p_burn_air.h +
                              m_fuel * (p_fuel_gas.h - config.vaporization_J_kg);
    out.burner_heat_rejection_W = H_reactant -
                                   m_prod * p_products.h;
    out.max_energy_relative = std::max(out.max_energy_relative,
        std::abs(H_reactant - m_prod * p_products.h -
                 out.burner_heat_rejection_W) /
        std::max(std::abs(H_reactant) + std::abs(out.burner_heat_rejection_W), 1.0));
    for (size_t e = 0; e < p_products.elements.size(); ++e) {
        const double element_in = m_burn_air * p_burn_air.elements[e] +
                                  m_fuel * p_fuel_gas.elements[e];
        const double element_out = m_prod * p_products.elements[e];
        out.max_element_relative = std::max(out.max_element_relative,
            std::abs(element_out - element_in) /
            std::max(std::abs(element_in), 1e-12));
    }

    MassStream dilution{burn_air, m_dilution};
    save("dilution_air", dilution);
    MixResult diluted = thermo_.mix({products, m_prod}, dilution, P_burn);
    audit(diluted);
    save("dilution_exit", diluted.stream);
    MassStream turbine_in = diluted.stream;
    const double compressor_work = r.compressor.work_specific * m_core;
    if (ablation_level == 1) {
        const GasProperties pin = thermo_.properties(turbine_in.state);
        r.combustor = {turbine_in.state.T, turbine_in.state.P, pin.h, pin.cp,
                       pin.R, pin.gamma, turbine_in.state.Y};
        r.turbine = v6_.run_turbine_analytic(r.combustor, turbine_in.mass_flow,
                                              compressor_work + fan_work);
        save("turbine_exit", {{r.turbine.T, r.turbine.p, turbine_in.state.Y}, turbine_in.mass_flow});
    } else {
        MixResult ngv_mix = thermo_.mix(turbine_in, {burn_air, m_ngv}, P_burn);
        audit(ngv_mix);
        turbine_in = ngv_mix.stream;
        save("ngv_exit_hp_in", turbine_in);
        ExpansionResult hp = thermo_.expand_for_work(turbine_in, compressor_work,
                                                      c.eta_polytropic, config.pressure_steps);
        out.stages["HP"] = hp;
        out.max_energy_relative = std::max(out.max_energy_relative, hp.energy_relative);
        out.max_element_relative = std::max(out.max_element_relative, hp.element_relative);
        save("hp_rotor_exit", hp.stream);
        MixResult rotor_mix = thermo_.mix(hp.stream, {burn_air, m_rotor}, hp.stream.state.P);
        audit(rotor_mix);
        save("rotor_cooling_exit", rotor_mix.stream);
        ExpansionResult ip = thermo_.expand_for_work(rotor_mix.stream, 0.0,
                                                      c.eta_polytropic, config.pressure_steps);
        out.stages["IP"] = ip;
        save("ip_exit", ip.stream);
        ExpansionResult lp = thermo_.expand_for_work(ip.stream, fan_work,
                                                      c.eta_polytropic, config.pressure_steps);
        out.stages["LP"] = lp;
        out.max_energy_relative = std::max(out.max_energy_relative, lp.energy_relative);
        out.max_element_relative = std::max(out.max_element_relative, lp.element_relative);
        save("lp_exit", lp.stream);
        const GasProperties p_t4 = thermo_.properties(diluted.stream.state);
        r.combustor = {diluted.stream.state.T, diluted.stream.state.P,
                       p_t4.h, p_t4.cp, p_t4.R, p_t4.gamma,
                       diluted.stream.state.Y};
        const GasProperties p_out = thermo_.properties(lp.stream.state);
        const double area = c.A_combustor_exit * 1.82;
        const double u_out = lp.stream.mass_flow / (p_out.rho * area);
        r.turbine = {p_out.rho, u_out, lp.stream.state.P, lp.stream.state.T,
                     (compressor_work + fan_work) / lp.stream.mass_flow,
                     compressor_work + fan_work, p_out.cp, p_out.R, p_out.gamma};
        turbine_in = lp.stream;
    }
    save("core_nozzle_in", {{r.turbine.T, r.turbine.p, turbine_in.state.Y}, turbine_in.mass_flow});
    r.nozzle = v6_.run_nozzle(r.turbine, turbine_in.mass_flow);
    save("core_nozzle_exit", {{r.nozzle.T, r.nozzle.p, turbine_in.state.Y}, turbine_in.mass_flow});

    r.fuel_air_ratio = f;
    r.fuel_mass_flow = m_fuel;
    r.total_mass_flow = m_core + m_fuel;
    r.bypass_mass_flow = m_bypass;
    r.total_air_mass_flow = m_core + m_bypass;
    r.fan_work_W = fan_work;
    r.thrust_N = r.nozzle.thrust_total + (r.has_fan ? r.fan.thrust_bypass : 0.0);
    r.thrust_kN = r.thrust_N / 1000.0;
    r.thrust_core_kN = r.nozzle.thrust_total / 1000.0;
    r.thrust_bypass_kN = (r.has_fan ? r.fan.thrust_bypass : 0.0) / 1000.0;
    r.tsfc_SI = r.thrust_N > 0.0 ? m_fuel / r.thrust_N : std::numeric_limits<double>::infinity();
    r.tsfc_mg_per_Ns = r.tsfc_SI * 1e6;
    r.specific_thrust_Ns_kg = r.thrust_N / r.total_air_mass_flow;
    double ke_flux = 0.5 * turbine_in.mass_flow * r.nozzle.u * r.nozzle.u;
    if (r.has_fan) ke_flux += 0.5 * m_bypass * r.fan.u_bypass_exit * r.fan.u_bypass_exit;
    r.thermal_efficiency = m_fuel > 0.0 ? ke_flux / (m_fuel * 43e6) : 0.0;
    r.nox_g_s = v6_.estimate_nox(r.compressor.p_out / c.P_ambient, m_fuel);
    out.max_mass_relative = std::max(out.max_mass_relative,
        relative_error(turbine_in.mass_flow, m_core + m_fuel, 1e-12));
    return out;
}

P82AtThrustResult P82Engine::run_at_thrust(double target_kN, const std::string& fuel,
                                            const std::vector<std::string>& fuel_species,
                                            double combustor_efficiency, int ablation_level,
                                            double phi_lo, double phi_hi, double t4_max_K,
                                            double phi_xtol, double phi_guess)
{
    // The same cold bracketing/Brent procedure as the G0 path, with only the
    // cycle callback replaced. A cache ensures each phi is evaluated once.
    if (!(std::isfinite(target_kN) && target_kN > 0.0 &&
          std::isfinite(phi_lo) && std::isfinite(phi_hi) &&
          phi_lo > 0.0 && phi_lo < phi_hi &&
          std::isfinite(t4_max_K) && t4_max_K > 0.0 &&
          std::isfinite(phi_xtol) && phi_xtol > 0.0 &&
          combustor_efficiency > 0.0 && combustor_efficiency <= 1.0)) {
        throw std::invalid_argument("invalid P8.2 thrust-solve input");
    }
    std::map<double, std::variant<P82CycleResult, std::string>> cache;
    auto cycle = [&](double phi) -> const P82CycleResult& {
        auto it = cache.find(phi);
        if (it == cache.end()) {
            try {
                it = cache.emplace(phi, run_full_cycle(fuel, fuel_species, phi,
                                                        combustor_efficiency, ablation_level)).first;
            } catch (const CycleDoesNotClose& e) {
                it = cache.emplace(phi, std::string(e.what())).first;
            }
        }
        if (std::holds_alternative<std::string>(it->second)) {
            throw CycleDoesNotClose(std::get<std::string>(it->second));
        }
        return std::get<P82CycleResult>(it->second);
    };
    auto runs = [&](double phi) {
        try { cycle(phi); return true; }
        catch (const CycleDoesNotClose&) { return false; }
    };
    auto t4 = [&](double phi) { return cycle(phi).cycle.combustor.T_out; };
    auto residual = [&](double phi) { return cycle(phi).cycle.thrust_kN - target_kN; };
    double lo = phi_lo, hi = phi_hi;
    const double rtol = 4 * std::numeric_limits<double>::epsilon();
    std::map<std::string, double> info{{"phi_bounds_lo", phi_lo}, {"phi_bounds_hi", phi_hi},
                                       {"t4_max_K", t4_max_K}};
    auto guarded_hi = [&]() {
        if (t4(hi) <= t4_max_K) return hi;
        return brentq([&](double p) { return t4(p) - t4_max_K; }, lo, hi, phi_xtol);
    };
    if (!runs(lo)) {
        if (!runs(hi)) throw ThrustTargetUnreachable("cycle does not close anywhere in phi bounds", target_kN, info);
        double a = lo, b = hi;
        while (b - a > 1e-9 * b) {
            const double m = 0.5 * (a + b);
            if (runs(m)) b = m; else a = m;
        }
        lo = b;
        info["phi_lower_cycle_closure"] = lo;
    }
    const double upper = guarded_hi();
    info["phi_upper"] = upper;
    info["thrust_at_lower_kN"] = cycle(lo).cycle.thrust_kN;
    info["thrust_at_upper_kN"] = cycle(upper).cycle.thrust_kN;
    if (residual(lo) > 0.0) throw ThrustTargetUnreachable("below minimum thrust", target_kN, info);
    if (residual(upper) < 0.0) throw ThrustTargetUnreachable("above maximum thrust", target_kN, info);
    double a = lo, b = upper;
    if (std::isfinite(phi_guess) && lo < phi_guess && phi_guess < upper && runs(phi_guess)) {
        double p0 = phi_guess, p1 = p0, step = 1.02;
        double r0 = residual(p0);
        for (int i = 0; i < 12; ++i) {
            p1 = r0 < 0.0 ? std::min(p1 * step, upper) : std::max(p1 / step, lo);
            if (!runs(p1)) break;
            if ((residual(p1) > 0.0) != (r0 > 0.0)) {
                a = std::min(p0, p1); b = std::max(p0, p1); break;
            }
            if (p1 == lo || p1 == upper) break;
        }
    }
    const double phi = brentq(residual, a, b, phi_xtol, rtol);
    P82AtThrustResult result;
    result.result = cycle(phi);
    result.match = {phi, target_kN, residual(phi), t4(phi),
                    static_cast<int>(cache.size()), info};
    return result;
}

}  // namespace catjet
