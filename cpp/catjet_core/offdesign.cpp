#include "offdesign.hpp"

#include "brentq.hpp"

#include "cantera/base/stringUtils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace catjet {

namespace {

constexpr double LBM = 0.45359237;          // kg
constexpr double PSI = 6894.757293168361;   // Pa
constexpr double FT = 0.3048;               // m
constexpr double R_PER_K = 1.8;             // degR per K
constexpr double T_STD_R = 518.67;          // pyCycle T_STDeng
constexpr double P_STD_PSI = 14.695951;     // pyCycle P_STDeng
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

struct EvalError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// scipy.interpolate.Akima1DInterpolator (method 'akima'), as pyCycle's ambient.
double akima(const std::vector<double>& x, const std::vector<double>& y, double xi)
{
    const size_t n = x.size();
    if (n < 3 || y.size() != n) throw std::invalid_argument("Akima needs >= 3 matching points");
    std::vector<double> m(n - 1);
    for (size_t i = 0; i + 1 < n; ++i) m[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
    std::vector<double> m1(n + 3);
    const double mm = 2.0 * m[0] - m[1], mmm = 2.0 * mm - m[0];
    const double mp = 2.0 * m[n - 2] - m[n - 3], mpp = 2.0 * mp - m[n - 2];
    m1[0] = mmm;
    m1[1] = mm;
    for (size_t i = 0; i + 1 < n; ++i) m1[i + 2] = m[i];
    m1[n + 1] = mp;
    m1[n + 2] = mpp;
    std::vector<double> dm(n + 2);
    for (size_t i = 0; i + 1 < m1.size(); ++i) dm[i] = std::abs(m1[i + 1] - m1[i]);
    double f12max = 0.0;
    std::vector<double> f1(n), f2(n), f12(n), t(n);
    for (size_t i = 0; i < n; ++i) {
        f1[i] = dm[i + 2];
        f2[i] = dm[i];
        f12[i] = f1[i] + f2[i];
        f12max = std::max(f12max, f12[i]);
    }
    for (size_t i = 0; i < n; ++i) {
        t[i] = 0.5 * (m1[i + 3] + m1[i]);
        if (f12[i] > 1e-9 * f12max) t[i] = (f1[i] * m1[i + 1] + f2[i] * m1[i + 2]) / f12[i];
    }
    size_t i = std::upper_bound(x.begin(), x.end(), xi) - x.begin();
    i = std::clamp<size_t>(i, 1, n - 1) - 1;
    const double h = x[i + 1] - x[i], s = xi - x[i];
    const double c = (3.0 * m[i] - 2.0 * t[i] - t[i + 1]) / h;
    const double d = (t[i] + t[i + 1] - 2.0 * m[i]) / (h * h);
    return y[i] + t[i] * s + c * s * s + d * s * s * s;
}

void solve_linear(std::vector<std::vector<double>> A, std::vector<double>& b)
{
    // Gaussian elimination with partial pivoting (n <= 10).
    const size_t n = b.size();
    for (size_t k = 0; k < n; ++k) {
        size_t p = k;
        for (size_t i = k + 1; i < n; ++i) if (std::abs(A[i][k]) > std::abs(A[p][k])) p = i;
        if (std::abs(A[p][k]) < 1e-300) throw EvalError("singular Jacobian");
        std::swap(A[k], A[p]);
        std::swap(b[k], b[p]);
        for (size_t i = k + 1; i < n; ++i) {
            const double f = A[i][k] / A[k][k];
            for (size_t j = k; j < n; ++j) A[i][j] -= f * A[k][j];
            b[i] -= f * b[k];
        }
    }
    for (size_t k = n; k-- > 0;) {
        for (size_t j = k + 1; j < n; ++j) b[k] -= A[k][j] * b[j];
        b[k] /= A[k][k];
    }
}

double norm_inf(const std::vector<double>& r)
{
    double m = 0.0;
    for (double v : r) m = std::max(m, std::isfinite(v) ? std::abs(v) : std::numeric_limits<double>::infinity());
    return m;
}

double norm2(const std::vector<double>& r)
{
    double s = 0.0;
    for (double v : r) s += std::isfinite(v) ? v * v : std::numeric_limits<double>::infinity();
    return s;
}

}  // namespace

Hbtf::Hbtf(const std::string& mechanism, ThermoMode mode, const std::string& air,
           const std::string& fuel, std::map<std::string, double> fuel_element_weights)
    : mode_(mode), fuel_(fuel)
{
    sol_ = Cantera::newSolution(mechanism);
    gas_ = sol_->thermo();
    const size_t nk = gas_->nSpecies();
    if (mode_ == ThermoMode::Matched) {
        // air: element moles per gram (pyCycle b0), e.g. "Nx:0.0539,Ox:0.0145,Arx:3.2e-4,Cx:1.1e-5"
        Cantera::Composition b0 = Cantera::parseCompString(air);
        auto b = [&](const char* e) { return b0.count(e) ? b0.at(e) : 0.0; };
        Cantera::Composition X{{"CO2", b("Cx")}, {"O2", (b("Ox") - 2.0 * b("Cx")) / 2.0},
                               {"N2", b("Nx") / 2.0}, {"Ar", b("Arx")}};
        gas_->setState_TPX(300.0, Cantera::OneAtm, X);
        // fuel: element formula with pyCycle element weights (P8.4-A1)
        Cantera::Composition formula = Cantera::parseCompString(fuel);
        double M = 0.0;
        for (const auto& [e, n] : formula) M += n * fuel_element_weights.at(e);   // g/mol
        fuel_element_moles_per_kg_.assign(gas_->nElements(), 0.0);
        for (const auto& [e, n] : formula) {
            fuel_element_moles_per_kg_[gas_->elementIndex(e, true)] = n / M * 1000.0;  // mol/kg
        }
    } else {
        gas_->setState_TPX(300.0, Cantera::OneAtm, air);
        Y_fuel_.resize(nk);
        gas_->setState_TPX(300.0, Cantera::OneAtm, fuel);
        gas_->getMassFractions(Y_fuel_.data());
        gas_->setState_TPX(300.0, Cantera::OneAtm, air);
    }
    Y_air_.resize(nk);
    gas_->getMassFractions(Y_air_.data());
    if (mode_ == ThermoMode::Production) p83_ = std::make_unique<ChokingNozzle>(mechanism);
}

double Hbtf::us1976_T(double alt_m) const
{
    return akima(spec.atm_alt_ft, spec.atm_T_R, alt_m / FT) / R_PER_K;
}

double Hbtf::us1976_P(double alt_m) const
{
    return akima(spec.atm_alt_ft, spec.atm_P_psi, alt_m / FT) * PSI;
}

void Hbtf::set_Y(const std::vector<double>& Y)
{
    gas_->setMassFractions(Y.data());
}

namespace {
Flow read(Cantera::ThermoPhase& g, double W)
{
    Flow f;
    f.W = W;
    f.Tt = g.temperature();
    f.Pt = g.pressure();
    f.ht = g.enthalpy_mass();
    f.St = g.entropy_mass();
    f.Y.resize(g.nSpecies());
    g.getMassFractions(f.Y.data());
    return f;
}
}  // namespace

Flow Hbtf::state_TP(const std::vector<double>& Y, double T, double P, double W)
{
    set_Y(Y);
    gas_->setState_TP(T, P);
    if (mode_ == ThermoMode::Matched) gas_->equilibrate("TP", "auto", 1e-12, 5000, 500, 0, 0);
    return read(*gas_, W);
}

Flow Hbtf::state_hP(const std::vector<double>& Y, double h, double P, double W)
{
    set_Y(Y);
    gas_->setState_TP(1000.0, P);
    gas_->setState_HP(h, P, 1e-13);
    if (mode_ == ThermoMode::Matched) gas_->equilibrate("HP", "auto", 1e-12, 5000, 500, 0, 0);
    return read(*gas_, W);
}

Flow Hbtf::state_SP(const std::vector<double>& Y, double s, double P, double W)
{
    set_Y(Y);
    gas_->setState_TP(1000.0, P);
    gas_->setState_SP(s, P, 1e-13);
    if (mode_ == ThermoMode::Matched) gas_->equilibrate("SP", "auto", 1e-12, 5000, 500, 0, 0);
    return read(*gas_, W);
}

double Hbtf::sound_speed(const std::vector<double>& Y, double s, double P)
{
    if (mode_ == ThermoMode::Production) {
        state_SP(Y, s, P, 0.0);
        return gas_->soundSpeed();
    }
    // Equilibrium sound speed a^2 = (dP/drho)_s at shifting equilibrium (CEA gamma_s).
    const double e = 1e-5;
    state_SP(Y, s, P * (1.0 + e), 0.0);
    const double rp = gas_->density();
    state_SP(Y, s, P * (1.0 - e), 0.0);
    const double rm = gas_->density();
    return std::sqrt(2.0 * e * P / (rp - rm));
}

ThroatState Hbtf::static_at_Ps(const Flow& f, double Ps)
{
    ThroatState t;
    Flow s = state_SP(f.Y, f.St, Ps, f.W);
    t.Ps = Ps;
    t.Ts = s.Tt;
    t.rho = gas_->density();
    const double dh = f.ht - s.ht;
    if (!(dh > 0.0)) throw EvalError("non-positive throat enthalpy drop");
    t.V = std::sqrt(2.0 * dh);
    t.area = f.W / (t.rho * t.V);
    t.MN = t.V / sound_speed(f.Y, f.St, Ps);
    return t;
}

ThroatState Hbtf::static_at_MN(const Flow& f, double MN)
{
    auto resid = [&](double Ps) {
        Flow s = state_SP(f.Y, f.St, Ps, f.W);
        const double a = sound_speed(f.Y, f.St, Ps);
        return (f.ht - s.ht) - 0.5 * MN * MN * a * a;
    };
    double lo = f.Pt * 0.5;
    while (resid(lo) < 0.0) {
        lo *= 0.5;
        if (lo < f.Pt * 1e-6) throw EvalError("no static pressure bracket for the Mach number");
    }
    const double Ps = brentq(resid, lo, f.Pt, 1e-12 * f.Pt);
    return static_at_Ps(f, Ps);
}

Flow Hbtf::mix(const std::vector<Flow>& flows, double P)
{
    double W = 0.0, H = 0.0;
    std::vector<double> Y(gas_->nSpecies(), 0.0);
    for (const auto& f : flows) {
        W += f.W;
        H += f.W * f.ht;
        for (size_t k = 0; k < Y.size(); ++k) Y[k] += f.W * f.Y[k];
    }
    for (double& y : Y) y /= W;
    return state_hP(Y, H / W, P, W);
}

double Hbtf::fuel_enthalpy(const Flow& air)
{
    if (mode_ == ThermoMode::Matched) return 0.0;   // pyCycle mix:h default (P8.4-A1)
    set_Y(Y_fuel_);
    gas_->setState_TP(air.Tt, air.Pt);
    return gas_->enthalpy_mass() - 360000.0;        // liquid basis, v7 value
}

std::vector<double> Hbtf::burner_Y(const Flow& air, double W_fuel)
{
    const size_t nk = gas_->nSpecies();
    std::vector<double> Y(nk);
    if (mode_ == ThermoMode::Production) {
        for (size_t k = 0; k < nk; ++k) Y[k] = (air.W * air.Y[k] + W_fuel * Y_fuel_[k]) / (air.W + W_fuel);
        return Y;
    }
    // element moles per kg of mixture, then an element-consistent guess
    set_Y(air.Y);
    gas_->setState_TP(air.Tt, air.Pt);
    const size_t ne = gas_->nElements();
    std::vector<double> b(ne);
    for (size_t e = 0; e < ne; ++e) {
        const double air_mol_per_kg = gas_->elementalMassFraction(e) / gas_->atomicWeight(e) * 1000.0;
        b[e] = (air.W * air_mol_per_kg + W_fuel * fuel_element_moles_per_kg_[e]) / (air.W + W_fuel);
    }
    auto at = [&](const char* el) {
        const size_t e = gas_->elementIndex(el, false);
        return e == Cantera::npos ? 0.0 : b[e];
    };
    const double co2 = at("Cx"), h2o = at("Hx") / 2.0;
    const double o2 = (at("Ox") - 2.0 * co2 - h2o) / 2.0;
    if (o2 < 0.0) throw EvalError("rich burner mixture not supported by the initial guess");
    Cantera::Composition X{{"CO2", co2}, {"H2O", h2o}, {"O2", o2}, {"N2", at("Nx") / 2.0},
                           {"Ar", at("Arx")}};
    gas_->setState_TPX(air.Tt, air.Pt, X);
    gas_->getMassFractions(Y.data());
    return Y;
}

std::vector<double> Hbtf::residuals(const std::vector<double>& x, bool design_mode, CycleOutputs* out)
{
    const HbtfSpec& s = spec;
    const double alt = design_mode ? s.alt_m : alt_;
    const double MN = design_mode ? s.MN : MN_;
    const double dTs = design_mode ? s.dTs_K : dTs_;
    const double W = x[0], FAR = x[1];
    const double BPR = design_mode ? s.BPR_des : x[2];
    const double N_lp = design_mode ? s.N_lp_des : x[3];
    const double N_hp = design_mode ? s.N_hp_des : x[4];
    const double PR_hpt = design_mode ? x[2] : x[8];
    const double PR_lpt = design_mode ? x[3] : x[9];
    bool extrap = false;
    std::vector<std::string> extrap_maps;
    std::vector<double> r;

    // flight conditions and inlet
    const double Ts = us1976_T(alt) + dTs, Ps = us1976_P(alt);
    Flow stat = state_TP(Y_air_, Ts, Ps, W);
    const double V0 = MN * sound_speed(stat.Y, stat.St, Ps);
    const double ht0 = stat.ht + 0.5 * V0 * V0;
    double Pt0 = Ps;
    if (MN > 0.0) {
        auto f = [&](double P) { return state_SP(stat.Y, stat.St, P, W).ht - ht0; };
        double hiP = Ps * 1.5;
        while (f(hiP) < 0.0) hiP *= 1.5;
        Pt0 = brentq(f, Ps, hiP, 1e-12 * Ps);
    }
    Flow fc = state_SP(stat.Y, stat.St, Pt0, W);
    const double ram = MN < 1.0 ? s.ram_recovery : s.ram_recovery * (1 - 0.075 * std::pow(MN - 1, 1.35));
    Flow inlet = state_hP(fc.Y, fc.ht, fc.Pt * ram, W);
    const double F_ram = W * V0;

    auto duct = [&](const Flow& f, double dPqP) { return state_hP(f.Y, f.ht, f.Pt * (1.0 - dPqP), f.W); };

    // compressor (pyCycle Compressor + CompressorMap + BleedsAndPower)
    struct CompOut { Flow out; std::map<std::string, Flow> bleeds; double power, PR, eff, Wc, Nc, map_resid; };
    auto compressor = [&](const CompressorSpec& c, CompressorScalars& sc, const Flow& in, double N, double Rline) {
        CompOut o{};
        const double theta = in.Tt * R_PER_K / T_STD_R, delta = in.Pt / PSI / P_STD_PSI;
        o.Wc = in.W / LBM * std::sqrt(theta) / delta;
        o.Nc = N / std::sqrt(theta);
        const double alpha = c.map.defaults.at("alphaMap");
        if (design_mode) {
            std::vector<double> xm{alpha, c.map.defaults.at("NcMap"), c.map.defaults.at("RlineMap")};
            const double WcMap = c.map.eval("WcMap", xm, nullptr), PRmap = c.map.eval("PRmap", xm, nullptr);
            const double effMap = c.map.eval("effMap", xm, nullptr);
            sc.s_Nc = o.Nc / xm[1];
            sc.s_PR = (c.PR_des - 1.0) / (PRmap - 1.0);
            sc.s_eff = c.eff_des / effMap;
            sc.s_Wc = o.Wc / WcMap;
            sc.Wc_des = o.Wc;
            o.PR = c.PR_des;
            o.eff = c.eff_des;
            o.map_resid = 0.0;
        } else {
            bool ex = false;
            std::vector<double> xm{alpha, o.Nc / sc.s_Nc, Rline};
            o.PR = (c.map.eval("PRmap", xm, &ex) - 1.0) * sc.s_PR + 1.0;
            o.eff = c.map.eval("effMap", xm, &ex) * sc.s_eff;
            o.map_resid = (c.map.eval("WcMap", xm, &ex) * sc.s_Wc - o.Wc) / sc.Wc_des;
            if (ex) { extrap = true; extrap_maps.push_back(c.name); }
        }
        const double Pt_out = o.PR * in.Pt;
        const double h_ideal = state_SP(in.Y, in.St, Pt_out, in.W).ht;
        const double ht_out = in.ht + (h_ideal - in.ht) / o.eff;
        o.out = state_hP(in.Y, ht_out, Pt_out, in.W);
        o.power = in.W * (in.ht - ht_out);
        for (const auto& b : c.bleeds) {
            const double Wb = b.frac_W * in.W;
            const double htb = in.ht + b.frac_work * (ht_out - in.ht);
            const double Ptb = in.Pt + b.frac_P * (Pt_out - in.Pt);
            o.bleeds[b.name] = state_hP(in.Y, htb, Ptb, Wb);
            o.power -= Wb * (htb - ht_out);
            o.out.W -= Wb;
        }
        return o;
    };

    // turbine (pyCycle Turbine + TurbineMap + EnthalpyAndPower), bleeds with entry frac_P
    struct TurbOut { Flow out; double power, eff, Wp, Np, map_resid; };
    auto turbine = [&](const TurbineSpec& t, TurbineScalars& sc, const Flow& in,
                       const std::vector<std::pair<Flow, double>>& bleeds, double N, double PR) {
        TurbOut o{};
        o.Wp = in.W / LBM * std::sqrt(in.Tt * R_PER_K) / (in.Pt / PSI);
        o.Np = N / std::sqrt(in.Tt * R_PER_K);
        const double alpha = t.map.defaults.at("alphaMap");
        if (design_mode) {
            const double NpD = t.map.defaults.at("NpMap"), PRD = t.map.defaults.at("PRmap");
            std::vector<double> xm{alpha, NpD, PRD};
            const double WpMap = t.map.eval("WpMap", xm, nullptr), effMap = t.map.eval("effMap", xm, nullptr);
            sc.s_Np = o.Np / NpD;
            sc.s_PR = (PR - 1.0) / (PRD - 1.0);
            sc.s_eff = t.eff_des / effMap;
            sc.s_Wp = o.Wp / WpMap;
            sc.Wp_des = o.Wp;
            o.eff = t.eff_des;
            o.map_resid = 0.0;
        } else {
            bool ex = false;
            std::vector<double> xm{alpha, o.Np / sc.s_Np, (PR - 1.0) / sc.s_PR + 1.0};
            o.eff = t.map.eval("effMap", xm, &ex) * sc.s_eff;
            o.map_resid = (t.map.eval("WpMap", xm, &ex) * sc.s_Wp - o.Wp) / sc.Wp_des;
            if (ex) { extrap = true; extrap_maps.push_back(t.name); }
        }
        const double Pt_out = in.Pt / PR;
        const double h_ideal = state_SP(in.Y, in.St, Pt_out, in.W).ht;
        double W_out = in.W, H = in.W * (in.ht * (1.0 - o.eff) + h_ideal * o.eff);
        o.power = in.W * o.eff * (in.ht - h_ideal);
        std::vector<double> Y(in.Y.size());
        for (size_t k = 0; k < Y.size(); ++k) Y[k] = in.W * in.Y[k];
        for (const auto& [b, frac_P] : bleeds) {
            const double Ptb = Pt_out + frac_P * (in.Pt - Pt_out);
            const Flow bin = state_hP(b.Y, b.ht, Ptb, b.W);
            const double hbi = state_SP(b.Y, bin.St, Pt_out, b.W).ht;
            H += b.W * (b.ht * (1.0 - o.eff) + hbi * o.eff);
            o.power += b.W * o.eff * (b.ht - hbi);
            W_out += b.W;
            for (size_t k = 0; k < Y.size(); ++k) Y[k] += b.W * b.Y[k];
        }
        for (double& y : Y) y /= W_out;
        o.out = state_hP(Y, H / W_out, Pt_out, W_out);
        return o;
    };

    struct NozOut { ThroatState th; double Fg; bool choked; };
    auto nozzle = [&](const Flow& in, double Cv) {
        NozOut o{};
        const ThroatState star = static_at_MN(in, 1.0);
        o.choked = Ps < star.Ps;
        o.th = o.choked ? star : static_at_Ps(in, Ps);
        o.Fg = in.W * o.th.V * Cv + (o.th.Ps - Ps) * o.th.area;
        return o;
    };

    const double R_fan = design_mode ? 0.0 : x[5], R_lpc = design_mode ? 0.0 : x[6];
    const double R_hpc = design_mode ? 0.0 : x[7];
    CompOut fan = compressor(s.fan, design.fan, inlet, N_lp, R_fan);
    Flow core_in = fan.out, byp_in = fan.out;
    core_in.W = fan.out.W / (BPR + 1.0);
    byp_in.W = fan.out.W - core_in.W;
    Flow d4 = duct(core_in, s.dPqP_duct4);
    CompOut lpc = compressor(s.lpc, design.lpc, d4, N_lp, R_lpc);
    Flow d6 = duct(lpc.out, s.dPqP_duct6);
    CompOut hpc = compressor(s.hpc, design.hpc, d6, N_hp, R_hpc);
    Flow bld3 = hpc.out;
    Flow cool3 = hpc.out, cool4 = hpc.out;
    cool3.W = s.cool3_frac_W * hpc.out.W;
    cool4.W = s.cool4_frac_W * hpc.out.W;
    bld3.W = hpc.out.W - cool3.W - cool4.W;
    const double W_fuel = FAR * bld3.W;
    const double h_fuel = fuel_enthalpy(bld3);
    const double h4 = (bld3.W * bld3.ht + W_fuel * h_fuel) / (bld3.W + W_fuel);
    Flow burner = state_hP(burner_Y(bld3, W_fuel), h4, bld3.Pt * (1.0 - s.dPqP_burner), bld3.W + W_fuel);
    if (mode_ == ThermoMode::Production) {
        // production thermo is frozen outside the burner; the burner itself is HP equilibrium
        gas_->equilibrate("HP", "auto", 1e-12, 5000, 500, 0, 0);
        burner = read(*gas_, burner.W);
    }
    TurbOut hpt = turbine(s.hpt, design.hpt, burner, {{cool3, s.cool3_frac_P}, {cool4, s.cool4_frac_P}},
                          N_hp, PR_hpt);
    Flow d11 = duct(hpt.out, s.dPqP_duct11);
    TurbOut lpt = turbine(s.lpt, design.lpt, d11,
                          {{hpc.bleeds.at("cool1"), s.cool1_frac_P_lpt},
                           {hpc.bleeds.at("cool2"), s.cool2_frac_P_lpt}}, N_lp, PR_lpt);
    Flow d13 = duct(lpt.out, s.dPqP_duct13);
    NozOut core = nozzle(d13, s.Cv_core);
    Flow byp_bld = byp_in;
    byp_bld.W = byp_in.W * (1.0 - s.frac_byp_bleed);
    Flow d15 = duct(byp_bld, s.dPqP_duct15);
    NozOut byp = nozzle(d15, s.Cv_byp);

    const double Fg = core.Fg + byp.Fg, Fn = Fg - F_ram;
    const double lp_net = fan.power + lpc.power + lpt.power;
    const double hp_net = hpc.power + hpt.power - s.HPX_W;

    if (design_mode) {
        r = {(Fn - s.Fn_des_N) / s.Fn_des_N, (burner.Tt - s.T4_max_K) / s.T4_max_K,
             lp_net / std::max(std::abs(lpt.power), 1.0), hp_net / std::max(std::abs(hpt.power), 1.0)};
        design.A_core = core.th.area;
        design.A_byp = byp.th.area;
        design.P_hpt = hpt.power;
        design.P_lpt = lpt.power;
        design.W_des = W;
    } else {
        const double thr = throttle_ == "T4" ? (burner.Tt - throttle_target_) / s.T4_max_K
                                             : (Fn - throttle_target_) / s.Fn_des_N;
        r = {thr, (core.th.area - design.A_core) / design.A_core, (byp.th.area - design.A_byp) / design.A_byp,
             lp_net / design.P_lpt, hp_net / design.P_hpt, fan.map_resid, lpc.map_resid, hpc.map_resid,
             hpt.map_resid, lpt.map_resid};
    }

    if (out) {
        auto& o = *out;
        o.extrapolated = extrap;
        o.extrapolated_maps = extrap_maps;
        o.stations = {{"fc", fc}, {"inlet", inlet}, {"fan", fan.out}, {"duct4", d4}, {"lpc", lpc.out},
                      {"duct6", d6}, {"hpc", hpc.out}, {"bld3", bld3}, {"burner", burner},
                      {"hpt", hpt.out}, {"duct11", d11}, {"lpt", lpt.out}, {"duct13", d13},
                      {"byp_bld", byp_bld}, {"duct15", d15}, {"splitter1", core_in},
                      {"splitter2", byp_in}, {"core_nozz", d13}, {"byp_nozz", d15}};
        const Flow& cust = hpc.bleeds.at("cust");
        auto& v = o.scalars;
        v = {{"W", W}, {"FAR", FAR}, {"BPR", BPR}, {"N_lp", N_lp}, {"N_hp", N_hp},
             {"OPR", hpc.out.Pt / inlet.Pt}, {"Fn_N", Fn}, {"Fg_N", Fg}, {"F_ram_N", F_ram},
             {"Wfuel", W_fuel}, {"TSFC_kg_per_N_s", W_fuel / Fn}, {"Tt3", hpc.out.Tt}, {"Tt4", burner.Tt},
             {"PR_fan", fan.PR}, {"PR_lpc", lpc.PR}, {"PR_hpc", hpc.PR}, {"eff_fan", fan.eff},
             {"eff_lpc", lpc.eff}, {"eff_hpc", hpc.eff}, {"PR_hpt", PR_hpt}, {"PR_lpt", PR_lpt},
             {"eff_hpt", hpt.eff}, {"eff_lpt", lpt.eff}, {"P_hpt_W", hpt.power}, {"P_lpt_W", lpt.power},
             {"P_fan_W", fan.power}, {"P_lpc_W", lpc.power}, {"P_hpc_W", hpc.power},
             {"A_core_m2", core.th.area}, {"A_byp_m2", byp.th.area}, {"core_choked", core.choked},
             {"byp_choked", byp.choked}, {"Fg_core_N", core.Fg}, {"Fg_byp_N", byp.Fg},
             {"Ts0", Ts}, {"Ps0", Ps}, {"V0", V0}, {"R_fan", R_fan}, {"R_lpc", R_lpc}, {"R_hpc", R_hpc}};
        // closures: mass and energy over the whole engine; element mass in vs out
        const double W_out = d13.W + d15.W + cust.W + (byp_in.W - byp_bld.W);
        o.mass_closure = std::abs(W + W_fuel - W_out) / (W + W_fuel);
        const double E_in = W * fc.ht + W_fuel * h_fuel;
        const double E_out = d13.W * d13.ht + d15.W * d15.ht + cust.W * cust.ht +
                             (byp_in.W - byp_bld.W) * byp_in.ht + s.HPX_W + (lp_net + hp_net);
        o.energy_closure = std::abs(E_in - E_out) /
                           (std::abs(W * fc.ht) + std::abs(W_fuel * h_fuel) + std::abs(s.HPX_W) + 1.0);
        // element mass flows (kg/s) in (air + fuel) and out (nozzles, overboard bleeds)
        const size_t ne = gas_->nElements();
        auto elements = [&](const Flow& f, std::vector<double>& acc, double sign) {
            set_Y(f.Y);
            for (size_t e = 0; e < ne; ++e) acc[e] += sign * f.W * gas_->elementalMassFraction(e);
        };
        std::vector<double> bal(ne, 0.0), scale_e(ne, 0.0);
        elements(fc, bal, 1.0);
        elements(fc, scale_e, 1.0);
        for (size_t e = 0; e < ne; ++e) {
            const double fuel_e = mode_ == ThermoMode::Matched
                ? W_fuel * fuel_element_moles_per_kg_[e] * gas_->atomicWeight(e) / 1000.0
                : (set_Y(Y_fuel_), W_fuel * gas_->elementalMassFraction(e));
            bal[e] += fuel_e;
            scale_e[e] += fuel_e;
        }
        Flow dumped = byp_in;
        dumped.W = byp_in.W - byp_bld.W;
        for (const Flow* f : std::initializer_list<const Flow*>{&d13, &d15, &cust, &dumped}) elements(*f, bal, -1.0);
        o.element_closure = 0.0;
        for (size_t e = 0; e < ne; ++e) {
            if (scale_e[e] > 1e-12) o.element_closure = std::max(o.element_closure, std::abs(bal[e]) / scale_e[e]);
        }
    }
    return r;
}

// P8.4b three-shaft separate-flow turbofan (docs/phase8_p84b_registration.md).
// Element equations are the same as residuals() (two-shaft, P8.4 G1); the
// two-shaft function is kept unchanged so its verified result cannot move.
std::vector<double> Hbtf::residuals3(const std::vector<double>& x, bool design_mode, CycleOutputs* out)
{
    const HbtfSpec& s = spec;
    const double alt = design_mode ? s.alt_m : alt_;
    const double MN = design_mode ? s.MN : MN_;
    const double dTs = design_mode ? s.dTs_K : dTs_;
    const double W = x[0], FAR = x[1];
    const double BPR = design_mode ? s.BPR_des : x[2];
    const double N_lp = design_mode ? s.N_lp_des : x[3];
    const double N_ip = design_mode ? s.N_ip_des : x[4];
    const double N_hp = design_mode ? s.N_hp_des : x[5];
    const double R_fan = design_mode ? 0.0 : x[6], R_ipc = design_mode ? 0.0 : x[7];
    const double R_hpc = design_mode ? 0.0 : x[8];
    const double PR_hpt = design_mode ? x[2] : x[9];
    const double PR_ipt = design_mode ? x[3] : x[10];
    const double PR_lpt = design_mode ? x[4] : x[11];
    const bool bleed_on = !design_mode && s.ipc_bleed_active;
    const double beta = bleed_on ? x[12] : 0.0;
    bool extrap = false;
    std::vector<std::string> extrap_maps;

    const double Ts = s.Ts_override_K > 0.0 ? s.Ts_override_K : us1976_T(alt) + dTs;
    const double Ps = s.Ps_override_Pa > 0.0 ? s.Ps_override_Pa : us1976_P(alt);
    Flow stat = state_TP(Y_air_, Ts, Ps, W);
    const double V0 = MN * sound_speed(stat.Y, stat.St, Ps);
    const double ht0 = stat.ht + 0.5 * V0 * V0;
    double Pt0 = Ps;
    if (MN > 0.0) {
        auto f = [&](double P) { return state_SP(stat.Y, stat.St, P, W).ht - ht0; };
        double hiP = Ps * 1.5;
        while (f(hiP) < 0.0) hiP *= 1.5;
        Pt0 = brentq(f, Ps, hiP, 1e-12 * Ps);
    }
    Flow fc = state_SP(stat.Y, stat.St, Pt0, W);
    const double ram = MN < 1.0 ? s.ram_recovery : s.ram_recovery * (1 - 0.075 * std::pow(MN - 1, 1.35));
    Flow inlet = state_hP(fc.Y, fc.ht, fc.Pt * ram, W);
    const double F_ram = W * V0;
    auto duct = [&](const Flow& f, double dPqP) { return state_hP(f.Y, f.ht, f.Pt * (1.0 - dPqP), f.W); };

    struct CompOut { Flow out; double power, PR, eff, map_resid, smn; };
    auto compressor = [&](const CompressorSpec& c, CompressorScalars& sc, const Flow& in, double N, double Rline) {
        CompOut o{};
        const double theta = in.Tt * R_PER_K / T_STD_R, delta = in.Pt / PSI / P_STD_PSI;
        const double Wc = in.W / LBM * std::sqrt(theta) / delta, Nc = N / std::sqrt(theta);
        const double alpha = c.map.defaults.at("alphaMap");
        if (design_mode) {
            std::vector<double> xm{alpha, c.map.defaults.at("NcMap"), c.map.defaults.at("RlineMap")};
            sc.s_Nc = Nc / xm[1];
            sc.s_PR = (c.PR_des - 1.0) / (c.map.eval("PRmap", xm, nullptr) - 1.0);
            sc.s_eff = c.eff_des / c.map.eval("effMap", xm, nullptr);
            sc.s_Wc = Wc / c.map.eval("WcMap", xm, nullptr);
            sc.Wc_des = Wc;
            o.PR = c.PR_des;
            o.eff = c.eff_des;
            // pyCycle StallCalcs.SMN on unscaled map values (P8.4b-A1)
            std::vector<double> xs{alpha, xm[1], c.map.rline_stall};
            o.smn = ((c.map.eval("WcMap", xm, nullptr) / c.map.eval("WcMap", xs, nullptr)) /
                     (c.map.eval("PRmap", xm, nullptr) / c.map.eval("PRmap", xs, nullptr)) - 1.0) * 100.0;
        } else {
            bool ex = false;
            std::vector<double> xm{alpha, Nc / sc.s_Nc, Rline};
            o.PR = (c.map.eval("PRmap", xm, &ex) - 1.0) * sc.s_PR + 1.0;
            o.eff = c.map.eval("effMap", xm, &ex) * sc.s_eff;
            o.map_resid = (c.map.eval("WcMap", xm, &ex) * sc.s_Wc - Wc) / sc.Wc_des;
            if (ex) { extrap = true; extrap_maps.push_back(c.name); }
            std::vector<double> xs{alpha, xm[1], c.map.rline_stall};
            o.smn = ((c.map.eval("WcMap", xm, nullptr) / c.map.eval("WcMap", xs, nullptr)) /
                     (c.map.eval("PRmap", xm, nullptr) / c.map.eval("PRmap", xs, nullptr)) - 1.0) * 100.0;
        }
        const double Pt_out = o.PR * in.Pt;
        const double h_ideal = state_SP(in.Y, in.St, Pt_out, in.W).ht;
        const double ht_out = in.ht + (h_ideal - in.ht) / o.eff;
        o.out = state_hP(in.Y, ht_out, Pt_out, in.W);
        o.power = in.W * (in.ht - ht_out);
        return o;
    };
    struct TurbOut { Flow out; double power, eff, map_resid; };
    auto turbine = [&](const TurbineSpec& t, TurbineScalars& sc, const Flow& in,
                       const std::vector<std::pair<Flow, double>>& bleeds, double N, double PR) {
        TurbOut o{};
        const double Wp = in.W / LBM * std::sqrt(in.Tt * R_PER_K) / (in.Pt / PSI);
        const double Np = N / std::sqrt(in.Tt * R_PER_K);
        const double alpha = t.map.defaults.at("alphaMap");
        if (design_mode) {
            const double NpD = t.map.defaults.at("NpMap"), PRD = t.map.defaults.at("PRmap");
            std::vector<double> xm{alpha, NpD, PRD};
            sc.s_Np = Np / NpD;
            sc.s_PR = (PR - 1.0) / (PRD - 1.0);
            sc.s_eff = t.eff_des / t.map.eval("effMap", xm, nullptr);
            sc.s_Wp = Wp / t.map.eval("WpMap", xm, nullptr);
            sc.Wp_des = Wp;
            o.eff = t.eff_des;
        } else {
            bool ex = false;
            std::vector<double> xm{alpha, Np / sc.s_Np, (PR - 1.0) / sc.s_PR + 1.0};
            o.eff = t.map.eval("effMap", xm, &ex) * sc.s_eff;
            o.map_resid = (t.map.eval("WpMap", xm, &ex) * sc.s_Wp - Wp) / sc.Wp_des;
            if (ex) { extrap = true; extrap_maps.push_back(t.name); }
        }
        const double Pt_out = in.Pt / PR;
        const double h_ideal = state_SP(in.Y, in.St, Pt_out, in.W).ht;
        double W_out = in.W, H = in.W * (in.ht * (1.0 - o.eff) + h_ideal * o.eff);
        o.power = in.W * o.eff * (in.ht - h_ideal);
        std::vector<double> Y(in.Y.size());
        for (size_t k = 0; k < Y.size(); ++k) Y[k] = in.W * in.Y[k];
        for (const auto& [b, frac_P] : bleeds) {
            const double Ptb = Pt_out + frac_P * (in.Pt - Pt_out);
            const Flow bin = state_hP(b.Y, b.ht, Ptb, b.W);
            const double hbi = state_SP(b.Y, bin.St, Pt_out, b.W).ht;
            H += b.W * (b.ht * (1.0 - o.eff) + hbi * o.eff);
            o.power += b.W * o.eff * (b.ht - hbi);
            W_out += b.W;
            for (size_t k = 0; k < Y.size(); ++k) Y[k] += b.W * b.Y[k];
        }
        for (double& y : Y) y /= W_out;
        o.out = state_hP(Y, H / W_out, Pt_out, W_out);
        return o;
    };
    // nozzle: P8.3 law (production) or pyCycle CV (matched); returns thrust and capacity
    struct NozOut { double Fg = 0, capacity = 0, area = 0; bool choked = false; };
    auto nozzle = [&](const Flow& in, double Cd, double Cv, double design_area) {
        NozOut o{};
        if (s.nozzle_p83) {
            if (!p83_) throw EvalError("P8.3 nozzle law needs production thermo");
            const GasState g{in.Tt, in.Pt, in.Y};
            if (design_mode) {
                const double pstar = p83_->critical_pressure(g);
                const double pexit = std::max(Ps, pstar);
                o.area = in.W / (Cd * p83_->mass_flux(g, pexit));
            } else {
                o.area = design_area;
            }
            const ChokingNozzleResult r = p83_->run(g, Ps, o.area, Cd, Cv);
            o.Fg = r.thrust_total;
            o.capacity = r.mass_flow;
            o.choked = r.choked;
        } else {
            const ThroatState star = static_at_MN(in, 1.0);
            o.choked = Ps < star.Ps;
            const ThroatState th = o.choked ? star : static_at_Ps(in, Ps);
            o.Fg = in.W * th.V * Cv + (th.Ps - Ps) * th.area;
            o.area = th.area;
            o.capacity = design_mode ? in.W : in.W * design_area / th.area;   // area-equivalent capacity
        }
        return o;
    };

    CompOut fan = compressor(s.fan, design.fan, inlet, N_lp, R_fan);
    Flow core_in = fan.out, byp_in = fan.out;
    core_in.W = fan.out.W / (BPR + 1.0);
    byp_in.W = fan.out.W - core_in.W;
    Flow d_fi = duct(core_in, s.dPqP_duct4);
    CompOut ipc = compressor(s.ipc, design.ipc, d_fi, N_ip, R_ipc);
    // P8.4b-A1 handling bleed: beta of IPC exit flow, mixed into the bypass at bypass pressure
    Flow hbleed = ipc.out;
    hbleed.W = beta * ipc.out.W;
    ipc.out.W -= hbleed.W;
    Flow d_ih = duct(ipc.out, s.dPqP_duct6);
    CompOut hpc = compressor(s.hpc, design.hpc, d_ih, N_hp, R_hpc);
    Flow bld = hpc.out, cool3 = hpc.out, cool4 = hpc.out;
    cool3.W = s.cool3_frac_W * hpc.out.W;
    cool4.W = s.cool4_frac_W * hpc.out.W;
    bld.W = hpc.out.W - cool3.W - cool4.W;
    const double W_fuel = FAR * bld.W;
    const double h_fuel = fuel_enthalpy(bld);
    const double h4 = (bld.W * bld.ht + W_fuel * h_fuel) / (bld.W + W_fuel);
    const double P4 = bld.Pt * (1.0 - s.dPqP_burner);
    Flow burner = state_hP(burner_Y(bld, W_fuel), h4, P4, bld.W + W_fuel);
    double Q_rej = 0.0;
    if (mode_ == ThermoMode::Production) {
        gas_->equilibrate("HP", "auto", 1e-12, 5000, 500, 0, 0);
        Flow eq = read(*gas_, burner.W);
        // v6 / P8.2 eta_b convention: scale the temperature rise at the equilibrium composition
        const double T_out = bld.Tt + s.eta_b * (eq.Tt - bld.Tt);
        burner = state_TP(eq.Y, T_out, P4, eq.W);
        Q_rej = eq.W * (eq.ht - burner.ht);
    }
    TurbOut hpt = turbine(s.hpt, design.hpt, burner, {{cool3, s.cool3_frac_P}, {cool4, s.cool4_frac_P}},
                          N_hp, PR_hpt);
    Flow d_hi = duct(hpt.out, s.dPqP_duct11);
    TurbOut ipt = turbine(s.ipt, design.ipt, d_hi, {}, N_ip, PR_ipt);
    Flow d_il = duct(ipt.out, s.dPqP_duct_ipt_lpt);
    TurbOut lpt = turbine(s.lpt, design.lpt, d_il, {}, N_lp, PR_lpt);
    Flow d_ln = duct(lpt.out, s.dPqP_duct13);
    Flow byp_bld = byp_in;
    byp_bld.W = byp_in.W * (1.0 - s.frac_byp_bleed);
    Flow byp_mixed = hbleed.W > 0.0 ? mix({byp_bld, hbleed}, byp_bld.Pt) : byp_bld;
    Flow d_b = duct(byp_mixed, s.dPqP_duct15);
    NozOut core = nozzle(d_ln, s.Cd_core, s.Cv_core, design.A_core);
    NozOut byp = nozzle(d_b, s.Cd_byp, s.Cv_byp, design.A_byp);

    const double Fg = core.Fg + byp.Fg, Fn = Fg - F_ram;
    const double lp_net = fan.power + lpt.power;
    const double ip_net = ipc.power + ipt.power;
    const double hp_net = hpc.power + hpt.power - s.HPX_W;
    std::vector<double> r;
    if (design_mode) {
        r = {(Fn - s.Fn_des_N) / s.Fn_des_N, (burner.Tt - s.T4_max_K) / s.T4_max_K,
             lp_net / std::max(std::abs(lpt.power), 1.0), ip_net / std::max(std::abs(ipt.power), 1.0),
             hp_net / std::max(std::abs(hpt.power), 1.0)};
        design.A_core = core.area;
        design.A_byp = byp.area;
        design.P_hpt = hpt.power;
        design.P_ipt = ipt.power;
        design.P_lpt = lpt.power;
        design.W_des = W;
        design.W_core_noz = d_ln.W;
        design.W_byp_noz = d_b.W;
    } else {
        const double thr = throttle_ == "T4" ? (burner.Tt - throttle_target_) / s.T4_max_K
                                             : (Fn - throttle_target_) / s.Fn_des_N;
        r = {thr, (core.capacity - d_ln.W) / design.W_core_noz, (byp.capacity - d_b.W) / design.W_byp_noz,
             lp_net / design.P_lpt, ip_net / design.P_ipt, hp_net / design.P_hpt,
             fan.map_resid, ipc.map_resid, hpc.map_resid, hpt.map_resid, ipt.map_resid, lpt.map_resid};
        if (bleed_on) r.push_back((ipc.smn - s.sm_floor_pct) / 100.0);
    }
    if (out) {
        auto& o = *out;
        o.extrapolated = extrap;
        o.extrapolated_maps = extrap_maps;
        o.stations = {{"fc", fc}, {"inlet", inlet}, {"fan", fan.out}, {"splitter1", core_in},
                      {"splitter2", byp_in}, {"ipc", ipc.out}, {"hpc", hpc.out}, {"burner", burner},
                      {"hpt", hpt.out}, {"ipt", ipt.out}, {"lpt", lpt.out}, {"core_nozz", d_ln},
                      {"byp_nozz", d_b}};
        o.scalars = {{"W", W}, {"FAR", FAR}, {"BPR", BPR}, {"N_lp", N_lp}, {"N_ip", N_ip}, {"N_hp", N_hp},
                     {"OPR", hpc.out.Pt / inlet.Pt}, {"Fn_N", Fn}, {"Fg_N", Fg}, {"F_ram_N", F_ram},
                     {"Wfuel", W_fuel}, {"TSFC_kg_per_N_s", W_fuel / Fn}, {"Tt3", hpc.out.Tt},
                     {"Tt4", burner.Tt}, {"PR_fan", fan.PR}, {"PR_ipc", ipc.PR}, {"PR_hpc", hpc.PR},
                     {"eff_fan", fan.eff}, {"eff_ipc", ipc.eff}, {"eff_hpc", hpc.eff},
                     {"PR_hpt", PR_hpt}, {"PR_ipt", PR_ipt}, {"PR_lpt", PR_lpt}, {"eff_hpt", hpt.eff},
                     {"eff_ipt", ipt.eff}, {"eff_lpt", lpt.eff}, {"A_core_m2", core.area},
                     {"A_byp_m2", byp.area}, {"core_choked", core.choked}, {"byp_choked", byp.choked},
                     {"Fg_core_N", core.Fg}, {"Fg_byp_N", byp.Fg}, {"burner_heat_rejection_W", Q_rej},
                     {"R_fan", R_fan}, {"R_ipc", R_ipc}, {"R_hpc", R_hpc}, {"Ts0", Ts}, {"Ps0", Ps},
                     {"SMN_fan", fan.smn}, {"SMN_ipc", ipc.smn}, {"SMN_hpc", hpc.smn},
                     {"handling_bleed_frac", beta}, {"handling_bleed_W", hbleed.W}};
        Flow dumped = byp_in;
        dumped.W = byp_in.W - byp_bld.W;
        const double W_out = d_ln.W + d_b.W + dumped.W;
        o.mass_closure = std::abs(W + W_fuel - W_out) / (W + W_fuel);
        const double E_in = W * fc.ht + W_fuel * h_fuel;
        const double E_out = d_ln.W * d_ln.ht + d_b.W * d_b.ht + dumped.W * dumped.ht + s.HPX_W +
                             (lp_net + ip_net + hp_net) + Q_rej;
        o.energy_closure = std::abs(E_in - E_out) /
                           (std::abs(W * fc.ht) + std::abs(W_fuel * h_fuel) + std::abs(Q_rej) + 1.0);
        const size_t ne = gas_->nElements();
        std::vector<double> bal(ne, 0.0), scale_e(ne, 0.0);
        auto elements = [&](const Flow& f, std::vector<double>& acc, double sign) {
            set_Y(f.Y);
            for (size_t e = 0; e < ne; ++e) acc[e] += sign * f.W * gas_->elementalMassFraction(e);
        };
        elements(fc, bal, 1.0);
        elements(fc, scale_e, 1.0);
        for (size_t e = 0; e < ne; ++e) {
            const double fuel_e = mode_ == ThermoMode::Matched
                ? W_fuel * fuel_element_moles_per_kg_[e] * gas_->atomicWeight(e) / 1000.0
                : (set_Y(Y_fuel_), W_fuel * gas_->elementalMassFraction(e));
            bal[e] += fuel_e;
            scale_e[e] += fuel_e;
        }
        for (const Flow* f : std::initializer_list<const Flow*>{&d_ln, &d_b, &dumped}) elements(*f, bal, -1.0);
        o.element_closure = 0.0;
        for (size_t e = 0; e < ne; ++e) {
            if (scale_e[e] > 1e-12) o.element_closure = std::max(o.element_closure, std::abs(bal[e]) / scale_e[e]);
        }
    }
    return r;
}

SolveResult Hbtf::newton(std::vector<double> x, const std::vector<double>& lo,
                         const std::vector<double>& hi, const std::vector<double>& scale, bool design_mode)
{
    SolveResult res;
    const size_t n = x.size();
    auto eval = [&](const std::vector<double>& v) {
        try {
            return dispatch(v, design_mode, nullptr);
        } catch (const std::exception&) {
            return std::vector<double>(n, std::numeric_limits<double>::infinity());
        }
    };
    std::vector<double> r = eval(x);
    for (int it = 0; it < 50; ++it) {
        res.norm_history.push_back(norm_inf(r));
        if (norm_inf(r) < 1e-10) {
            res.converged = true;
            break;
        }
        if (!std::isfinite(norm_inf(r))) {
            res.reason = "evaluation failed at the current iterate";
            break;
        }
        std::vector<std::vector<double>> J(n, std::vector<double>(n));
        for (size_t j = 0; j < n; ++j) {
            const double h = 1e-6 * scale[j];
            std::vector<double> xp = x, xm = x;
            xp[j] += h;
            xm[j] -= h;
            const auto rp = eval(xp), rm = eval(xm);
            for (size_t i = 0; i < n; ++i) J[i][j] = (rp[i] - rm[i]) / (2.0 * h);
        }
        std::vector<double> dx(n);
        for (size_t i = 0; i < n; ++i) dx[i] = -r[i];
        try {
            solve_linear(J, dx);
        } catch (const std::exception& e) {
            res.reason = e.what();
            break;
        }
        double alpha = 1.0;
        bool accepted = false;
        std::vector<double> x_new(n), r_new;
        for (int ls = 0; ls <= 6; ++ls) {
            for (size_t i = 0; i < n; ++i) x_new[i] = std::clamp(x[i] + alpha * dx[i], lo[i], hi[i]);
            r_new = eval(x_new);
            if (norm2(r_new) <= (1.0 - 1e-4 * alpha) * norm2(r)) {
                accepted = true;
                break;
            }
            alpha *= 0.75;
        }
        if (!accepted) {
            res.reason = "line search failed";
            break;
        }
        x = x_new;
        r = r_new;
        res.iterations = it + 1;
    }
    res.x = x;
    res.residuals = r;
    if (res.converged) dispatch(x, design_mode, &res.out);
    else if (res.reason.empty()) res.reason = "iteration limit";
    return res;
}

SolveResult Hbtf::solve_design(std::vector<double> guess)
{
    const double W_max = spec.design_W_max_kg_s;
    if (!std::isfinite(W_max) || W_max < 0.0)
        throw std::invalid_argument("design_W_max_kg_s must be finite and positive, or zero for the default");
    if (W_max > 0.0 && W_max < 10 * LBM)
        throw std::invalid_argument("design_W_max_kg_s is below the design-flow lower bound");
    if (spec.three_shaft) {
        if (guess.size() != 5) throw std::invalid_argument("design unknowns: W, FAR, PR_hpt, PR_ipt, PR_lpt");
        const std::vector<double> lo{10 * LBM, 1e-4, 1.001, 1.001, 1.001},
                                  hi{W_max == 0.0 ? 3000 * LBM : W_max, 0.06, 8.0, 8.0, 12.0};
        SolveResult r = newton(guess, lo, hi, {guess[0], 0.01, 1.0, 1.0, 1.0}, true);
        design.valid = r.converged;
        return r;
    }
    if (guess.size() != 4) throw std::invalid_argument("design unknowns: W, FAR, PR_hpt, PR_lpt");
    const std::vector<double> lo{10 * LBM, 1e-4, 1.001, 1.001},
                              hi{W_max == 0.0 ? 1000 * LBM : W_max, 0.06, 8.0, 8.0};
    SolveResult r = newton(guess, lo, hi, {guess[0], 0.01, 1.0, 1.0}, true);
    design.valid = r.converged;
    return r;
}

SolveResult Hbtf::solve_offdesign(std::vector<double> guess, double alt_m, double MN, double dTs_K,
                                  const std::string& throttle, double target)
{
    if (!design.valid) throw std::invalid_argument("solve the design point first");
    if (throttle != "T4" && throttle != "Fn") throw std::invalid_argument("throttle is T4 or Fn");
    if (spec.three_shaft) {
        const size_t n_unk = spec.ipc_bleed_active ? 13 : 12;
        if (guess.size() != n_unk) throw std::invalid_argument("three-shaft off-design: 12 unknowns (13 with handling bleed)");
        alt_ = alt_m;
        MN_ = MN;
        dTs_ = dTs_K;
        throttle_ = throttle;
        throttle_target_ = target;
        const auto& s = spec;
        std::vector<double> lo{1 * LBM, 1e-4, 2.0, 0.2 * s.N_lp_des, 0.2 * s.N_ip_des, 0.2 * s.N_hp_des,
                               s.fan.map.rline_stall, s.ipc.map.rline_stall, s.hpc.map.rline_stall,
                               1.001, 1.001, 1.001};
        std::vector<double> hi{3000 * LBM, 0.06, 20.0, 1.3 * s.N_lp_des, 1.3 * s.N_ip_des,
                               1.3 * s.N_hp_des, 3.0, 3.0, 3.0, 8.0, 8.0, 12.0};
        std::vector<double> scale{design.W_des, 0.01, 1.0, s.N_lp_des, s.N_ip_des, s.N_hp_des,
                                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        if (spec.ipc_bleed_active) {
            lo.push_back(0.0);
            hi.push_back(0.5);
            scale.push_back(0.1);
        }
        return newton(guess, lo, hi, scale, false);
    }
    if (guess.size() != 10) throw std::invalid_argument("off-design needs 10 unknowns");
    alt_ = alt_m;
    MN_ = MN;
    dTs_ = dTs_K;
    throttle_ = throttle;
    throttle_target_ = target;
    const double Wd = design.W_des;
    const std::vector<double> lo{10 * LBM, 1e-4, 2.0, 0.3 * spec.N_lp_des, 0.3 * spec.N_hp_des,
                                 spec.fan.map.rline_stall, spec.lpc.map.rline_stall,
                                 spec.hpc.map.rline_stall, 1.001, 1.001};
    const std::vector<double> hi{1000 * LBM, 0.06, 10.0, 1.3 * spec.N_lp_des, 1.3 * spec.N_hp_des,
                                 3.0, 3.0, 3.0, 8.0, 8.0};
    const std::vector<double> scale{Wd, 0.01, 1.0, spec.N_lp_des, spec.N_hp_des, 1.0, 1.0, 1.0, 1.0, 1.0};
    return newton(guess, lo, hi, scale, false);
}

}  // namespace catjet
