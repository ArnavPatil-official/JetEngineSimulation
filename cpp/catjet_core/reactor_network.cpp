#include "reactor_network.hpp"

#include "cantera/zerodim.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>

namespace catjet {

void gauss_hermite(int K, std::vector<double>& x, std::vector<double>& w)
{
    // Physicists' Gauss-Hermite rule; weights divided by sqrt(pi) so they sum to 1.
    static const double s = std::sqrt(M_PI);
    if (K == 7) {
        x = {-2.651961356835233, -1.673551628767471, -0.8162878828589647, 0.0,
             0.8162878828589647, 1.673551628767471, 2.651961356835233};
        w = {0.0009717812450995192, 0.05451558281912703, 0.4256072526101278, 0.8102646175568073,
             0.4256072526101278, 0.05451558281912703, 0.0009717812450995192};
    } else if (K == 9) {
        x = {-3.190993201781528, -2.266580584531843, -1.468553289216668, -0.7235510187528376, 0.0,
             0.7235510187528376, 1.468553289216668, 2.266580584531843, 3.190993201781528};
        w = {3.960697726326438e-05, 0.004943624275536947, 0.08847452739437657, 0.4326515590025558,
             0.7202352156060510, 0.4326515590025558, 0.08847452739437657, 0.004943624275536947,
             3.960697726326438e-05};
    } else {
        throw std::invalid_argument("registered Gauss-Hermite orders are 7 and 9");
    }
    for (double& v : w) v /= s;
}

ReactorNetwork::ReactorNetwork(const std::string& mechanism, const std::string& fuel, int n_solutions)
    : mechanism_(mechanism), fuel_(fuel.find(':') == std::string::npos ? fuel + ":1" : fuel)
{
    if (n_solutions < 1) throw std::invalid_argument("need at least one Solution");
    for (int i = 0; i < n_solutions; ++i) sols_.push_back(Cantera::newSolution(mechanism));
    auto& th = *sols_[0]->thermo();
    const size_t nk = th.nSpecies();
    // Fuel: a single species or a molar composition (e.g. a CRECK surrogate).
    th.setState_TPX(300.0, Cantera::OneAtm, fuel_);
    fuel_Y_.resize(nk);
    th.getMassFractions(fuel_Y_.data());
    lhv_.assign(nk, 0.0);
    counted_.assign(nk, false);
    uhc_.assign(nk, false);
    const size_t iC = th.elementIndex("C", false), iH = th.elementIndex("H", false);
    for (size_t k = 0; k < nk; ++k) {
        const std::string name = th.speciesName(k);
        const double c = iC == Cantera::npos ? 0.0 : th.nAtoms(k, iC);
        const double h = iH == Cantera::npos ? 0.0 : th.nAtoms(k, iH);
        if ((c > 0 || h > 0) && name != "CO2" && name != "H2O") {
            counted_[k] = true;
            lhv_[k] = lhv_mass(name);
        }
        uhc_[k] = c > 0 && h > 0;
    }
}

size_t ReactorNetwork::n_species() const { return sols_[0]->thermo()->nSpecies(); }

std::vector<std::string> ReactorNetwork::species_names() const
{
    return sols_[0]->thermo()->speciesNames();
}

double ReactorNetwork::lhv_mass(const std::string& species)
{
    // C_c H_h O_o N_n + (c + h/4 - o/2) O2 -> c CO2 + h/2 H2O(g) + n/2 N2 at 298.15 K
    auto& th = *sols_[0]->thermo();
    std::vector<double> state(th.stateSize());
    th.saveState(state);
    th.setState_TP(298.15, Cantera::OneAtm);
    std::vector<double> h_rt(th.nSpecies());
    th.getEnthalpy_RT(h_rt.data());
    const double RT = Cantera::GasConstant * 298.15;
    auto H = [&](const char* n) { return h_rt[th.speciesIndex(n)] * RT; };  // J/kmol
    const size_t k = th.speciesIndex(species);
    auto atoms = [&](const char* e) {
        const size_t m = th.elementIndex(e, false);
        return m == Cantera::npos ? 0.0 : th.nAtoms(k, m);
    };
    const double c = atoms("C"), h = atoms("H"), o = atoms("O"), n = atoms("N");
    const double dH = h_rt[k] * RT + (c + h / 4.0 - o / 2.0) * H("O2")
                      - c * H("CO2") - h / 2.0 * H("H2O") - n / 2.0 * H("N2");
    const double M = th.molecularWeight(k);
    th.restoreState(state);
    return dH / M;  // J/kg
}

GasState ReactorNetwork::air(double T, double P)
{
    auto& th = *sols_[0]->thermo();
    th.setState_TPX(T, P, "O2:1, N2:3.76");
    GasState s{T, P, std::vector<double>(th.nSpecies())};
    th.getMassFractions(s.Y.data());
    return s;
}

GasState ReactorNetwork::fuel_gas(double T, double P)
{
    auto& th = *sols_[0]->thermo();
    th.setState_TPX(T, P, fuel_);
    GasState s{T, P, std::vector<double>(th.nSpecies())};
    th.getMassFractions(s.Y.data());
    return s;
}

double ReactorNetwork::h_of(Cantera::ThermoPhase& th, const GasState& s)
{
    th.setMassFractions(s.Y.data());
    th.setState_TP(s.T, s.P);
    return th.enthalpy_mass();
}

GasState ReactorNetwork::mix(Cantera::ThermoPhase& th,
                             const std::vector<std::pair<GasState, double>>& streams,
                             double P, double& energy_rel, double& element_rel)
{
    // P8.2 rule: mass-weighted Y and h at common P; T from HP at 1e-13.
    const size_t nk = th.nSpecies(), ne = th.nElements();
    double m = 0.0, H = 0.0, H_abs = 0.0;
    std::vector<double> Y(nk, 0.0), el_in(ne, 0.0);
    for (const auto& [s, mdot] : streams) {
        const double h = h_of(th, s);
        m += mdot;
        H += mdot * h;
        H_abs += std::abs(mdot * h);
        for (size_t k = 0; k < nk; ++k) Y[k] += mdot * s.Y[k];
        for (size_t e = 0; e < ne; ++e) el_in[e] += mdot * th.elementalMassFraction(e);
    }
    for (double& y : Y) y /= m;
    th.setMassFractions(Y.data());
    th.setState_TP(300.0, P);
    th.setState_HP(H / m, P, 1e-13);
    GasState out{th.temperature(), P, std::vector<double>(nk)};
    th.getMassFractions(out.Y.data());
    energy_rel = std::abs(m * th.enthalpy_mass() - H) / std::max(H_abs, 1.0);
    element_rel = 0.0;
    for (size_t e = 0; e < ne; ++e) {
        element_rel = std::max(element_rel, std::abs(m * th.elementalMassFraction(e) - el_in[e]) /
                                                std::max(std::abs(el_in[e]), 1e-12));
    }
    return out;
}

GasState ReactorNetwork::feed(Cantera::ThermoPhase& th, double T3, double P, double m_air, double m_fuel)
{
    // Air + fuel gas at T3 whose enthalpy carries the liquid-basis deficit.
    GasState a = air(T3, P), f = fuel_gas(T3, P);
    const double h_a = h_of(th, a), h_f = h_of(th, f) - kVaporization;
    const double m = m_air + m_fuel;
    std::vector<double> Y(th.nSpecies());
    for (size_t k = 0; k < Y.size(); ++k) Y[k] = (m_air * a.Y[k] + m_fuel * f.Y[k]) / m;
    th.setMassFractions(Y.data());
    th.setState_TP(T3, P);
    th.setState_HP((m_air * h_a + m_fuel * h_f) / m, P, 1e-13);
    GasState out{th.temperature(), P, Y};
    th.getMassFractions(out.Y.data());
    return out;
}

GasState ReactorNetwork::equilibrium(double T3, double P, double m_air, double m_fuel)
{
    auto& th = *sols_[0]->thermo();
    GasState in = feed(th, T3, P, m_air, m_fuel);
    th.setMassFractions(in.Y.data());
    th.setState_TP(in.T, P);
    th.equilibrate("HP", "auto", 1e-9, 1000, 100, 0, 0);
    GasState out{th.temperature(), P, std::vector<double>(th.nSpecies())};
    th.getMassFractions(out.Y.data());
    return out;
}

PsrState ReactorNetwork::psr(std::shared_ptr<Cantera::Solution> sol, const GasState& inlet,
                             double mass_flow, double volume)
{
    using namespace Cantera;
    PsrState r;
    r.inlet = inlet;
    r.mass_flow = mass_flow;
    r.volume = volume;
    auto& th = *sol->thermo();
    th.setMassFractions(inlet.Y.data());
    th.setState_TP(inlet.T, inlet.P);
    auto source = newReservoir(sol, true, "inlet");
    th.equilibrate("HP", "auto", 1e-9, 1000, 100, 0, 0);  // burning-branch start, h = h_in
    auto reactor = newReactor4("ConstPressureReactor", sol, true, "psr");
    reactor->setInitialVolume(volume);
    auto exhaust = newReservoir(sol, true, "exhaust");
    auto mfc = newFlowDevice("MassFlowController", source, reactor, "mfc");
    std::dynamic_pointer_cast<MassFlowController>(mfc)->setMassFlowRate(mass_flow);
    auto pc = newFlowDevice("PressureController", reactor, exhaust, "pc");
    auto pcc = std::dynamic_pointer_cast<PressureController>(pc);
    pcc->setPrimary(mfc);
    pcc->setPressureCoeff(0.01);
    std::vector<shared_ptr<ReactorBase>> reactors{reactor};
    ReactorNet net(reactors);
    const double rtol = 1e-9;
    net.setTolerances(rtol, 1e-20);
    net.initialize();
    // Port of Cantera's Python ReactorNet.advance_to_steady_state with its
    // defaults: residual atol = rtol, threshold = 10 rtol, 10 steps per check.
    const double atol = rtol, threshold = 10.0 * rtol;
    const int max_steps = 10000;
    const size_t n = net.neq();
    std::vector<double> state(n), previous(n), max_state(n);
    net.getState(max_state.data());
    for (int step = 0; step < max_steps; ++step) {
        net.getState(previous.data());
        for (int i = 0; i < 10; ++i) net.step();
        net.getState(state.data());
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            max_state[i] = std::max(max_state[i], state[i]);
            const double d = (state[i] - previous[i]) / (max_state[i] + atol);
            sum += d * d;
        }
        r.final_residual = std::sqrt(sum) / std::sqrt(static_cast<double>(n));
        r.steady_iterations = step + 1;
        if (r.final_residual < threshold) {
            r.converged = true;
            break;
        }
    }
    reactor->restoreState();
    auto& out = *reactor->phase()->thermo();
    r.outlet = {out.temperature(), out.pressure(), std::vector<double>(out.nSpecies())};
    out.getMassFractions(r.outlet.Y.data());
    r.residence_time = out.density() * volume / mass_flow;
    r.extinguished = r.outlet.T < inlet.T + 50.0;
    return r;
}

NetworkDesign ReactorNetwork::design(double T3, double P3, double m_air, double m_fuel,
                                     const NetworkParams& p, double pressure_loss)
{
    NetworkDesign d;
    auto& th = *sols_[0]->thermo();
    th.setState_TP(300.0, Cantera::OneAtm);
    th.setEquivalenceRatio(1.0, fuel_, "O2:1, N2:3.76");
    std::vector<double> Y(th.nSpecies());
    th.getMassFractions(Y.data());
    double y_f = 0.0;
    for (size_t k = 0; k < Y.size(); ++k) if (fuel_Y_[k] > 0.0) y_f += Y[k];
    d.far_st = y_f / (1.0 - y_f);
    const double phi_global = m_fuel / m_air / d.far_st;
    d.alpha_pz = phi_global / p.phi_pz_design;
    const double P = P3 * (1.0 - pressure_loss);
    GasState eq = equilibrium(T3, P, m_air, m_fuel);
    th.setMassFractions(eq.Y.data());
    th.setState_TP(eq.T, P);
    d.rho_mean = th.density();
    d.V_ref = kTauRef * (m_air + m_fuel) / d.rho_mean;
    return d;
}

NetworkResult ReactorNetwork::run(double T3, double P3, double m_air, double m_fuel,
                                  const NetworkParams& p, const NetworkDesign& d,
                                  double pressure_loss)
{
    NetworkResult res;
    if (!(m_air > 0 && m_fuel > 0 && T3 > 0 && P3 > 0)) throw std::invalid_argument("invalid inlet");
    std::vector<double> x, w;
    gauss_hermite(p.K, x, w);
    if (static_cast<int>(sols_.size()) < p.K) throw std::invalid_argument("too few Solutions for K");
    const double P = P3 * (1.0 - pressure_loss);
    const double phi_global = m_fuel / m_air / d.far_st;
    res.alpha_pz = d.alpha_pz;
    res.phi_pz = phi_global / d.alpha_pz;
    res.alpha_qq = p.no_dilution ? 1.0 - d.alpha_pz : p.alpha_qq;
    res.alpha_dil = 1.0 - d.alpha_pz - res.alpha_qq;
    if (!(d.alpha_pz > 0 && d.alpha_pz < 1 && res.alpha_qq > 0 && res.alpha_dil >= -1e-15)) {
        throw std::invalid_argument("air splits outside (0,1) or negative dilution");
    }
    res.alpha_dil = std::max(res.alpha_dil, 0.0);
    const double sigma = p.sigma_rel * res.phi_pz;
    const double V = p.volume_scale * d.V_ref;
    auto& th = *sols_[0]->thermo();

    // Primary zone: K PSRs on separate threads, each with its own Solution.
    res.phi_k.resize(p.K);
    res.primary.resize(p.K);
    std::vector<GasState> inlets(p.K);
    std::vector<double> mdot(p.K);
    for (int k = 0; k < p.K; ++k) {
        res.phi_k[k] = res.phi_pz + std::sqrt(2.0) * sigma * x[k];
        if (!(res.phi_k[k] > 0.0)) throw std::invalid_argument("non-positive primary phi_k");
        const double a_k = w[k] * d.alpha_pz * m_air;
        const double f_k = res.phi_k[k] * d.far_st * a_k;
        inlets[k] = feed(th, T3, P, a_k, f_k);
        mdot[k] = a_k + f_k;
    }
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors(p.K);
    for (int k = 0; k < p.K; ++k) {
        threads.emplace_back([&, k] {
            try {
                res.primary[k] = psr(sols_[k], inlets[k], mdot[k], w[k] * kFracPZ * V);
            } catch (...) {
                errors[k] = std::current_exception();
            }
        });
    }
    for (auto& t : threads) t.join();
    for (auto& e : errors) if (e) std::rethrow_exception(e);

    double e_rel = 0.0, el_rel = 0.0;
    auto audit = [&] {
        res.max_mixer_energy_relative = std::max(res.max_mixer_energy_relative, e_rel);
        res.max_mixer_element_relative = std::max(res.max_mixer_element_relative, el_rel);
    };
    std::vector<std::pair<GasState, double>> pz;
    double m = 0.0;
    for (int k = 0; k < p.K; ++k) {
        pz.emplace_back(res.primary[k].outlet, mdot[k]);
        m += mdot[k];
    }
    // Quick quench: PZ mixture + alpha_qq air, one PSR.
    pz.emplace_back(air(T3, P), res.alpha_qq * m_air);
    GasState qq_in = mix(th, pz, P, e_rel, el_rel);
    audit();
    m += res.alpha_qq * m_air;
    res.quench = psr(sols_[0], qq_in, m, kFracQQ * V);
    // Lean zone: n_lean equal PSRs in series.
    GasState state = res.quench.outlet;
    for (int i = 0; i < p.n_lean; ++i) {
        res.lean.push_back(psr(sols_[0], state, m, kFracLean * V / p.n_lean));
        state = res.lean.back().outlet;
    }
    res.lean_exit = state;
    // Dilution: frozen adiabatic mixing.
    if (res.alpha_dil > 0.0) {
        state = mix(th, {{state, m}, {air(T3, P), res.alpha_dil * m_air}}, P, e_rel, el_rel);
        audit();
        m += res.alpha_dil * m_air;
    }
    res.exit = state;
    res.mass_flow_exit = m;

    // Whole-network closure against the external inflows.
    GasState a = air(T3, P), f = fuel_gas(T3, P);
    const double H_in = m_air * h_of(th, a) + m_fuel * (h_of(th, f) - kVaporization);
    const double H_abs = std::abs(m_air * h_of(th, a)) + std::abs(m_fuel * (h_of(th, f) - kVaporization));
    const double h_exit = h_of(th, res.exit);
    res.energy_relative = std::abs(m * h_exit - H_in) / std::max(H_abs, 1.0);
    std::vector<double> el_exit(th.nElements());
    for (size_t e = 0; e < el_exit.size(); ++e) el_exit[e] = m * th.elementalMassFraction(e);
    h_of(th, a);
    std::vector<double> el_air(th.nElements());
    for (size_t e = 0; e < el_air.size(); ++e) el_air[e] = th.elementalMassFraction(e);
    h_of(th, f);
    for (size_t e = 0; e < el_exit.size(); ++e) {
        const double in = m_air * el_air[e] + m_fuel * th.elementalMassFraction(e);
        res.element_relative = std::max(res.element_relative,
                                        std::abs(el_exit[e] - in) / std::max(std::abs(in), 1e-12));
    }

    // Emissions and combustion efficiency (registered energy basis).
    const size_t iNO = th.speciesIndex("NO", false), iNO2 = th.speciesIndex("NO2", false);
    const size_t iCO = th.speciesIndex("CO", false);
    double unburned = 0.0, uhc = 0.0;
    for (size_t k = 0; k < res.exit.Y.size(); ++k) {
        if (counted_[k]) unburned += m * res.exit.Y[k] * lhv_[k];
        if (uhc_[k]) uhc += m * res.exit.Y[k];
    }
    double lhv_f = -kVaporization;
    for (size_t k = 0; k < fuel_Y_.size(); ++k) {
        if (fuel_Y_[k] > 0.0) lhv_f += fuel_Y_[k] * lhv_mass(th.speciesName(k));
    }
    res.eta_b = 1.0 - unburned / (m_fuel * lhv_f);
    double nox = 0.0;
    if (iNO != Cantera::npos) nox += res.exit.Y[iNO] * th.molecularWeight(iNO2) / th.molecularWeight(iNO);
    if (iNO2 != Cantera::npos) nox += res.exit.Y[iNO2];
    res.EI_NOx_g_kg = 1000.0 * nox * m / m_fuel;
    res.EI_CO_g_kg = iCO == Cantera::npos ? 0.0 : 1000.0 * res.exit.Y[iCO] * m / m_fuel;
    res.EI_UHC_g_kg = 1000.0 * uhc / m_fuel;
    for (const auto& z : res.primary) {
        res.all_converged = res.all_converged && z.converged;
        res.any_extinguished = res.any_extinguished || z.extinguished;
    }
    // Extinction (outlet within 50 K of inlet) is meaningful only for the
    // primary PSRs; downstream zones are fed hot gas. Their flags are kept.
    res.all_converged = res.all_converged && res.quench.converged;
    for (const auto& z : res.lean) res.all_converged = res.all_converged && z.converged;
    return res;
}

}  // namespace catjet
