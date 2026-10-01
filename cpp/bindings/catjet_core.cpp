// pybind11 module catjet_core (P8.1). Results are returned as dicts shaped like
// the Python v6 results so simulation/catjet_backend.py can stand in for
// IntegratedTurbofanEngine in lto_v5.solve_task. Built with the static-Cantera,
// hidden-symbol recipe (cpp/CMakeLists.txt) so it coexists with the pip wheel.
#include "../catjet_core/v6_engine.hpp"
#include "../catjet_core/enthalpy_turbine.hpp"
#include "../catjet_core/choking_nozzle.hpp"
#include "../catjet_core/thrust_match.hpp"
#include "../catjet_core/reactor_network.hpp"
#include "../catjet_core/offdesign.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace catjet;

namespace {

py::dict to_dict(const CompressorResult& c)
{
    py::dict d;
    d["T_out"] = c.T_out; d["p_out"] = c.p_out; d["h_in"] = c.h_in; d["h_out"] = c.h_out;
    d["work_specific"] = c.work_specific;
    return d;
}

py::dict to_dict(const FanResult& f)
{
    py::dict d;
    d["T_exit"] = f.T_exit; d["p_exit"] = f.p_exit; d["dT"] = f.dT; d["work_total"] = f.work_total;
    d["u_bypass_exit"] = f.u_bypass_exit; d["thrust_bypass"] = f.thrust_bypass;
    return d;
}

py::dict to_dict(const CombustorResult& c)
{
    py::dict d;
    d["T_out"] = c.T_out; d["p_out"] = c.p_out; d["h_out"] = c.h_out; d["cp_out"] = c.cp_out;
    d["R_out"] = c.R_out; d["gamma_out"] = c.gamma_out; d["Y_out"] = c.Y_out;
    return d;
}

py::dict to_dict(const TurbineResult& t)
{
    py::dict d;
    d["rho"] = t.rho; d["u"] = t.u; d["p"] = t.p; d["T"] = t.T; d["work_specific"] = t.work_specific;
    d["work_total"] = t.work_total; d["cp"] = t.cp; d["R"] = t.R; d["gamma"] = t.gamma;
    return d;
}

py::dict to_dict(const NozzleResult& n)
{
    py::dict d;
    d["rho"] = n.rho; d["u"] = n.u; d["p"] = n.p; d["T"] = n.T; d["thrust_total"] = n.thrust_total;
    d["thrust_momentum"] = n.thrust_momentum; d["thrust_pressure"] = n.thrust_pressure;
    d["thrust_model"] = "static_test_stand"; d["A_exit_effective"] = n.A_exit_effective;
    return d;
}

py::dict to_dict(const CycleResult& r)
{
    py::dict d;
    d["compressor"] = to_dict(r.compressor);
    d["combustor"] = to_dict(r.combustor);
    d["turbine"] = to_dict(r.turbine);
    d["nozzle"] = to_dict(r.nozzle);
    d["fan"] = r.has_fan ? py::object(to_dict(r.fan)) : py::none();
    py::dict p;
    p["thrust_N"] = r.thrust_N; p["thrust_kN"] = r.thrust_kN; p["thrust_core_kN"] = r.thrust_core_kN;
    p["thrust_bypass_kN"] = r.thrust_bypass_kN; p["tsfc_SI"] = r.tsfc_SI; p["tsfc_mg_per_Ns"] = r.tsfc_mg_per_Ns;
    p["thermal_efficiency"] = r.thermal_efficiency; p["fuel_mass_flow"] = r.fuel_mass_flow;
    p["total_mass_flow"] = r.total_mass_flow; p["bypass_mass_flow"] = r.bypass_mass_flow;
    p["total_air_mass_flow"] = r.total_air_mass_flow; p["specific_thrust_Ns_kg"] = r.specific_thrust_Ns_kg;
    p["fan_work_W"] = r.fan_work_W; p["fuel_air_ratio"] = r.fuel_air_ratio;
    d["performance"] = p;
    py::dict e;
    e["NOx_g_s"] = r.nox_g_s;
    d["emissions"] = e;
    return d;
}

py::dict to_dict(const GasState& s)
{
    py::dict d;
    d["T"] = s.T; d["P"] = s.P; d["Y"] = s.Y;
    return d;
}

py::dict to_dict(const MassStream& s)
{
    py::dict d = to_dict(s.state);
    d["mass_flow"] = s.mass_flow;
    return d;
}

py::dict to_dict(const GasProperties& p)
{
    py::dict d;
    d["h"] = p.h; d["s"] = p.s; d["cp"] = p.cp; d["R"] = p.R;
    d["gamma"] = p.gamma; d["rho"] = p.rho; d["elements"] = p.elements;
    return d;
}

py::dict to_dict(const MixResult& m)
{
    py::dict d = to_dict(m.stream);
    d["mass_relative"] = m.mass_relative;
    d["energy_relative"] = m.energy_relative;
    d["element_relative"] = m.element_relative;
    return d;
}

py::dict to_dict(const ExpansionResult& e)
{
    py::dict d = to_dict(e.stream);
    d["requested_work"] = e.requested_work;
    d["actual_work"] = e.actual_work;
    d["energy_relative"] = e.energy_relative;
    d["element_relative"] = e.element_relative;
    d["steps"] = e.steps;
    return d;
}

py::dict to_dict(const P82CycleResult& r)
{
    py::dict d = to_dict(r.cycle);
    py::dict stations, stages;
    for (const auto& kv : r.stations) stations[py::str(kv.first)] = to_dict(kv.second);
    for (const auto& kv : r.stages) stages[py::str(kv.first)] = to_dict(kv.second);
    d["stations"] = stations;
    d["stages"] = stages;
    d["burner_heat_rejection_W"] = r.burner_heat_rejection_W;
    d["max_mass_relative"] = r.max_mass_relative;
    d["max_energy_relative"] = r.max_energy_relative;
    d["max_element_relative"] = r.max_element_relative;
    return d;
}

py::dict to_dict(const ChokingNozzleResult& n)
{
    py::dict d;
    d["exit"] = to_dict(n.exit);
    d["critical_pressure"] = n.critical_pressure; d["critical_mass_flux"] = n.critical_mass_flux;
    d["area"] = n.area; d["Cd"] = n.Cd; d["Cv"] = n.Cv; d["mass_flow"] = n.mass_flow;
    d["velocity_ideal"] = n.velocity_ideal; d["velocity"] = n.velocity;
    d["thrust_momentum"] = n.thrust_momentum; d["thrust_pressure"] = n.thrust_pressure;
    d["thrust_total"] = n.thrust_total; d["energy_relative"] = n.energy_relative;
    d["choked"] = n.choked;
    return d;
}

py::dict to_dict(const PsrState& z)
{
    py::dict d;
    d["inlet"] = to_dict(z.inlet); d["outlet"] = to_dict(z.outlet);
    d["mass_flow"] = z.mass_flow; d["volume"] = z.volume; d["residence_time"] = z.residence_time;
    d["steady_iterations"] = z.steady_iterations; d["final_residual"] = z.final_residual;
    d["converged"] = z.converged; d["extinguished"] = z.extinguished;
    d["error"] = z.error;
    return d;
}

py::dict to_dict(const NetworkResult& r)
{
    py::dict d;
    d["exit"] = to_dict(r.exit); d["lean_exit"] = to_dict(r.lean_exit);
    d["mass_flow_exit"] = r.mass_flow_exit; d["eta_b"] = r.eta_b;
    d["EI_NOx_g_kg"] = r.EI_NOx_g_kg; d["EI_CO_g_kg"] = r.EI_CO_g_kg; d["EI_UHC_g_kg"] = r.EI_UHC_g_kg;
    d["phi_pz"] = r.phi_pz; d["alpha_pz"] = r.alpha_pz; d["alpha_qq"] = r.alpha_qq;
    d["alpha_dil"] = r.alpha_dil; d["phi_k"] = r.phi_k;
    py::list primary, lean;
    for (const auto& z : r.primary) primary.append(to_dict(z));
    for (const auto& z : r.lean) lean.append(to_dict(z));
    d["primary"] = primary; d["quench"] = to_dict(r.quench); d["lean"] = lean;
    d["energy_relative"] = r.energy_relative; d["element_relative"] = r.element_relative;
    d["max_mixer_energy_relative"] = r.max_mixer_energy_relative;
    d["max_mixer_element_relative"] = r.max_mixer_element_relative;
    d["all_converged"] = r.all_converged; d["any_extinguished"] = r.any_extinguished;
    return d;
}

py::dict to_dict(const Flow& f)
{
    py::dict d;
    d["W"] = f.W; d["Tt"] = f.Tt; d["Pt"] = f.Pt; d["ht"] = f.ht; d["St"] = f.St;
    return d;
}

py::dict to_dict(const SolveResult& r)
{
    py::dict d;
    d["converged"] = r.converged; d["iterations"] = r.iterations; d["x"] = r.x;
    d["residuals"] = r.residuals; d["norm_history"] = r.norm_history; d["reason"] = r.reason;
    d["scalars"] = r.out.scalars;
    py::dict st;
    for (const auto& kv : r.out.stations) st[py::str(kv.first)] = to_dict(kv.second);
    d["stations"] = st;
    d["mass_closure"] = r.out.mass_closure; d["energy_closure"] = r.out.energy_closure;
    d["element_closure"] = r.out.element_closure;
    d["extrapolated"] = r.out.extrapolated; d["extrapolated_maps"] = r.out.extrapolated_maps;
    return d;
}

CombustorResult combustor_from(const py::dict& d)
{
    CombustorResult c{};
    c.T_out = d["T"].cast<double>(); c.p_out = d["p"].cast<double>(); c.cp_out = d["cp"].cast<double>();
    c.R_out = d["R"].cast<double>(); c.gamma_out = d["gamma"].cast<double>();
    return c;
}

TurbineResult turbine_from(const py::dict& d)
{
    TurbineResult t{};
    t.T = d["T"].cast<double>(); t.p = d["p"].cast<double>(); t.cp = d["cp"].cast<double>();
    t.R = d["R"].cast<double>(); t.gamma = d["gamma"].cast<double>();
    return t;
}

}  // namespace

PYBIND11_MODULE(catjet_core, m)
{
    m.doc() = "CAT-JET C++ core (Phase 8): v6 cycle port";
    m.attr("cantera_version") = CANTERA_VERSION;
    static py::exception<CycleDoesNotClose> exc_cdnc(m, "CycleDoesNotClose", PyExc_ValueError);
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const CycleDoesNotClose& e) {
            py::set_error(exc_cdnc, e.what());
        }
    });

    py::class_<V6Config>(m, "V6Config")
        .def(py::init<>())
        .def_readwrite("mass_flow_core", &V6Config::mass_flow_core)
        .def_readwrite("bypass_ratio", &V6Config::bypass_ratio)
        .def_readwrite("fpr", &V6Config::fpr)
        .def_readwrite("eta_fan", &V6Config::eta_fan)
        .def_readwrite("pi_c", &V6Config::pi_c)
        .def_readwrite("combustor_pressure_loss", &V6Config::combustor_pressure_loss)
        .def_readwrite("combustor_heat_loss_fraction", &V6Config::combustor_heat_loss_fraction)
        .def_readwrite("combustor_air_fraction", &V6Config::combustor_air_fraction)
        .def_readwrite("A_combustor_exit", &V6Config::A_combustor_exit)
        .def_readwrite("A_nozzle_exit", &V6Config::A_nozzle_exit)
        .def_readwrite("P_ambient", &V6Config::P_ambient)
        .def_readwrite("T_ambient", &V6Config::T_ambient)
        .def_readwrite("eta_c", &V6Config::eta_c)
        .def_readwrite("eta_polytropic", &V6Config::eta_polytropic)
        .def_readwrite("nox_A", &V6Config::nox_A)
        .def_readwrite("nox_B", &V6Config::nox_B)
        .def_readwrite("nox_C", &V6Config::nox_C);

    py::class_<GasState>(m, "GasState")
        .def(py::init<>())
        .def(py::init<double, double, std::vector<double>>())
        .def_readwrite("T", &GasState::T)
        .def_readwrite("P", &GasState::P)
        .def_readwrite("Y", &GasState::Y);
    py::class_<MassStream>(m, "MassStream")
        .def(py::init<>())
        .def(py::init<GasState, double>())
        .def_readwrite("state", &MassStream::state)
        .def_readwrite("mass_flow", &MassStream::mass_flow);
    py::class_<GasThermo>(m, "GasThermo")
        .def(py::init<const std::string&>(), py::arg("mechanism"))
        .def("from_moles", &GasThermo::from_moles)
        .def("at_enthalpy", &GasThermo::at_enthalpy)
        .def("compress", &GasThermo::compress)
        .def("properties", [](GasThermo& g, const GasState& s) { return to_dict(g.properties(s)); })
        .def("mix", [](GasThermo& g, const MassStream& a, const MassStream& b, double P) {
            return to_dict(g.mix(a, b, P));
        })
        .def("expand_for_work", [](GasThermo& g, const MassStream& in, double W, double eta,
                                    int steps, double cp, double R) {
            return to_dict(g.expand_for_work(in, W, eta, steps, cp, R));
        }, py::arg("stream"), py::arg("work_W"), py::arg("eta_poly"),
           py::arg("steps") = 50, py::arg("constant_cp") = 0.0,
           py::arg("constant_R") = 0.0)
        .def_property_readonly("n_species", &GasThermo::n_species);
    py::class_<ChokingNozzle>(m, "ChokingNozzle")
        .def(py::init<const std::string&>(), py::arg("mechanism"))
        .def("critical_pressure", py::overload_cast<const GasState&>(&ChokingNozzle::critical_pressure))
        .def("mass_flux", &ChokingNozzle::mass_flux)
        .def("run", [](ChokingNozzle& n, const GasState& s, double pa, double A, double Cd, double Cv) {
            return to_dict(n.run(s, pa, A, Cd, Cv));
        }, py::arg("stagnation"), py::arg("ambient_pressure"), py::arg("area"),
           py::arg("Cd"), py::arg("Cv"))
        .def("run_dual", [](ChokingNozzle& n, const GasState& core, const GasState& bypass,
                             double pa, double Ac, double Ab, double Cdc, double Cvc,
                             double Cdb, double Cvb) {
            const DualNozzleResult r = n.run_dual(core, bypass, pa, Ac, Ab, Cdc, Cvc, Cdb, Cvb);
            py::dict d;
            d["core"] = to_dict(r.core); d["bypass"] = to_dict(r.bypass);
            d["thrust_total"] = r.thrust_total; d["mass_flow_total"] = r.mass_flow_total;
            return d;
        });
    // Test hook: the thrust_match.hpp template driven by the G0 v6 cycle must
    // reproduce V6Engine::run_at_thrust exactly (phi, evaluations, reasons).
    m.def("v6_template_run_at_thrust", [](V6Engine& e, double target, const std::string& fuel,
                                          const std::vector<std::string>& species, double eff,
                                          py::object guess) {
        const double g = guess.is_none() ? -1.0 : guess.cast<double>();
        py::dict d;
        try {
            auto solved = v6_thrust_match<CycleResult>(
                [&](double phi) { return e.run_full_cycle(fuel, species, phi, eff); },
                [](const CycleResult& r) { return r.combustor.T_out; },
                [](const CycleResult& r) { return r.thrust_kN; },
                target, eff, 0.05, 1.0, 3800.0 * 5.0 / 9.0, 1e-12, g);
            d["status"] = "converged"; d["phi"] = solved.second.phi;
            d["n_cycle_evaluations"] = solved.second.n_cycle_evaluations;
            d["fuel_mass_flow"] = solved.first.fuel_mass_flow;
        } catch (const ThrustTargetUnreachable& u) {
            d["status"] = "unreachable"; d["reason"] = u.reason;
        }
        return d;
    }, py::arg("engine"), py::arg("target_kN"), py::arg("fuel"), py::arg("fuel_species"),
       py::arg("combustor_efficiency"), py::arg("phi_guess") = py::none());
    py::class_<NetworkParams>(m, "NetworkParams")
        .def(py::init<>())
        .def_readwrite("phi_pz_design", &NetworkParams::phi_pz_design)
        .def_readwrite("sigma_rel", &NetworkParams::sigma_rel)
        .def_readwrite("alpha_qq", &NetworkParams::alpha_qq)
        .def_readwrite("volume_scale", &NetworkParams::volume_scale)
        .def_readwrite("K", &NetworkParams::K)
        .def_readwrite("n_lean", &NetworkParams::n_lean)
        .def_readwrite("no_dilution", &NetworkParams::no_dilution);
    py::class_<NetworkDesign>(m, "NetworkDesign")
        .def(py::init<>())
        .def_readwrite("alpha_pz", &NetworkDesign::alpha_pz)
        .def_readwrite("V_ref", &NetworkDesign::V_ref)
        .def_readwrite("far_st", &NetworkDesign::far_st)
        .def_readwrite("rho_mean", &NetworkDesign::rho_mean);
    m.def("gauss_hermite", [](int K) {
        std::vector<double> x, w;
        gauss_hermite(K, x, w);
        return py::make_tuple(x, w);
    });
    py::class_<ReactorNetwork>(m, "ReactorNetwork")
        .def(py::init<const std::string&, const std::string&, int>(), py::arg("mechanism"),
             py::arg("fuel") = "POSF10325", py::arg("n_solutions") = 9)
        .def("design", &ReactorNetwork::design, py::arg("T3"), py::arg("P3"), py::arg("m_air"),
             py::arg("m_fuel"), py::arg("params"), py::arg("pressure_loss") = 0.045)
        .def("run", [](ReactorNetwork& n, double T3, double P3, double ma, double mf,
                       const NetworkParams& p, const NetworkDesign& d, double dp) {
            NetworkResult r;
            {
                py::gil_scoped_release release;
                r = n.run(T3, P3, ma, mf, p, d, dp);
            }
            return to_dict(r);
        }, py::arg("T3"), py::arg("P3"), py::arg("m_air"), py::arg("m_fuel"), py::arg("params"),
           py::arg("design"), py::arg("pressure_loss") = 0.045)
        .def("equilibrium", &ReactorNetwork::equilibrium)
        .def("lhv_mass", &ReactorNetwork::lhv_mass)
        .def("species_names", &ReactorNetwork::species_names);
    // P8.4 HBTF (docs/phase8_p84_registration.md)
    py::class_<GridTable>(m, "GridTable")
        .def(py::init<std::vector<std::vector<double>>, std::vector<double>>())
        .def("__call__", [](const GridTable& t, const std::vector<double>& x) {
            bool ex = false;
            const double v = t(x, &ex);
            return py::make_tuple(v, ex);
        });
    py::class_<ComponentMap>(m, "ComponentMap")
        .def(py::init<>())
        .def_readwrite("name", &ComponentMap::name)
        .def_readwrite("params", &ComponentMap::params)
        .def_readwrite("outputs", &ComponentMap::outputs)
        .def_readwrite("defaults", &ComponentMap::defaults)
        .def_readwrite("rline_stall", &ComponentMap::rline_stall);
    py::class_<Bleed>(m, "Bleed")
        .def(py::init<>())
        .def_readwrite("name", &Bleed::name).def_readwrite("frac_W", &Bleed::frac_W)
        .def_readwrite("frac_P", &Bleed::frac_P).def_readwrite("frac_work", &Bleed::frac_work);
    py::class_<CompressorSpec>(m, "CompressorSpec")
        .def(py::init<>())
        .def_readwrite("name", &CompressorSpec::name).def_readwrite("map", &CompressorSpec::map)
        .def_readwrite("PR_des", &CompressorSpec::PR_des).def_readwrite("eff_des", &CompressorSpec::eff_des)
        .def_readwrite("bleeds", &CompressorSpec::bleeds);
    py::class_<TurbineSpec>(m, "TurbineSpec")
        .def(py::init<>())
        .def_readwrite("name", &TurbineSpec::name).def_readwrite("map", &TurbineSpec::map)
        .def_readwrite("eff_des", &TurbineSpec::eff_des);
    py::class_<HbtfSpec>(m, "HbtfSpec")
        .def(py::init<>())
        .def_readwrite("alt_m", &HbtfSpec::alt_m).def_readwrite("MN", &HbtfSpec::MN)
        .def_readwrite("dTs_K", &HbtfSpec::dTs_K).def_readwrite("Fn_des_N", &HbtfSpec::Fn_des_N)
        .def_readwrite("T4_max_K", &HbtfSpec::T4_max_K).def_readwrite("N_lp_des", &HbtfSpec::N_lp_des)
        .def_readwrite("N_hp_des", &HbtfSpec::N_hp_des).def_readwrite("BPR_des", &HbtfSpec::BPR_des)
        .def_readwrite("ram_recovery", &HbtfSpec::ram_recovery)
        .def_readwrite("dPqP_duct4", &HbtfSpec::dPqP_duct4).def_readwrite("dPqP_duct6", &HbtfSpec::dPqP_duct6)
        .def_readwrite("dPqP_burner", &HbtfSpec::dPqP_burner).def_readwrite("dPqP_duct11", &HbtfSpec::dPqP_duct11)
        .def_readwrite("dPqP_duct13", &HbtfSpec::dPqP_duct13).def_readwrite("dPqP_duct15", &HbtfSpec::dPqP_duct15)
        .def_readwrite("Cv_core", &HbtfSpec::Cv_core).def_readwrite("Cv_byp", &HbtfSpec::Cv_byp)
        .def_readwrite("frac_byp_bleed", &HbtfSpec::frac_byp_bleed)
        .def_readwrite("cool3_frac_W", &HbtfSpec::cool3_frac_W).def_readwrite("cool4_frac_W", &HbtfSpec::cool4_frac_W)
        .def_readwrite("cool3_frac_P", &HbtfSpec::cool3_frac_P).def_readwrite("cool4_frac_P", &HbtfSpec::cool4_frac_P)
        .def_readwrite("cool1_frac_P_lpt", &HbtfSpec::cool1_frac_P_lpt)
        .def_readwrite("cool2_frac_P_lpt", &HbtfSpec::cool2_frac_P_lpt)
        .def_readwrite("HPX_W", &HbtfSpec::HPX_W)
        .def_readwrite("fan", &HbtfSpec::fan).def_readwrite("lpc", &HbtfSpec::lpc)
        .def_readwrite("hpc", &HbtfSpec::hpc).def_readwrite("hpt", &HbtfSpec::hpt)
        .def_readwrite("lpt", &HbtfSpec::lpt)
        .def_readwrite("atm_alt_ft", &HbtfSpec::atm_alt_ft).def_readwrite("atm_T_R", &HbtfSpec::atm_T_R)
        .def_readwrite("atm_P_psi", &HbtfSpec::atm_P_psi)
        .def_readwrite("three_shaft", &HbtfSpec::three_shaft).def_readwrite("ipc", &HbtfSpec::ipc)
        .def_readwrite("ipt", &HbtfSpec::ipt).def_readwrite("N_ip_des", &HbtfSpec::N_ip_des)
        .def_readwrite("dPqP_duct_ipt_lpt", &HbtfSpec::dPqP_duct_ipt_lpt)
        .def_readwrite("nozzle_p83", &HbtfSpec::nozzle_p83)
        .def_readwrite("Cd_core", &HbtfSpec::Cd_core).def_readwrite("Cd_byp", &HbtfSpec::Cd_byp)
        .def_readwrite("eta_b", &HbtfSpec::eta_b)
        .def_readwrite("Ts_override_K", &HbtfSpec::Ts_override_K)
        .def_readwrite("Ps_override_Pa", &HbtfSpec::Ps_override_Pa);
    py::class_<Hbtf>(m, "Hbtf")
        .def(py::init([](const std::string& mech, const std::string& mode, const std::string& air,
                         const std::string& fuel, std::map<std::string, double> weights) {
            return std::make_unique<Hbtf>(mech, mode == "matched" ? ThermoMode::Matched : ThermoMode::Production,
                                          air, fuel, weights);
        }), py::arg("mechanism"), py::arg("mode"), py::arg("air"), py::arg("fuel"),
            py::arg("fuel_element_weights") = std::map<std::string, double>{})
        .def_readwrite("spec", &Hbtf::spec)
        .def("solve_design", [](Hbtf& h, std::vector<double> g) { return to_dict(h.solve_design(g)); })
        .def("solve_offdesign", [](Hbtf& h, std::vector<double> g, double alt, double MN, double dTs,
                                   const std::string& throttle, double target) {
            return to_dict(h.solve_offdesign(g, alt, MN, dTs, throttle, target));
        })
        .def("residuals", [](Hbtf& h, const std::vector<double>& x, bool design) {
            return h.residuals(x, design);
        })
        .def("design_areas", [](const Hbtf& h) {
            return py::make_tuple(h.design.A_core, h.design.A_byp, h.design.valid);
        })
        .def("us1976", [](const Hbtf& h, double alt_m) {
            return py::make_tuple(h.us1976_T(alt_m), h.us1976_P(alt_m));
        });
    py::class_<P82Config>(m, "P82Config")
        .def(py::init<>())
        .def_readwrite("base", &P82Config::base)
        .def_readwrite("vaporization_J_kg", &P82Config::vaporization_J_kg)
        .def_readwrite("ngv_fraction", &P82Config::ngv_fraction)
        .def_readwrite("rotor_fraction", &P82Config::rotor_fraction)
        .def_readwrite("pressure_steps", &P82Config::pressure_steps);
    py::class_<P82Engine>(m, "P82Engine")
        .def(py::init<const std::string&>(), py::arg("mechanism"))
        .def_readwrite("config", &P82Engine::config)
        .def("run_full_cycle", [](P82Engine& e, const std::string& fuel,
                                   const std::vector<std::string>& species,
                                   double phi, double eff, int level) {
            P82CycleResult r;
            {
                py::gil_scoped_release release;
                r = e.run_full_cycle(fuel, species, phi, eff, level);
            }
            return to_dict(r);
        }, py::arg("fuel"), py::arg("fuel_species"), py::arg("phi"),
           py::arg("combustor_efficiency"), py::arg("ablation_level") = 2)
        .def("run_at_thrust", [](P82Engine& e, double target, const std::string& fuel,
                                  const std::vector<std::string>& species,
                                  double eff, int level, double lo, double hi,
                                  double t4max, double xtol, py::object guess) {
            const double g = guess.is_none() ? -1.0 : guess.cast<double>();
            py::dict d;
            try {
                P82AtThrustResult r;
                {
                    py::gil_scoped_release release;
                    r = e.run_at_thrust(target, fuel, species, eff, level,
                                         lo, hi, t4max, xtol, g);
                }
                d = to_dict(r.result);
                py::dict tm;
                for (const auto& kv : r.match.info) tm[py::str(kv.first)] = kv.second;
                tm["phi"] = r.match.phi; tm["status"] = "converged";
                tm["target_kN"] = r.match.target_kN;
                tm["residual_kN"] = r.match.residual_kN;
                tm["t4_K"] = r.match.t4_K;
                tm["n_cycle_evaluations"] = r.match.n_cycle_evaluations;
                d["thrust_match"] = tm; d["status"] = "converged";
            } catch (const ThrustTargetUnreachable& u) {
                d["status"] = "unreachable"; d["reason"] = u.reason;
                d["message"] = std::string(u.what());
                py::dict inf;
                for (const auto& kv : u.info) inf[py::str(kv.first)] = kv.second;
                d["info"] = inf;
            }
            return d;
        }, py::arg("target_kN"), py::arg("fuel"), py::arg("fuel_species"),
           py::arg("combustor_efficiency"), py::arg("ablation_level") = 2,
           py::arg("phi_lo") = 0.05, py::arg("phi_hi") = 1.0,
           py::arg("t4_max_K") = 3800.0 * 5.0 / 9.0,
           py::arg("phi_xtol") = 1e-12, py::arg("phi_guess") = py::none());

    py::class_<V6Engine>(m, "V6Engine")
        .def(py::init<const std::string&>(), py::arg("mechanism"))
        .def_readwrite("config", &V6Engine::config)
        .def("run_compressor", [](V6Engine& e, double T, double p) { return to_dict(e.run_compressor(T, p)); })
        .def("run_fan", [](V6Engine& e, double T, double p, double m) { return to_dict(e.run_fan(T, p, m)); })
        .def("fuel_air_ratio", &V6Engine::fuel_air_ratio)
        .def("combustor_run", [](V6Engine& e, double T, double p, const std::string& fuel, double phi,
                                 double eff, double hl) { return to_dict(e.combustor_run(T, p, fuel, phi, eff, hl)); })
        .def("run_turbine_analytic", [](V6Engine& e, const py::dict& state, double m, double w) {
            return to_dict(e.run_turbine_analytic(combustor_from(state), m, w));
        })
        .def("run_nozzle", [](V6Engine& e, const py::dict& state, double m) {
            return to_dict(e.run_nozzle(turbine_from(state), m));
        })
        .def("estimate_nox", &V6Engine::estimate_nox)
        .def("run_full_cycle", [](V6Engine& e, const std::string& fuel, const std::vector<std::string>& species,
                                  double phi, double eff) { return to_dict(e.run_full_cycle(fuel, species, phi, eff)); })
        .def("run_at_thrust",
             [](V6Engine& e, double target, const std::string& fuel, const std::vector<std::string>& species,
                double eff, double lo, double hi, double t4max, double xtol, py::object guess) {
                 double g = guess.is_none() ? -1.0 : guess.cast<double>();
                 py::dict out;
                 try {
                     AtThrustResult r;
                     {
                         py::gil_scoped_release release;
                         r = e.run_at_thrust(target, fuel, species, eff, lo, hi, t4max, xtol, g);
                     }
                     out = to_dict(r.cycle);
                     py::dict tm;
                     for (const auto& kv : r.match.info) tm[py::str(kv.first)] = kv.second;
                     tm["phi"] = r.match.phi; tm["status"] = "converged"; tm["target_kN"] = r.match.target_kN;
                     tm["residual_kN"] = r.match.residual_kN; tm["t4_K"] = r.match.t4_K;
                     tm["n_cycle_evaluations"] = r.match.n_cycle_evaluations;
                     out["thrust_match"] = tm;
                     out["status"] = "converged";
                 } catch (const ThrustTargetUnreachable& u) {
                     out["status"] = "unreachable";
                     out["reason"] = u.reason;
                     out["message"] = std::string(u.what());
                     py::dict inf;
                     for (const auto& kv : u.info) inf[py::str(kv.first)] = kv.second;
                     out["info"] = inf;
                 }
                 return out;
             },
             py::arg("target_kN"), py::arg("fuel"), py::arg("fuel_species"), py::arg("combustor_efficiency"),
             py::arg("phi_lo") = 0.05, py::arg("phi_hi") = 1.0, py::arg("t4_max_K") = 3800.0 * 5.0 / 9.0,
             py::arg("phi_xtol") = 1e-12, py::arg("phi_guess") = py::none());
}
