// pybind11 module catjet_core (P8.1). Results are returned as dicts shaped like
// the Python v6 results so simulation/catjet_backend.py can stand in for
// IntegratedTurbofanEngine in lto_v5.solve_task. Built with the static-Cantera,
// hidden-symbol recipe (cpp/CMakeLists.txt) so it coexists with the pip wheel.
#include "../catjet_core/v6_engine.hpp"

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
