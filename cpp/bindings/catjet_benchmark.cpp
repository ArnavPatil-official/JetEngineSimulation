// P8.1 benchmark-only std::thread batch runner. The v6 engine and protected
// Python model are unchanged. Each worker owns one V6Engine/Cantera state.
// Pool and engine construction occur before the benchmark warm-up.
#include "../catjet_core/v6_engine.hpp"
#include "../catjet_core/v6_variant.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <condition_variable>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace py = pybind11;
using catjet::AtThrustResult;
using catjet::ThrustTargetUnreachable;
using catjet::V6Config;
using catjet::V6Engine;
using catjet::V6VariantEngine;

namespace {

struct Job {
    V6Config cfg;
    double target_kN;
    double eta_b;
};

struct Answer {
    bool reachable = false;
    std::string reason;
    AtThrustResult solved{};
    std::exception_ptr error;
    bool probe_failed = false;
    long cycle_calls = -1;  // every cycle call incl. failed probes; -1 = not counted
};

// P8-A1.2 variant b bracket, identical to scripts/phase8/v6_optimized.py:
// a narrower cold probe solve; any unreachable probe falls back to the
// registered cold (0.05, 1.0) solve, which then decides result and reason.
template <class Engine>
AtThrustResult solve_with_probe(Engine& engine, const Job& job, const std::string& fuel,
                                const std::vector<std::string>& species, bool& probe_failed)
{
    const double pi_c = job.cfg.pi_c;
    const double probe = pi_c >= 20.0 ? 0.20 : (pi_c >= 5.0 ? 0.10 : -1.0);
    if (probe > 0.0) {
        try {
            return engine.run_at_thrust(job.target_kN, fuel, species, job.eta_b, probe, 1.0);
        } catch (const ThrustTargetUnreachable&) {
            probe_failed = true;
        }
    }
    return engine.run_at_thrust(job.target_kN, fuel, species, job.eta_b);
}

V6Config parse_config(const py::dict& d)
{
    V6Config c;
    c.mass_flow_core = d["mass_flow_core"].cast<double>();
    c.bypass_ratio = d["bypass_ratio"].cast<double>();
    c.fpr = d["fpr"].cast<double>();
    c.eta_fan = d["eta_fan"].cast<double>();
    c.pi_c = d["pi_c"].cast<double>();
    c.combustor_pressure_loss = d["combustor_pressure_loss"].cast<double>();
    c.combustor_heat_loss_fraction = d["combustor_heat_loss_fraction"].cast<double>();
    c.combustor_air_fraction = d["combustor_air_fraction"].cast<double>();
    c.A_combustor_exit = d["A_combustor_exit"].cast<double>();
    c.A_nozzle_exit = d["A_nozzle_exit"].cast<double>();
    c.P_ambient = d["P_ambient"].cast<double>();
    c.T_ambient = d["T_ambient"].cast<double>();
    c.eta_c = d["eta_c"].cast<double>();
    c.eta_polytropic = d["eta_polytropic"].cast<double>();
    c.nox_A = d["nox_A"].cast<double>();
    c.nox_B = d["nox_B"].cast<double>();
    c.nox_C = d["nox_C"].cast<double>();
    return c;
}

py::dict as_row(const Answer& answer, const Job& job)
{
    py::dict d;
    if (!answer.reachable) {
        d["status"] = "unreachable";
        d["reason"] = answer.reason;
        d["ff"] = std::numeric_limits<double>::quiet_NaN();
        d["phi"] = std::numeric_limits<double>::quiet_NaN();
        if (answer.cycle_calls >= 0) d["cycle_calls"] = answer.cycle_calls;
        return d;
    }
    const auto& r = answer.solved;
    const auto& c = r.cycle;
    d["status"] = "converged";
    d["reason"] = "";
    d["ff"] = c.fuel_mass_flow;
    d["phi"] = r.match.phi;
    d["thrust_kN"] = c.thrust_kN;
    d["tsfc_mg_Ns"] = c.tsfc_mg_per_Ns;
    d["T3"] = c.compressor.T_out;
    d["T4"] = c.combustor.T_out;
    d["T5"] = c.turbine.T;
    d["p3_bar"] = c.compressor.p_out / 1e5;
    d["thrust_core_kN"] = c.thrust_core_kN;
    d["thrust_bypass_kN"] = c.thrust_bypass_kN;
    d["m_core"] = job.cfg.mass_flow_core;
    d["pi_c"] = job.cfg.pi_c;
    d["nox_corr_g_s"] = c.nox_g_s;
    d["n_cycle_evaluations"] = r.match.n_cycle_evaluations;
    d["probe_failed"] = answer.probe_failed;
    if (answer.cycle_calls >= 0) d["cycle_calls"] = answer.cycle_calls;
    return d;
}

class NativePool {
public:
    // variant: "a" V6Engine; "b" V6Engine + probe; "c" products-only copy +
    // probe; "full" the copied engine with full equilibrium and no probe
    // (untimed faithfulness check of the copy against "a").
    NativePool(const std::string& mechanism, const std::string& fuel,
               const std::vector<std::string>& fuel_species, unsigned int n_threads,
               const std::string& variant)
        : fuel_(fuel), fuel_species_(fuel_species), variant_(variant)
    {
        if (n_threads == 0) throw std::invalid_argument("n_threads must be positive");
        if (variant != "a" && variant != "b" && variant != "c" && variant != "full") {
            throw std::invalid_argument("variant must be a, b, c or full");
        }
        // Cantera mechanism construction is done serially before any worker
        // thread starts; it is excluded from every timed repeat.
        std::vector<std::unique_ptr<V6Engine>> engines;
        std::vector<std::unique_ptr<V6VariantEngine>> copies;
        for (unsigned int i = 0; i < n_threads; ++i) {
            if (variant == "a" || variant == "b") {
                engines.push_back(std::make_unique<V6Engine>(mechanism));
                copies.push_back(nullptr);
            } else {
                engines.push_back(nullptr);
                copies.push_back(std::make_unique<V6VariantEngine>(mechanism,
                    variant == "c" ? V6VariantEngine::Equilibrium::ProductsOnly
                                   : V6VariantEngine::Equilibrium::Full));
            }
        }
        for (unsigned int i = 0; i < n_threads; ++i) {
            workers_.emplace_back(&NativePool::worker, this, std::move(engines[i]),
                                  std::move(copies[i]));
        }
    }

    NativePool(const NativePool&) = delete;
    NativePool& operator=(const NativePool&) = delete;

    ~NativePool()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto& thread : workers_) thread.join();
    }

    py::list run_many(const py::list& requested)
    {
        std::vector<Job> jobs;
        jobs.reserve(requested.size());
        for (auto obj : requested) {
            auto d = py::cast<py::dict>(obj);
            jobs.push_back({parse_config(d["config"].cast<py::dict>()),
                            d["target_kN"].cast<double>(), d["eta_b"].cast<double>()});
        }
        if (jobs.empty()) return py::list();

        std::vector<Answer> answers(jobs.size());
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lock(mutex_);
            jobs_ = &jobs;
            answers_ = &answers;
            next_.store(0);
            finished_ = 0;
            ++generation_;
            cv_.notify_all();
            done_.wait(lock, [&] { return finished_ == workers_.size(); });
            jobs_ = nullptr;
            answers_ = nullptr;
        }
        py::list out;
        for (size_t i = 0; i < answers.size(); ++i) {
            if (answers[i].error) std::rethrow_exception(answers[i].error);
            out.append(as_row(answers[i], jobs[i]));
        }
        return out;
    }

    unsigned int n_threads() const { return static_cast<unsigned int>(workers_.size()); }

private:
    void worker(std::unique_ptr<V6Engine> engine, std::unique_ptr<V6VariantEngine> copy)
    {
        size_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return stopping_ || generation_ != seen; });
                if (stopping_) return;
                seen = generation_;
            }
            for (;;) {
                size_t i = next_.fetch_add(1);
                if (i >= jobs_->size()) break;
                const auto& job = (*jobs_)[i];
                auto& answer = (*answers_)[i];
                const long calls_before = copy ? copy->cycle_calls : 0;
                try {
                    if (variant_ == "a") {
                        engine->config = job.cfg;
                        answer.solved = engine->run_at_thrust(job.target_kN, fuel_, fuel_species_, job.eta_b);
                    } else if (variant_ == "b") {
                        engine->config = job.cfg;
                        answer.solved = solve_with_probe(*engine, job, fuel_, fuel_species_,
                                                         answer.probe_failed);
                    } else if (variant_ == "c") {
                        copy->config = job.cfg;
                        answer.solved = solve_with_probe(*copy, job, fuel_, fuel_species_,
                                                         answer.probe_failed);
                    } else {
                        copy->config = job.cfg;
                        answer.solved = copy->run_at_thrust(job.target_kN, fuel_, fuel_species_, job.eta_b);
                    }
                    answer.reachable = true;
                } catch (const ThrustTargetUnreachable& exc) {
                    answer.reason = exc.reason;
                } catch (...) {
                    answer.error = std::current_exception();
                }
                if (copy) answer.cycle_calls = copy->cycle_calls - calls_before;
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (++finished_ == workers_.size()) done_.notify_one();
            }
        }
    }

    const std::string fuel_;
    const std::vector<std::string> fuel_species_;
    const std::string variant_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_, done_;
    std::atomic<size_t> next_{0};
    const std::vector<Job>* jobs_ = nullptr;
    std::vector<Answer>* answers_ = nullptr;
    size_t generation_ = 0, finished_ = 0;
    bool stopping_ = false;
};

}  // namespace

PYBIND11_MODULE(catjet_benchmark, m)
{
    m.doc() = "P8.1 v6 benchmark std::thread batch pool";
    py::class_<NativePool>(m, "NativePool")
        .def(py::init<const std::string&, const std::string&,
                      const std::vector<std::string>&, unsigned int, const std::string&>(),
             py::arg("mechanism"), py::arg("fuel"), py::arg("fuel_species"),
             py::arg("n_threads"), py::arg("variant") = "a")
        .def("run_many", &NativePool::run_many)
        .def_property_readonly("n_threads", &NativePool::n_threads);
}
