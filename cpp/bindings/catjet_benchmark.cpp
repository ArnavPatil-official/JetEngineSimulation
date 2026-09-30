// P8.1 benchmark-only std::thread batch runner. The v6 engine and protected
// Python model are unchanged. Each worker owns one V6Engine/Cantera state.
// Pool and engine construction occur before the benchmark warm-up.
#include "../catjet_core/v6_engine.hpp"

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
};

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
    return d;
}

class NativePool {
public:
    NativePool(const std::string& mechanism, const std::string& fuel,
               const std::vector<std::string>& fuel_species, unsigned int n_threads)
        : fuel_(fuel), fuel_species_(fuel_species)
    {
        if (n_threads == 0) throw std::invalid_argument("n_threads must be positive");
        // Cantera mechanism construction is done serially before any worker
        // thread starts; it is excluded from every timed repeat.
        std::vector<std::unique_ptr<V6Engine>> engines;
        engines.reserve(n_threads);
        for (unsigned int i = 0; i < n_threads; ++i) {
            engines.push_back(std::make_unique<V6Engine>(mechanism));
        }
        for (unsigned int i = 0; i < n_threads; ++i) {
            workers_.emplace_back(&NativePool::worker, this, std::move(engines[i]));
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
    void worker(std::unique_ptr<V6Engine> engine)
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
                try {
                    engine->config = job.cfg;
                    answer.solved = engine->run_at_thrust(job.target_kN, fuel_, fuel_species_, job.eta_b);
                    answer.reachable = true;
                } catch (const ThrustTargetUnreachable& exc) {
                    answer.reason = exc.reason;
                } catch (...) {
                    answer.error = std::current_exception();
                }
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (++finished_ == workers_.size()) done_.notify_one();
            }
        }
    }

    const std::string fuel_;
    const std::vector<std::string> fuel_species_;
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
                      const std::vector<std::string>&, unsigned int>(),
             py::arg("mechanism"), py::arg("fuel"), py::arg("fuel_species"),
             py::arg("n_threads"))
        .def("run_many", &NativePool::run_many)
        .def_property_readonly("n_threads", &NativePool::n_threads);
}
