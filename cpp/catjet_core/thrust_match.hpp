// The v6 matched-thrust phi solve as a template over the cycle callback.
// Line-for-line copy of V6Engine::run_at_thrust (v6_engine.cpp at 74f53c8,
// itself a port of integrated_engine.run_at_thrust): same bracket, closure
// bisection, T4 guard, warm-start path, Brent tolerances and reason strings.
// Ablation-ladder engines (P8-A2) swap only the cycle; the G0 engine keeps
// its own copy so its translation unit stays byte-identical.
#pragma once

#include "brentq.hpp"
#include "v6_engine.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <variant>

namespace catjet {
namespace thrust_detail {

inline std::string py_repr(double x)
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

inline std::string fmt(const char* f, double x)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), f, x);
    return buf;
}

}  // namespace thrust_detail

// run_cycle(phi) -> R (may throw CycleDoesNotClose); t4_of(R), thrust_kN_of(R).
// phi_guess < 0 means None. Returns the cycle at the root and the match record.
template <class R, class RunCycle, class T4Of, class ThrustOf>
std::pair<R, ThrustMatch> v6_thrust_match(RunCycle run_cycle, T4Of t4_of, ThrustOf thrust_kN_of,
                                          double target_kN, double combustor_efficiency,
                                          double phi_lo, double phi_hi, double t4_max_K,
                                          double phi_xtol, double phi_guess)
{
    using thrust_detail::fmt;
    using thrust_detail::py_repr;
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

    std::map<double, std::variant<R, std::string>> cache;
    auto cycle = [&](double phi) -> const R& {
        auto it = cache.find(phi);
        if (it == cache.end()) {
            try {
                it = cache.emplace(phi, run_cycle(phi)).first;
            } catch (const CycleDoesNotClose& exc) {
                it = cache.emplace(phi, std::string(exc.what())).first;
            }
        }
        if (std::holds_alternative<std::string>(it->second)) {
            throw CycleDoesNotClose(std::get<std::string>(it->second));
        }
        return std::get<R>(it->second);
    };
    auto runs = [&](double phi) {
        try {
            cycle(phi);
            return true;
        } catch (const CycleDoesNotClose&) {
            return false;
        }
    };
    auto t4 = [&](double phi) { return t4_of(cycle(phi)); };
    auto residual = [&](double phi) { return thrust_kN_of(cycle(phi)) - target; };
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
    ThrustMatch match;
    match.phi = phi;
    match.target_kN = target;
    match.residual_kN = residual(phi);
    match.t4_K = t4(phi);
    match.n_cycle_evaluations = static_cast<int>(cache.size());
    match.info = info;
    return {cycle(phi), match};
}

}  // namespace catjet
