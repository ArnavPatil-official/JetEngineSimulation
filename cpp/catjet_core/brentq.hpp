// SciPy 1.16.3 brentq, ported line for line from scipy/optimize/Zeros/brentq.c
// (Charles Harris) and the Python wrapper scipy/optimize/_zeros_py.py::brentq
// (argument checks, NaN guard, disp=True error behaviour). G0 requires the same
// iterates as the Python v6 path; do not "improve" this routine.
#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace catjet {

struct BrentqStats {
    int funcalls = 0;
    int iterations = 0;
};

// the C macro MIN(a, b) ((a) < (b) ? (a) : (b)); std::min differs for NaN
inline double c_min(double a, double b) { return (a) < (b) ? (a) : (b); }

// scipy defaults: _xtol = 2e-12, _rtol = 4*eps, _iter = 100
constexpr double BRENTQ_RTOL_MIN = 4 * std::numeric_limits<double>::epsilon();

inline double brentq(const std::function<double(double)>& fun, double xa, double xb,
                     double xtol = 2e-12, double rtol = BRENTQ_RTOL_MIN, int iter = 100,
                     BrentqStats* stats = nullptr)
{
    if (xtol <= 0) {
        throw std::invalid_argument("xtol too small (<= 0)");
    }
    if (rtol < BRENTQ_RTOL_MIN) {
        throw std::invalid_argument("rtol too small");
    }
    // _wrap_nan_raise
    auto f = [&](double x) {
        double fx = fun(x);
        if (std::isnan(fx)) {
            throw std::domain_error("The function value at x=" + std::to_string(x) +
                                    " is NaN; solver cannot continue.");
        }
        return fx;
    };
    BrentqStats local;
    BrentqStats& st = stats ? *stats : local;

    double xpre = xa, xcur = xb;
    double xblk = 0., fpre, fcur, fblk = 0., spre = 0., scur = 0., sbis;
    double delta;
    double stry, dpre, dblk;
    int i;

    fpre = f(xpre);
    fcur = f(xcur);
    st.funcalls = 2;
    if (fpre == 0) {
        return xpre;
    }
    if (fcur == 0) {
        return xcur;
    }
    if (std::signbit(fpre) == std::signbit(fcur)) {
        throw std::invalid_argument("f(a) and f(b) must have different signs");
    }
    st.iterations = 0;
    for (i = 0; i < iter; i++) {
        st.iterations++;
        if (fpre != 0 && fcur != 0 && (std::signbit(fpre) != std::signbit(fcur))) {
            xblk = xpre;
            fblk = fpre;
            spre = scur = xcur - xpre;
        }
        if (std::fabs(fblk) < std::fabs(fcur)) {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;

            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        delta = (xtol + rtol * std::fabs(xcur)) / 2;
        sbis = (xblk - xcur) / 2;
        if (fcur == 0 || std::fabs(sbis) < delta) {
            return xcur;
        }

        if (std::fabs(spre) > delta && std::fabs(fcur) < std::fabs(fpre)) {
            if (xpre == xblk) {
                /* interpolate */
                stry = -fcur * (xcur - xpre) / (fcur - fpre);
            } else {
                /* extrapolate */
                dpre = (fpre - fcur) / (xpre - xcur);
                dblk = (fblk - fcur) / (xblk - xcur);
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre));
            }
            if (2 * std::fabs(stry) < c_min(std::fabs(spre), 3 * std::fabs(sbis) - delta)) {
                /* good short step */
                spre = scur;
                scur = stry;
            } else {
                /* bisect */
                spre = sbis;
                scur = sbis;
            }
        } else {
            /* bisect */
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;
        if (std::fabs(scur) > delta) {
            xcur += scur;
        } else {
            xcur += (sbis > 0 ? delta : -delta);
        }

        fcur = f(xcur);
        st.funcalls++;
    }
    throw std::runtime_error("Failed to converge after " + std::to_string(iter) + " iterations");
}

}  // namespace catjet
