// P8.1 toolchain probe: a pybind11 module linked to the conda libcantera 3.2.0,
// imported by the .venv interpreter next to the pip Cantera wheel. It checks
// that the two Cantera builds and C++ runtimes coexist in one process before
// the real catjet_core bindings are written. Same recipe as hello_equilibrium.
#include "cantera/core.h"
#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;

static double hp_equilibrium_T(const std::string& mech, double T, double p, double phi,
                               const std::string& fuel)
{
    auto gas = Cantera::newSolution(mech)->thermo();
    gas->setState_TP(T, p);
    gas->setEquivalenceRatio(phi, fuel, "O2:1.0, N2:3.76");
    gas->equilibrate("HP");
    return gas->temperature();
}

PYBIND11_MODULE(catjet_probe, m)
{
    m.doc() = "P8.1 toolchain probe (not the catjet_core API)";
    m.def("hp_equilibrium_T", &hp_equilibrium_T);
    m.attr("cantera_version") = CANTERA_VERSION;
}
