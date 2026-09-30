// P8.1 step 2: HP equilibrium of a fuel/air mixture, printed at full precision,
// for comparison with Python Cantera (scripts/phase8/hello_equilibrium_check.py).
// Mirrors simulation/combustor/combustor.py: fresh Solution, TP, then
// set_equivalence_ratio with oxidizer "O2:1.0, N2:3.76", then equilibrate("HP").
//
// Usage: hello_equilibrium <mechanism.yaml> <T_K> <p_Pa> <phi> <fuel composition>
// Output: one "key value" line each for T, P, h_mass, cp_mass, mean_mw, then
// "Y <species> <value>" for every species, in mechanism order.
#include "cantera/core.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s mech T p phi fuel\n", argv[0]);
        return 2;
    }
    try {
        auto sol = Cantera::newSolution(argv[1]);
        auto gas = sol->thermo();
        gas->setState_TP(std::strtod(argv[2], nullptr), std::strtod(argv[3], nullptr));
        gas->setEquivalenceRatio(std::strtod(argv[4], nullptr), argv[5], "O2:1.0, N2:3.76");
        gas->equilibrate("HP");
        std::printf("T %.17g\nP %.17g\nh_mass %.17g\ncp_mass %.17g\nmean_mw %.17g\n",
                    gas->temperature(), gas->pressure(), gas->enthalpy_mass(),
                    gas->cp_mass(), gas->meanMolecularWeight());
        for (size_t k = 0; k < gas->nSpecies(); k++) {
            std::printf("Y %s %.17g\n", gas->speciesName(k).c_str(), gas->massFraction(k));
        }
    } catch (Cantera::CanteraError& err) {
        std::fprintf(stderr, "%s\n", err.what());
        return 1;
    }
    return 0;
}
