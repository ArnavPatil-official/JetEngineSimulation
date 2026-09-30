// P8.2 composition-carrying thermodynamic station state.
#pragma once

#include <vector>

namespace catjet {

struct GasState {
    double T = 0.0;  // K
    double P = 0.0;  // Pa
    std::vector<double> Y;  // mechanism-order species mass fractions
};

}  // namespace catjet
