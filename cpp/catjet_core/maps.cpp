#include "maps.hpp"

#include <algorithm>
#include <stdexcept>

namespace catjet {

GridTable::GridTable(std::vector<std::vector<double>> grids, std::vector<double> data)
    : grids_(std::move(grids)), data_(std::move(data))
{
    size_t n = 1;
    strides_.assign(grids_.size(), 1);
    for (size_t d = grids_.size(); d-- > 0;) {
        const auto& g = grids_[d];
        if (g.size() < 2) throw std::invalid_argument("map axis needs at least two points");
        for (size_t i = 1; i < g.size(); ++i) {
            if (!(g[i] > g[i - 1])) throw std::invalid_argument("map axis must be strictly ascending");
        }
        strides_[d] = n;
        n *= g.size();
    }
    if (n != data_.size()) throw std::invalid_argument("map data size does not match its grids");
}

double GridTable::operator()(const std::vector<double>& x, bool* extrapolated) const
{
    if (x.size() != grids_.size()) throw std::invalid_argument("map coordinate count mismatch");
    const size_t D = grids_.size();
    std::vector<size_t> lo(D);
    std::vector<double> t(D);
    for (size_t d = 0; d < D; ++d) {
        const auto& g = grids_[d];
        // interval index; values outside use the end interval (linear extrapolation)
        size_t i = std::upper_bound(g.begin(), g.end(), x[d]) - g.begin();
        i = std::clamp<size_t>(i, 1, g.size() - 1) - 1;
        lo[d] = i;
        t[d] = (x[d] - g[i]) / (g[i + 1] - g[i]);
        if (extrapolated && (x[d] < g.front() || x[d] > g.back())) *extrapolated = true;
    }
    double value = 0.0;
    for (size_t corner = 0; corner < (size_t(1) << D); ++corner) {
        double w = 1.0;
        size_t index = 0;
        for (size_t d = 0; d < D; ++d) {
            const bool up = (corner >> d) & 1;
            w *= up ? t[d] : 1.0 - t[d];
            index += (lo[d] + (up ? 1 : 0)) * strides_[d];
        }
        value += w * data_[index];
    }
    return value;
}

double ComponentMap::eval(const std::string& output, const std::vector<double>& x, bool* extrapolated) const
{
    auto it = outputs.find(output);
    if (it == outputs.end()) throw std::invalid_argument(name + " has no map output " + output);
    return it->second(x, extrapolated);
}

}  // namespace catjet
