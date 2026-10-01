// P8.4 generic component maps (docs/phase8_p84_registration.md section 4).
// Multilinear interpolation on the map's own grids with linear
// extrapolation from the end interval, as OpenMDAO MetaModelStructuredComp
// method 'slinear', extrapolate=True (the pinned pyCycle setting).
#pragma once

#include <map>
#include <string>
#include <vector>

namespace catjet {

class GridTable {
public:
    GridTable() = default;
    // grids: one ascending axis per dimension; data: row-major, last axis fastest.
    GridTable(std::vector<std::vector<double>> grids, std::vector<double> data);
    // Value at x (one coordinate per axis); sets extrapolated when any
    // coordinate lies outside its axis range.
    double operator()(const std::vector<double>& x, bool* extrapolated = nullptr) const;
    size_t dims() const { return grids_.size(); }

private:
    std::vector<std::vector<double>> grids_;
    std::vector<double> data_;
    std::vector<size_t> strides_;
};

// One pyCycle map: parameter axes (alphaMap, NcMap|NpMap, RlineMap|PRmap),
// named outputs (WcMap, effMap, PRmap | WpMap, effMap) and design defaults.
struct ComponentMap {
    std::string name;
    std::vector<std::string> params;
    std::map<std::string, GridTable> outputs;
    std::map<std::string, double> defaults;
    double rline_stall = 0.0;
    double eval(const std::string& output, const std::vector<double>& x, bool* extrapolated) const;
};

}  // namespace catjet
