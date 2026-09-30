// brentq port: SciPy 1.16.3 reference values (computed with scipy.optimize.brentq,
// see scripts/phase8/brentq_reference.py) must be reproduced bit for bit.
#include "catjet_core/brentq.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

using catjet::brentq;

TEST_CASE("brentq reproduces the scipy reference roots bit for bit")
{
    // file written by scripts/phase8/brentq_reference.py: name a b xtol rtol root funcalls (%.17g)
    std::ifstream in(CATJET_BRENTQ_REFERENCE);
    REQUIRE(in.good());
    std::string line;
    int n = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string name;
        double a, b, xtol, rtol, root_ref;
        int calls_ref;
        ss >> name >> a >> b >> xtol >> rtol >> root_ref >> calls_ref;
        std::function<double(double)> f;
        if (name == "cubic") f = [](double x) { return x * x * x - 2 * x - 5; };
        else if (name == "cos") f = [](double x) { return std::cos(x) - x; };
        else if (name == "exp") f = [](double x) { return std::exp(x) - 3.0; };
        else if (name == "tanh") f = [](double x) { return std::tanh(x - 0.3); };
        else FAIL("unknown function " << name);
        catjet::BrentqStats st;
        double root = brentq(f, a, b, xtol, rtol, 100, &st);
        INFO(name << " " << a << " " << b);
        CHECK(root == root_ref);
        CHECK(st.funcalls == calls_ref);
        n++;
    }
    CHECK(n >= 8);
}

TEST_CASE("brentq argument and sign errors match scipy")
{
    auto f = [](double x) { return x - 1.0; };
    CHECK_THROWS_AS(brentq(f, 2.0, 3.0), std::invalid_argument);
    CHECK_THROWS_AS(brentq(f, 0.0, 3.0, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(brentq(f, 0.0, 3.0, 1e-12, 1e-17), std::invalid_argument);
    CHECK_THROWS_AS(brentq([](double) { return std::nan(""); }, 0.0, 1.0), std::domain_error);
    CHECK(brentq(f, 1.0, 3.0) == 1.0);   // f(a) == 0 returns a
}
