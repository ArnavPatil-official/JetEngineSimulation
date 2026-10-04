"""Frozen NASA7 thermochemistry, shared NumPy/MLX algebra and energy checks."""
from __future__ import annotations

from pathlib import Path

from .inputs import FUELS, canonical_query, state_for
from .registration import sha256_file


def backend_array(value,xp):
    """Explicit backend dtype; never send a NumPy64 buffer to MLX implicitly."""
    import numpy as np
    return xp.array(value,dtype=np.float64 if xp is np else xp.float32)


def freeze_properties(root, reg):
    """Called only after the shared scientific gate/lease authorizes imports."""
    import cantera as ct
    import numpy as np
    import yaml

    root = Path(root)
    mechanism = root / "data/creck_c1c16_full.yaml"
    gas = ct.Solution(str(mechanism))
    source = yaml.safe_load(mechanism.read_text())
    fuel_source = yaml.safe_load((root / "data/fuel_properties_v7.yaml").read_text())
    lca = yaml.safe_load((root / "data/corsia_lca_values.yaml").read_text())
    by_name = {row["name"]: row for row in source["species"]}
    names = list(gas.species_names)
    if len(names) != 492:
        raise ValueError("The frozen 492-species contract changed")
    coefficients, bounds = [], []
    for name in names:
        data = by_name[name]["thermo"]
        if data["model"] != "NASA7" or len(data["data"]) != 2:
            raise ValueError("Unsupported thermo; no approximating fallback")
        coefficients.append(data["data"])
        bounds.append(data["temperature-ranges"])
    mw, aw = np.asarray(gas.molecular_weights), np.asarray(gas.atomic_weights)
    elements = list(gas.element_names)
    matrix = [[gas.n_atoms(k, e) * aw[e] / mw[k] for e in range(len(elements))]
              for k in range(len(names))]
    result = {"species_order": names, "molecular_weights_kg_kmol": mw.tolist(),
              "atomic_weights_kg_kmol": aw.tolist(), "elements": elements,
              "element_mass_matrix": matrix, "gas_constant_J_kmol_K": float(ct.gas_constant),
              "coefficients": coefficients, "temperature_bounds_K": bounds,
              "mechanism_sha256": sha256_file(mechanism),
              "fuel_yaml_sha256": sha256_file(root / "data/fuel_properties_v7.yaml"),
              "lca_yaml_sha256": sha256_file(root / "data/corsia_lca_values.yaml"),
              "lifecycle": lca, "fuels": {}}
    thermo = Thermo(result)
    ref = thermo.species_h(298.15)
    for name in (*reg["fuel"]["surrogates"].values(), "JetA_dooley2010"):
        composition = fuel_source["surrogates"][name]["mole_fractions"]
        mole = np.zeros(len(names), dtype=np.float64)
        for species, value in composition.items():
            mole[names.index(species)] = float(value)
        mole /= mole.sum()
        mean_mw = float(mole @ mw)
        mass = mole * mw / mean_mw
        n_c = sum(mole[k] * gas.n_atoms(k, "C") for k in range(len(names)))
        n_h = sum(mole[k] * gas.n_atoms(k, "H") for k in range(len(names)))
        molar_h = ref * mw
        chemical = float(mole @ molar_h + (n_c + n_h / 4) * molar_h[names.index("O2")]
                         - n_c * molar_h[names.index("CO2")]
                         - n_h / 2 * molar_h[names.index("H2O")])
        gas_lhv = chemical / mean_mw / 1e6
        result["fuels"][name] = {"X": mole.tolist(), "Y": mass.tolist(),
            "lhv_gas_MJ_kg": gas_lhv, "lhv_liquid_MJ_kg": gas_lhv - .360,
            "carbon_mass_fraction": float(mass @ np.asarray(matrix)[:, elements.index("C")]),
            "hydrogen_mass_fraction": float(mass @ np.asarray(matrix)[:, elements.index("H")])}
    result["surrogates"] = reg["fuel"]["surrogates"]
    result["compressor_air_Y"] = thermo.mole_to_mass({"O2": .21, "N2": .79}).tolist()
    result["burner_air_Y"] = thermo.mole_to_mass({"O2": 1, "N2": 3.76}).tolist()
    return result


class Thermo:
    def __init__(self, properties):
        import numpy as np
        self.properties = properties
        self.names = properties["species_order"]
        self.mw = np.asarray(properties["molecular_weights_kg_kmol"], dtype=np.float64)
        self.coefficients = np.asarray(properties["coefficients"], dtype=np.float64)
        self.bounds = np.asarray(properties["temperature_bounds_K"], dtype=np.float64)
        self.Ru = float(properties["gas_constant_J_kmol_K"])
        self.element_matrix = np.asarray(properties["element_mass_matrix"], dtype=np.float64)
        self._compressor_cache = {}

    def mole_to_mass(self, composition):
        import numpy as np
        x = np.zeros(len(self.names), dtype=np.float64)
        for name, value in composition.items():
            x[self.names.index(name)] = value
        if (x < 0).any() or not np.isfinite(x).all() or x.sum() <= 0:
            raise ValueError("Invalid mole composition")
        return x * self.mw / float(x @ self.mw)

    def _species(self, temperature, field, xp=None, *, species_index=None, allow_extrapolation=False):
        import numpy as np
        xp = np if xp is None else xp
        temperature = backend_array(temperature,xp)
        if xp is np:
            temperature = temperature.astype(np.float64)
            check_bounds=self.bounds if species_index is None else self.bounds[[species_index]]
            if not np.isfinite(temperature).all() or (not allow_extrapolation and
                    ((temperature[..., None] < check_bounds[:, 0]) |
                    (temperature[..., None] > check_bounds[:, 2])).any()):
                raise ValueError("Temperature outside frozen species validity")
        t = temperature[..., None]
        coeff = backend_array(self.coefficients if species_index is None else self.coefficients[[species_index]],xp)
        bounds = backend_array(self.bounds if species_index is None else self.bounds[[species_index]],xp)
        c = xp.where((t > bounds[:, 1])[..., None], coeff[:, 1, :], coeff[:, 0, :])
        a = [c[..., i] for i in range(7)]
        if field == "h":
            value = a[0]*t + a[1]*t**2/2 + a[2]*t**3/3 + a[3]*t**4/4 + a[4]*t**5/5 + a[5]
        elif field == "cp":
            value = a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4
        elif field == "s":
            value = a[0]*xp.log(t) + a[1]*t + a[2]*t**2/2 + a[3]*t**3/3 + a[4]*t**4/4 + a[6]
        else:
            raise ValueError("Unsupported NASA7 field")
        return value * self.Ru / backend_array(self.mw if species_index is None else self.mw[[species_index]],xp)

    def species_h(self, temperature, xp=None):
        import numpy as np
        # The unchanged chemical reference is below some source Tmin values.
        # Cantera evaluates these NASA polynomials at the fixed reference too;
        # this explicit exception does not widen model-mixture validity.
        reference = (xp is None or xp is np) and np.all(np.asarray(temperature)==298.15)
        return self._species(temperature, "h", xp, allow_extrapolation=bool(reference))

    def species_cp(self, temperature, xp=None):
        return self._species(temperature, "cp", xp)

    def species_s(self, temperature, xp=None):
        import numpy as np
        ambient = (xp is None or xp is np) and np.all(np.asarray(temperature)==288.15)
        return self._species(temperature, "s", xp, allow_extrapolation=bool(ambient))

    def mixture(self, temperature, mass, xp=None):
        import numpy as np
        xp = np if xp is None else xp
        mass = backend_array(mass,xp)
        cp = xp.sum(mass * self.species_cp(temperature, xp), axis=-1)
        R = self.Ru * xp.sum(mass / backend_array(self.mw,xp), axis=-1)
        h = xp.sum(mass * self.species_h(temperature, xp), axis=-1)
        if xp is np and (not np.isfinite(cp).all() or (cp <= R).any() or (R <= 0).any()):
            raise ValueError("Invalid mixture cp/R")
        return {"h": h, "cp": cp, "R": R, "gamma": cp / (cp - R)}

    def fuel_mass(self, query):
        import numpy as np
        if query.get("fuel_parts") is not None:
            parts = query["fuel_parts"]
        else:
            parts = {self.properties["surrogates"][name]: query[f"f_{name}"] for name in FUELS}
        return sum(float(weight) * np.asarray(self.properties["fuels"][name]["Y"])
                   for name, weight in parts.items())

    def fuel_args(self, query):
        mass = self.fuel_mass(query)
        mole = mass / self.mw
        mole /= mole.sum()
        names = [name for name, value in zip(self.names, mole) if value > 0]
        text = ", ".join(f"{name}:{value:.17g}" for name, value in zip(self.names, mole) if value > 0)
        return text, names

    def compressor_temperature(self, pi_c, eta_c):
        import numpy as np
        key = (float(pi_c), float(eta_c))
        if key in self._compressor_cache:
            return self._compressor_cache[key]
        y = np.asarray(self.properties["compressor_air_Y"], dtype=np.float64)
        R = float(self.Ru * (y / self.mw).sum())
        initial = float(y @ self.species_s(288.15))
        def residual(t):
            # Only O2/N2 contribute to this public compressor-air root. Other
            # species are neither evaluated nor used as validity constraints.
            entropy=sum(y[i]*float(self._species(t,"s",species_index=i,allow_extrapolation=t==288.15)[0]) for i in np.flatnonzero(y))
            return entropy - initial - R * np.log(pi_c)
        lo, hi = 288.15, 3500.0
        if residual(lo) > 0 or residual(hi) < 0:
            raise ValueError("Input-only compressor entropy root not bracketed")
        for _ in range(80):
            mid = (lo + hi) / 2
            if residual(mid) < 0:
                lo = mid
            else:
                hi = mid
            if hi - lo <= 1e-10:
                value = 288.15 + ((lo + hi)/2 - 288.15) / eta_c
                self._compressor_cache[key] = value
                return value
        raise ValueError("Input-only compressor entropy solve exhausted fixed budget")

    def input_states(self, queries, public, draws):
        import numpy as np
        states = []
        fuels = []
        for query in queries:
            q = query
            if query.get("in_product_API") is False:
                q = dict(query, f_JetA=1.0, f_HEFA=0.0, f_FT=0.0, f_ATJ=0.0)
            state = state_for(q, public, draws)
            state["T3"] = self.compressor_temperature(state["pi_c"], state["eta_c"])
            states.append(state)
            fuels.append(self.fuel_mass(query))
        return {key: np.asarray([state[key] for state in states], dtype=np.float64)
                for key in states[0]} | {"fuel_Y": np.asarray(fuels, dtype=np.float64)}

    def residuals(self, ff, T4, Y4, states, xp=None):
        import numpy as np
        xp = np if xp is None else xp
        ma, eta, T3 = (backend_array(states[key],xp) for key in ("ma", "eta_b", "T3"))
        fuel = backend_array(states["fuel_Y"],xp)
        air = backend_array(self.properties["burner_air_Y"],xp)
        href = self.species_h(298.15, xp)
        air_ref = xp.sum(air * href)
        fuel_ref = xp.sum(fuel * href, axis=-1)
        product_ref = xp.sum(Y4 * href, axis=-1)
        hs3 = self.species_h(T3, xp) - href
        incoming = ma*xp.sum(air*hs3, axis=-1) + ff*xp.sum(fuel*hs3, axis=-1)
        outgoing = (ma+ff)*xp.sum(Y4*(self.species_h(T4, xp)-href), axis=-1)
        Qchem = ma*air_ref + ff*fuel_ref - (ma+ff)*product_ref
        energy = (outgoing-incoming-eta*Qchem)/(ma*1e6)
        element = ((ma+ff)[..., None]*(Y4 @ backend_array(self.element_matrix,xp))
                   - ma[..., None]*(air @ backend_array(self.element_matrix,xp))
                   - ff[..., None]*(fuel @ backend_array(self.element_matrix,xp)))/(ma+ff)[..., None]
        return energy, element

    def energy_diagnostics(self,ff,T4,Y4,states):
        import numpy as np
        energy,element=self.residuals(ff,T4,Y4,states)
        ma,eta=states["ma"],states["eta_b"];fuel=states["fuel_Y"]
        href=self.species_h(298.15);air=np.asarray(self.properties["burner_air_Y"])
        Qchem=ma*(air@href)+ff*(fuel@href)-(ma+ff)*(Y4@href)
        # Reconstruct gas LHV from each frozen component mass composition;
        # species formation reference avoids any fitted correction.
        C=self.element_matrix[:,self.properties["elements"].index("C")]
        H=self.element_matrix[:,self.properties["elements"].index("H")]
        carbon=fuel@C;hydrogen=fuel@H
        nC=carbon/self.properties["atomic_weights_kg_kmol"][self.properties["elements"].index("C")]
        nH=hydrogen/self.properties["atomic_weights_kg_kmol"][self.properties["elements"].index("H")]
        molar=href*self.mw
        lhv=(fuel@href+(nC+nH/4)*molar[self.names.index("O2")]
             -nC*molar[self.names.index("CO2")]-nH/2*molar[self.names.index("H2O")])
        rLHV=energy+eta*(Qchem-ff*lhv)/(ma*1e6)
        return {"rE":energy,"rE_LHV":rLHV,"elements":element}
