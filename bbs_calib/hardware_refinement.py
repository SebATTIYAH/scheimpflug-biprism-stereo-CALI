"""Opt-in bounded prism corrections, not independent hardware metrology."""
from copy import deepcopy
from numbers import Real
import math

DEFAULTS = {
    'wedge_L_deg': {'max_delta': 0.2, 'prior_sigma': 0.1},
    'wedge_R_deg': {'max_delta': 0.2, 'prior_sigma': 0.1},
    'refractive_index': {'max_delta': 0.003, 'prior_sigma': 0.0015},
}

def add_arguments(parser):
    parser.add_argument('--refine-hardware', choices=['wedges', 'all', 'off'],
                        help='prism only: bounded correction of wedges, wedges+n, or disable; default uses config')

def apply_mode(c, mode):
    if mode is None:
        return
    names = list(DEFAULTS) if mode == 'all' else list(DEFAULTS)[:2]
    c.setdefault('optimization', {})['hardware_refinement'] = {
        'enabled': mode != 'off',
        'parameters': {name: deepcopy(DEFAULTS[name]) for name in names} if mode != 'off' else {},
    }

def specifications(c):
    cfg = c.get('optimization', {}).get('hardware_refinement', {})
    if not isinstance(cfg, dict) or set(cfg) - {'enabled', 'parameters'}:
        raise ValueError('hardware_refinement requires an object with enabled/parameters only')
    enabled = cfg.get('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError('hardware_refinement.enabled must be boolean')
    if not enabled:
        return []
    params = cfg.get('parameters')
    if not isinstance(params, dict) or not params:
        raise ValueError('Enabled hardware_refinement requires explicit parameters')
    result = []
    for name, spec in params.items():
        if name not in DEFAULTS:
            raise ValueError('Unsupported hardware refinement parameter: ' + name)
        if not isinstance(spec, dict) or set(spec) != {'max_delta', 'prior_sigma'}:
            raise ValueError(name + ' requires max_delta and prior_sigma only')
        for key, value in spec.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + '.' + key + ' must be positive and finite')
        nominal = float(c['prism'][name])
        low, high = nominal - spec['max_delta'], nominal + spec['max_delta']
        if not all(math.isfinite(v) for v in (nominal,low,high)) or (name == 'refractive_index' and low <= 1) or (name != 'refractive_index' and (low <= 0 or high >= 70)):
            raise ValueError(name + ' refinement bounds leave the physical domain')
        result.append(dict(name=name, nominal=nominal, lower=low, upper=high, **spec))
    return result

def report(c, state, identifiable):
    specs = specifications(c)
    if not specs:
        return None
    parameters = {}
    for spec in specs:
        item = dict(spec)
        name = item.pop('name')
        value = float(state['prism'][name])
        item.update(fitted=value, delta=value-spec['nominal'],
                    at_bound=min(value-spec['lower'], spec['upper']-value)/spec['max_delta'] < 1e-4)
        parameters[name] = item
    return dict(enabled=True, parameters=parameters,
                data_identifiability_passed=bool(identifiable), independently_measured=False,
                regularization='Nominal-centered residual (value-nominal)/prior_sigma; same configured robust loss as image residuals',
                note='Engineering sensitivity fit. Low RMS and prior-augmented rank do not prove physical n/wedge values. Compare held-out RMS, bounds and multistart spread.')
