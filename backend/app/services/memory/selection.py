"""Pure budget selection over source-bound Wiki addresses and semantic units."""
from __future__ import annotations

import math


def dependency_closure(roots, dependencies):
    found, pending = set(), list(roots)
    while pending:
        page = pending.pop()
        if page in found:
            continue
        found.add(page)
        pending.extend(dependencies.get(page, ()))
    return found


def fit_pages(ordered, required, dependencies, costs, remaining, loaded=()):
    """Reserve required closures first; skip optional bundles that do not fit.

    A bundle is one chosen page plus its directed prerequisites. Shared pages
    cost once; cached pages cost zero. Cycles are finite atomic closures.
    """
    cached = set(loaded)
    mandatory = dependency_closure(required, dependencies)
    cost = sum(costs[p] for p in mandatory - cached)
    if cost > remaining:
        return {'accepted': [], 'required': sorted(mandatory), 'skipped': list(ordered),
                'required_tokens': cost, 'remaining_tokens': remaining, 'needs_reselection': True}
    selected = set(mandatory)
    skipped = []
    for page in ordered:
        bundle = dependency_closure([page], dependencies)
        additional = sum(costs[p] for p in bundle - selected - cached)
        if cost + additional <= remaining:
            selected.update(bundle)
            cost += additional
        else:
            skipped.append(page)
    return {'accepted': sorted(selected), 'required': sorted(mandatory), 'skipped': skipped,
            'selected_tokens': cost, 'remaining_tokens': remaining - cost,
            'needs_reselection': bool(ordered and not selected and not cached)}


def _unit_value(values, unit):
    """Explicit value signal only; never inferred from recency or truth."""
    if values is None:
        return 1.0
    try:
        value = float(values.get(unit, 0.0))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def fit_units(ordered, required, dependencies, costs, remaining, loaded=(), values=None,
              ranking="density"):
    """Greedy incremental-value assembly over complete unit closures.

    Required units and their *strong dependency* closure are reserved first.
    A required dependency missing from ``costs``/``loaded`` is an explicit
    insufficiency, never silently omitted. Optional units are scored by
    ``value / full incremental closure cost`` where the closure excludes units
    already selected or cached. The ranking is recomputed after every insertion,
    so shared dependencies change density without being charged twice. A unit is
    accepted or skipped as a whole; conditions are never truncated. Cycles are
    finite closures. Zero-cost units are taken first; ties are deterministic by
    value and unit identifier. Recency is not a value or truth signal.

    ``ranking='value'`` is the executable same-budget ablation that disables
    the density denominator; closure costs are still charged and the fit check
    is unchanged. It is not a production truth rule.
    """
    ranking = "value" if ranking == "value" else "density"
    cached = set(loaded)
    known = set(costs) | cached
    dependencies = {key: list(dict.fromkeys(values_ or []))
                    for key, values_ in (dependencies or {}).items()}

    required_roots = list(dict.fromkeys(required))
    required_unknown = [unit for unit in required_roots if unit not in known]
    required_closure = dependency_closure(required_roots, dependencies)
    missing_required = sorted(required_closure - known)
    if required_unknown or missing_required:
        return {
            'accepted': [], 'required': sorted(required_closure),
            'skipped': list(dict.fromkeys(ordered)),
            'required_tokens': 0, 'remaining_tokens': remaining,
            'needs_reselection': False, 'insufficient': True,
            'reason': 'missing_dependency', 'ranking': ranking,
            'missing_dependencies': {
                unit: [dep for dep in dependencies.get(unit, [])
                       if dep not in known]
                for unit in required_closure
                if any(dep not in known for dep in dependencies.get(unit, []))
            } | ({unit: [] for unit in required_unknown} if required_unknown else {}),
        }

    mandatory = dependency_closure(required_roots, dependencies) & known
    cost = sum(costs[unit] for unit in mandatory - cached)
    if cost > remaining:
        return {'accepted': [], 'required': sorted(mandatory), 'skipped': list(dict.fromkeys(ordered)),
                'required_tokens': cost, 'remaining_tokens': remaining,
                'needs_reselection': True, 'insufficient': True,
                'reason': 'required_over_budget', 'missing_dependencies': {},
                'ranking': ranking}

    selected = set(mandatory)
    missing_optional = {}
    ordered = list(dict.fromkeys(ordered))
    pending = [unit for unit in ordered if unit not in selected and unit not in cached]
    deferred: list[str] = []
    while pending:
        scored = []
        for unit in pending:
            if unit in selected:
                continue
            bundle = dependency_closure([unit], dependencies)
            unknown = bundle - known
            if unknown:
                missing_optional[unit] = sorted(unknown)
                continue
            additional = sum(costs[item] for item in bundle - selected - cached)
            value = _unit_value(values, unit)
            if additional <= 0:
                density = math.inf
            else:
                density = value / additional
            scored.append((density, value, unit, bundle, additional))
        if not scored:
            break
        if ranking == "value":
            # Ablation: same closure/budget checks, but order by explicit value
            # only. Density is retained as a deterministic tie-breaker.
            scored.sort(key=lambda item: (-item[1], -item[0], item[2]))
        else:
            # Zero-cost bundles first, then highest density, then explicit value,
            # then stable unit id. No recency/order fallback for equal value.
            scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        _, _, unit, bundle, additional = scored[0]
        pending = [item for item in pending if item != unit]
        if additional <= remaining - cost:
            selected.update(bundle - cached)
            cost += additional
            # Shared prerequisites can make previously oversized bundles fit;
            # re-evaluate them after every insertion instead of freezing the
            # first-pass skip decision.
            retry = [item for item in deferred
                     if item not in selected and item not in pending]
            if retry:
                pending.extend(retry)
                deferred = [item for item in deferred if item not in retry]
        else:
            deferred.append(unit)
    skipped = [unit for unit in ordered
               if unit not in selected and unit not in cached]
    return {
        'accepted': sorted(selected),
        'required': sorted(mandatory),
        'skipped': skipped,
        'selected_tokens': cost,
        'required_tokens': sum(costs[unit] for unit in mandatory - cached),
        'remaining_tokens': remaining - cost,
        'needs_reselection': False,
        'insufficient': False,
        'reason': '',
        'ranking': ranking,
        'missing_dependencies': missing_optional,
        'cached_units': sorted(cached),
    }
