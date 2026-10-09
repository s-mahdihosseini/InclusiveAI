"""Task allocation at resource shadow values with fixed occupation employment.

alpha is the technologically feasible automation frontier. On those tasks the
firm chooses capital or traditional labor. The remaining tasks can use traditional
labor or the labor-compute augmentation technique. Both adoption and physical
input allocation use resource shadow values omega and R. Wage markdowns divide
labor's resource value between wages and owner rents. At fixed capital and
occupation employment, markdowns leave production and technique choice unchanged.
"""
from __future__ import annotations
import numpy as np


def allocation_at_scale(p, scale):
    """Use scale = Y/(markup*R) and solve labor clearing in each occupation.

    v_j = omega_j/(R*psi_j) governs adoption and physical input demands.
    Automation compares 1 with v_j; augmentation compares v_j with
    v_j/G_j+m_j. At equal resource costs, fractions of the task set can
    use each technique to clear fixed occupation employment.
    """
    alpha, gain, req = p.alpha, p.augmentation_gain, p.augmentation_compute_requirement
    target = p.psi * p.masses / (p.eta * scale)
    n = len(alpha)
    v = np.empty(n)
    auto = np.empty(n)
    aug = np.empty(n)
    for j in range(n):
        a, G, m, L = alpha[j], gain[j], req[j], target[j]
        auto_threshold = 1.0
        aug_threshold = m/(1.0-1.0/G) if G > 1.0+1e-14 else None
        boundaries = sorted(set([auto_threshold] + ([aug_threshold] if aug_threshold is not None and aug_threshold > 0 else [])))
        def regimes(z):
            use_auto = float(z > auto_threshold)
            use_aug = float(aug_threshold is not None and z > aug_threshold)
            demand = a*(1-use_auto)/z + (1-a)*((1-use_aug)/z + use_aug/(z+G*m))
            return demand, use_auto, use_aug
        found = False
        for boundary in boundaries:
            lower, aa_low, gg_low = regimes(boundary*(1-1e-10))
            upper, aa_high, gg_high = regimes(boundary*(1+1e-10))
            # Evaluate the exact one-sided quantities at the tie.
            d_low = a*(1-aa_low)/boundary + (1-a)*((1-gg_low)/boundary+gg_low/(boundary+G*m))
            d_high = a*(1-aa_high)/boundary + (1-a)*((1-gg_high)/boundary+gg_high/(boundary+G*m))
            if d_high-1e-13 <= L <= d_low+1e-13 and d_low-d_high > 1e-14:
                mix = np.clip((d_low-L)/(d_low-d_high),0,1)
                v[j] = boundary
                auto[j] = aa_low + mix*(aa_high-aa_low)
                aug[j] = gg_low + mix*(gg_high-gg_low)
                found = True
                break
        if found:
            continue
        for aa in (0.0, 1.0):
            for gg in ((0.0, 1.0) if aug_threshold is not None else (0.0,)):
                A = a*(1-aa)+(1-a)*(1-gg)
                B = (1-a)*gg
                d = G*m
                z = ((A+B-L*d)+np.sqrt((L*d-A-B)**2+4*L*A*d))/(2*L)
                if z > 0 and bool(z > auto_threshold) == bool(aa) and bool(aug_threshold is not None and z > aug_threshold) == bool(gg):
                    v[j], auto[j], aug[j] = z, aa, gg
                    found = True
                    break
            if found:
                break
        if not found:
            raise ValueError('No labor allocation satisfies resource-cost technique choice')
    # Integrate the log resource costs of the techniques actually used.
    # At a mixed allocation, the two techniques have equal resource costs.
    log_cost = np.sum(p.eta*(alpha*(1-auto)*np.log(v)
                    +(1-alpha)*((1-aug)*np.log(v)+aug*np.log(v/gain+req))))
    # A in solver.py absorbs prod(eta**eta), a constant across all scenarios.
    raw_A = p.productivity
    rental = raw_A / p.product_markup * np.exp(-log_cost)
    output = p.product_markup*rental*scale
    shadow = rental*p.psi*v
    auto_capital = scale*float(np.sum(p.eta*alpha*auto))
    aug_capital = scale*float(np.sum(p.eta*(1-alpha)*aug*req/(v/gain+req)))
    labor_implied = scale*p.eta/p.psi*(alpha*(1-auto)/v+(1-alpha)*((1-aug)/v+aug/(v+gain*req)))
    return dict(output=output, capital_rental=rental, shadow_wages=shadow,
                automation_capital=auto_capital, augmentation_capital=aug_capital,
                capital=auto_capital+aug_capital, v=v, resource_v=v,
                private_v=(1-p.wage_markdown)*v,
                automation_adoption=auto, augmentation_adoption=aug,
                labor_implied=labor_implied, scale=scale)


def production_at_capital(p, capital):
    low, high = 1e-10, max(10.0, 4.0*capital)
    while allocation_at_scale(p, high)['capital'] < capital:
        high *= 2
        if high > 1e12:
            raise ValueError('Could not bracket the task allocation')
    for _ in range(52):
        mid = 0.5*(low+high)
        if allocation_at_scale(p, mid)['capital'] < capital:
            low = mid
        else:
            high = mid
    d = allocation_at_scale(p, 0.5*(low+high))
    output, rental, shadow = d['output'], d['capital_rental'], d['shadow_wages']
    wages = (1-p.wage_markdown)*shadow
    labor_bill = float(p.masses@wages)
    markdown_profit = float(p.masses@(p.wage_markdown*shadow))
    product_profit = (1-1/p.product_markup)*output
    total_profit = markdown_profit+product_profit
    ordinary = rental*capital
    factor_residual = output-labor_bill-markdown_profit-product_profit-ordinary
    if abs(factor_residual) > 5e-9*max(1,output):
        raise AssertionError(f'Task payments do not exhaust revenue: {factor_residual}')
    auto_share = p.product_markup*rental*d['automation_capital']/output
    aug_share = p.product_markup*rental*d['augmentation_capital']/output
    slack = shadow/p.psi*(1-1/p.augmentation_gain)-rental*p.augmentation_compute_requirement
    inactive = (p.augmentation_gain==1)&(p.augmentation_compute_requirement==0)
    return dict(output=output, capital=capital, automation_capital=d['automation_capital'],
                augmentation_capital=d['augmentation_capital'], capital_rental=rental,
                r_risky=(ordinary+total_profit)/capital-p.delta,
                shadow_wages=shadow, wages=wages, labor_bill=labor_bill,
                markdown_profit=markdown_profit, product_profit=product_profit,
                total_profit=total_profit, ordinary_capital_income=ordinary,
                automation_capital_share=auto_share, augmentation_capital_share=aug_share,
                physical_capital_share=auto_share+aug_share,
                ordinary_capital_share=ordinary/output,
                product_profit_share=product_profit/output,
                risky_payout_share=(ordinary+total_profit)/output,
                adoption_slack=np.where(inactive,np.nan,slack), factor_residual=factor_residual,
                technique_details=d)
