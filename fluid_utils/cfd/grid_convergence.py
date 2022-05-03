def grid_convergence_metric(phi_fine: float, phi_coarse: float) -> float:
    """Grid convergence metric as used e.g. in [1].
    phi_fine and phi_coarse are point or integral variables on the fine
    and coarse grid respectively.

    References
    ----------
    [1] COLEMAN, Hugh W.; STERN, Fred. Uncertainties and CFD code validation. 1997.
    """
    eps = (phi_fine - phi_coarse) / phi_fine
    return eps


def grid_convergence_index(phi_fine: float, phi_coarse: float, r: float, p: int):
    """
    Grid convergence index [1].
    phi_fine and phi_coarse are point or integral variables on the fine
    and coarse grid respectively. r represents the grid refinement ratio and p the
    order of accuracy.

    References
    ----------
    [1] ROACHE, Patrick J. Quantification of uncertainty in computational fluid dynamics.
        Annual review of fluid Mechanics, 1997, 29. Jg., Nr. 1, S. 123-160.
    """
    gci = 3 * grid_convergence_metric(phi_fine, phi_coarse) / (r ** p - 1)
    return gci
