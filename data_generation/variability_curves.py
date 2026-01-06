# SPDX-FileCopyrightText: 2026 Marcel Schepelmann <schepelmann@chi.uni-hannover.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

def CtD(D):
    """Upper envelope diameter (Eq. 18)."""
    return (
        -1.48955986e-02 * D**5
        + 3.90132701e-01 * D**4
        - 3.91277322e00 * D**3
        + 1.84521194e01 * D**2
        - 3.87697694e01 * D
        + 3.13677831e01
    )


def CbD(D):
    """Lower envelope diameter (Eq. 19)."""
    return (
        -2.22353948e-02 * D**5
        + 5.37382372e-01 * D**4
        - 4.93536474e00 * D**3
        + 2.14395701e01 * D**2
        - 4.32336145e01 * D
        + 3.43052813e01
    )


def apply_isocurve(D_raw, r_I):
    """
    Apply Eq. 20:
        D_final = CbD(D) + (CtD(D) - CbD(D)) * r_I
    Works elementwise for arrays.
    """
    return CbD(D_raw) + (CtD(D_raw) - CbD(D_raw)) * r_I
