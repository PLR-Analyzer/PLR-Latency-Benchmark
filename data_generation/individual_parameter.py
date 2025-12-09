import argparse

import numpy as np


def CtD(D):
    """Upper envelope diameter (Eq. 18)."""
    return (
        -0.015 * D**5
        + 0.390 * D**4
        - 3.913 * D**3
        + 18.452 * D**2
        - 38.770 * D
        + 31.368
    )


def CbD(D):
    """Lower envelope diameter (Eq. 19)."""
    return (
        -0.022 * D**5
        + 0.537 * D**4
        - 4.935 * D**3
        + 21.440 * D**2
        - 43.234 * D
        + 34.305
    )


def apply_isocurve(D_raw, r_I):
    """
    Apply Eq. 20:
        D_final = CbD(D) + (CtD(D) - CbD(D)) * r_I
    """
    return CbD(D_raw) + (CtD(D_raw) - CbD(D_raw)) * r_I


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate isocurve transformation for a given diameter"
    )
    parser.add_argument(
        "-d",
        "--diameter",
        type=float,
        default=5.63,  # from José E Capó-Aponte et al.
        help="Initial pupil diameter",
    )
    parser.add_argument(
        "-r",
        "--rI",
        type=float,
        default=1.0,  # value for the blue-eye subject in Pamplona et al.
        help="Individual parameter r_I in [0, 1]",
    )

    args = parser.parse_args()

    print(apply_isocurve(args.diameter, args.rI))
args = parser.parse_args()
