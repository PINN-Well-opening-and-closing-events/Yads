from examples.to_recycle.plots import plot_P_S_newton

vanilla = True
impossible_pressure = False
cheating_PS = True
cheating_S = True
cheating_impossible_pressure = False
cheating_P = True

if vanilla:
    plot_P_S_newton(
        "../saves/newton_1d_rekterino/solss/vanilla/vanilla.json",
        "../saves/newton_1d_rekterino/solss/vanilla/vanilla",
    )
    """
    plot_P_S_newton("../saves/newton_1d_rekterino/impims/vanilla/vanilla.json",
                    "../saves/newton_1d_rekterino/impims/vanilla/vanilla")
    """

if impossible_pressure:
    plot_P_S_newton(
        "../saves/newton_1d_rekterino/solss/impossible_pressure/impossible_pressure.json",
        "../saves/newton_1d_rekterino/solss/impossible_pressure/impossible_pressure",
    )


if cheating_PS:
    plot_P_S_newton(
        "../saves/newton_1d_rekterino/solss/cheating_P_and_S/cheating_P_and_S.json",
        "../saves/newton_1d_rekterino/solss/cheating_P_and_S/cheating_P_and_S",
    )

if cheating_S:
    plot_P_S_newton(
        "../saves/newton_1d_rekterino/solss/cheating_saturation/cheating_S.json",
        "../saves/newton_1d_rekterino/solss/cheating_saturation/cheating_S",
    )

if cheating_P:
    plot_P_S_newton(
        "../saves/newton_1d_rekterino/solss/cheating_pressure/cheating_P.json",
        "../saves/newton_1d_rekterino/solss/cheating_pressure/cheating_P",
    )

if cheating_impossible_pressure:
    plot_P_S_newton(
        "../saves/newton_1d_rekterino/solss/cheating_impossible_pressure/cheating_impossible_pressure.json",
        "../saves/newton_1d_rekterino/solss/cheating_impossible_pressure/cheating_impossible_pressure",
    )
