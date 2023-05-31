import matplotlib.pyplot as plt
import numpy as np

t = 1 * (60 * 60 * 24 * 365.25)
q = np.log10(
    -np.array(
        [
            -1e-3,
            -1e-3,
            -1e-5,
            -1e-6,
            -1e-4,
            -1e-4,
            -1e-4,
            -1e-4,
            -1e-5,
            -1e-5,
            -1e-5,
            -1e-5,
            -4.5e-05,
            -1.0e-6,
            -4.5e-05,
        ]
    )
)
dt = (
    np.array(
        [
            t,
            0.5 * t,
            2 * t,
            2 * t,
            2 * t,
            4 * t,
            8 * t,
            6 * t,
            4 * t,
            6 * t,
            8 * t,
            16 * t,
            6 * t,
            16 * t,
            4 * t,
        ]
    )
    / t
)
annot = [
    ">100",
    "93",
    "8",
    "0",
    "32",
    "68",
    ">100",
    ">100",
    "16",
    "32",
    "48",
    "77",
    "75",
    "0",
    "40",
]

fig = plt.figure(figsize=(8, 8))

plt.scatter(q, dt)
plt.xlabel("log10(-q)")
plt.ylabel("Années")
for i in range(len(q)):
    plt.annotate(annot[i], (q[i], dt[i] + 0.3))
plt.show()
