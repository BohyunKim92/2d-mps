import pandas as pd

from base import create_solver


def main():
    solver = create_solver(100, 40, 100)
    eigenmodes = solver.find_eigenmodes(50)
    df = pd.DataFrame({
        'eigenmode': eigenmodes,
    })
    df.to_csv('eigenmodes.csv', index=False)


if __name__ == '__main__':
    main()
