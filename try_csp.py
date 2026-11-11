from csp.sudoku import solve

s = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
grid = [[int(s[r*9+c]) for c in range(9)] for r in range(9)]

for fc, ac in [(False, False), (True, False), (True, True)]:
    sol, stats = solve(grid, fc, ac)
    print("FC:", fc, "AC3:", ac, stats)
