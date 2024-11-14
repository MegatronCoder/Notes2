def karatsuba(x, y):
    if x < 10 or y < 10:
        return x*y
    n = max(len(str(x)), len(str(y)))
    half = n // 2
    high1, low1 = divmod(x, 10**half)
    high2, low2 = divmod(y, 10**half)
    z2 = karatsuba(high1, high2)
    z0 = karatsuba(low1, low2)
    z1 = karatsuba((low1 + high1), (low2 + high2)) - z0 - z2
    multiplication = (z2 * 10 ** (2 * half) + z1 * 10 ** half + z0)
    return multiplication
if __name__ == "__main__":
    num = input("Enter Multiplication :- ")
    x, y = num.split("x")
    mul = karatsuba(int(x.strip()), int(y.strip()))
    print(f"{num} = {mul}")