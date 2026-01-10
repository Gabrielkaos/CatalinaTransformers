def generate_synthetic_equations(n_samples, seed=42):
    """
    Generate clean math reasoning samples:
    - Linear equations
    - Quadratic equations (exact roots)
    - 2x2 linear systems

    Output format is instruction-style reasoning + final answer
    """
    import random
    import sympy as sp

    random.seed(seed)

    x, y = sp.symbols("x y")
    samples = []

    for _ in range(n_samples):
        eq_type = random.choice(["linear", "quadratic", "system"])

        # --------------------
        # LINEAR EQUATION
        # --------------------
        if eq_type == "linear":
            a = random.randint(1, 10)
            b = random.randint(-20, 20)
            c = random.randint(-50, 50)

            problem = f"Solve for x: {a}x + {b} = {c}"

            rhs = c - b
            sol = sp.Rational(rhs, a)

            solution = (
                f"{a}x + {b} = {c}\n"
                f"{a}x = {c} - {b}\n"
                f"{a}x = {rhs}\n"
                f"x = {rhs}/{a}\n"
                f"x = {sol}"
            )

            answer = f"x = {sol}"

        # --------------------
        # QUADRATIC EQUATION
        # --------------------
        elif eq_type == "quadratic":
            a = random.randint(1, 3)
            b = random.randint(-10, 10)
            c = random.randint(-20, 20)

            expr = a*x**2 + b*x + c
            roots = sp.solve(expr, x)

            problem = f"Solve for x: {a}x² + {b}x + {c} = 0"

            discriminant = b**2 - 4*a*c

            solution = (
                f"a = {a}, b = {b}, c = {c}\n"
                f"Discriminant = b² - 4ac = {discriminant}\n"
                f"x = (-b ± √{discriminant}) / (2a)\n"
                f"Solutions: {roots}"
            )

            answer = f"x = {roots}"

        # --------------------
        # SYSTEM OF EQUATIONS
        # --------------------
        else:
            a1 = random.randint(1, 5)
            b1 = random.randint(1, 5)
            c1 = random.randint(-10, 10)

            a2 = random.randint(1, 5)
            b2 = random.randint(1, 5)
            c2 = random.randint(-10, 10)

            eq1 = a1*x + b1*y - c1
            eq2 = a2*x + b2*y - c2

            sol = sp.solve([eq1, eq2], (x, y), dict=True)

            problem = (
                "Solve the system:\n"
                f"{a1}x + {b1}y = {c1}\n"
                f"{a2}x + {b2}y = {c2}"
            )

            solution = (
                f"Equation 1: {a1}x + {b1}y = {c1}\n"
                f"Equation 2: {a2}x + {b2}y = {c2}\n"
                f"Solving simultaneously gives:\n"
                f"{sol}"
            )

            answer = f"x = {sol[0][x]}, y = {sol[0][y]}" if sol else "No solution"

        samples.append({
            "text": f"{problem}\n\n{solution}\n\nFinal Answer:\n{answer}"
        })

    return samples

    
    # from datasets import Dataset
    # return Dataset.from_list(samples)

print(*generate_synthetic_equations(10),sep="\n")