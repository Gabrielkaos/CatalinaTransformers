import random
import sympy
from sympy import symbols, solve, Eq

def generate_systems_of_equations(n=10000, max_coeff=10, include_3eq=False):
    """
    Generate n systems of equations in Mathematica format
    """
    data = []
    variables = ['x', 'y', 'z', 'w', 'a', 'b', 'c', 'd']
    
    for i in range(n):
        # Choose number of equations (mostly 2, some 3)
        num_eq = 2 if random.random() < 0.8 else 3
        num_vars = num_eq  # Square systems
        
        # Select variables
        vars_used = variables[:num_vars]
        
        # Generate coefficients
        coeffs = []
        constants = []
        
        for eq in range(num_eq):
            eq_coeffs = [random.randint(-max_coeff, max_coeff) for _ in range(num_vars)]
            const = random.randint(-max_coeff*5, max_coeff*5)
            coeffs.append(eq_coeffs)
            constants.append(const)
        
        # Check if solvable (non-zero determinant for square)
        if num_eq == num_vars == 2:
            det = coeffs[0][0]*coeffs[1][1] - coeffs[0][1]*coeffs[1][0]
            if det == 0:
                continue
        elif num_eq == num_vars == 3:
            # 3x3 determinant check
            a, b, c = coeffs[0]
            d, e, f = coeffs[1]
            g, h, i = coeffs[2]
            det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
            if det == 0:
                continue
        
        # Solve symbolically
        try:
            sym_vars = symbols(' '.join(vars_used))
            equations = []
            
            for eq_idx in range(num_eq):
                expr = sum(coeffs[eq_idx][var_idx] * sym_vars[var_idx] 
                          for var_idx in range(num_vars))
                equations.append(Eq(expr, constants[eq_idx]))
            
            solution = solve(equations, sym_vars)
            
            if not solution:
                continue
                
            # Format equations
            eq_strings = []
            for eq_idx in range(num_eq):
                terms = []
                for var_idx in range(num_vars):
                    coeff = coeffs[eq_idx][var_idx]
                    if coeff != 0:
                        sign = "+" if coeff > 0 else ""
                        if abs(coeff) == 1:
                            term = f"{sign}{vars_used[var_idx]}"
                        else:
                            term = f"{sign}{coeff}*{vars_used[var_idx]}"
                        terms.append(term)
                
                eq_str = ' '.join(terms).strip()
                if eq_str.startswith('+'):
                    eq_str = eq_str[1:]
                eq_strings.append(f"{eq_str} == {constants[eq_idx]}")
            
            # Format solution
            sol_parts = []
            for var, value in zip(vars_used, sym_vars):
                if value in solution:
                    sol_parts.append(f"{'{'}{var} -> {solution[value]}{'}'}")
            
            # Final format
            equations_str = "{" + ", ".join(eq_strings) + "}"
            solution_str = "{" + ", ".join(sol_parts) + "}"
            
            text = f"Solve[{equations_str}, {{{', '.join(vars_used)}}}] == {solution_str}"
            data.append({"text": text})
            
        except Exception as e:
            continue
    
    return data



# Generate 5000 systems
systems_data = generate_systems_of_equations(10,include_3eq=True)

print(*systems_data,sep="\n")