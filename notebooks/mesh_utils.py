"""
Mesh generation and function creation utils.
"""
from fenics import Point, Expression, Function, interpolate
from mshr import Circle, Polygon, generate_mesh


def get_donut_mesh(n, r=0.3, R=1.0):
    outer = Circle(Point(0, 0), R)
    inner = Circle(Point(0, 0), r)

    domain = outer - inner
    return generate_mesh(domain, n)
    

def get_L_mesh(n):
    domain = Polygon([
        Point(0, 0),
        Point(2, 0),
        Point(2, 1),
        Point(1, 1),
        Point(1, 2),
        Point(0, 2)
    ])

    return generate_mesh(domain, n)


def get_ellipse_mesh(n):
    domain = Circle(Point(0,0), 1.0)
    mesh = generate_mesh(domain, n)

    # Stretch mesh into an ellipse
    X = mesh.coordinates()
    X[:,0] *= 2.0   # scale x
    X[:,1] *= 1.0   # scale y
    return mesh


def get_square_f(V, x0=0.5, y0=0.5, w=0.15, h=0.15):
    x1 = x0 + w
    y1 = y0 + h
    code = f'x[0] >= {x0} && x[0] <= {x1} && x[1] >= {y0} && x[1] <= {y1} ? 1.0 : 0.0'
    f_expr = Expression(code, degree=1)
    f = Function(V)
    f.interpolate(f_expr)
    return f


def get_Gaussian_f(V, x=0.5, y=0.5, sigma=0.05, A=1.0):
    f_expr = Expression(
        "A*exp(-((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0)) / (2*sigma*sigma))",
        degree=4, A=A, x0=x, y0=y, sigma=sigma
    )
    return interpolate(f_expr, V)


def get_disk_f(V, x, y, r=0.05):
    f_expr = Expression(
        "((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0) <= r*r) ? 1.0 : 0.0",
        degree=1, x0=x, y0=y, r=r
    )
    return interpolate(f_expr, V)
