#    Copyright (C) 2019 Abdul Razzaq Farooqi, abdul.farooqi[AT]uni-rostock.de
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import print_function
from dolfin import (File, Constant, near, IntervalMesh, interval, split, MeshFunction, cells, refine, CompiledSubDomain, DirichletBC, Point, FiniteElement, \
                    MixedElement, FunctionSpace, Function, UserExpression, TestFunctions, derivative, NonlinearVariationalProblem, assign, dot, ds, sqrt, \
                    inner, grad, dx, SubMesh, solve, plot, TestFunction, TrialFunction, VectorFunctionSpace, as_vector, VectorElement, project, FacetNormal, \
                    interpolate, Expression, NonlinearVariationalSolver, nabla_grad, TrialFunctions, assemble, LinearVariationalSolver, RectangleMesh, \
                    LinearVariationalProblem)
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import dolfin as d

from timeit import default_timer as timer
startime = timer() 

D_an = Constant(1.0E-7) # m^2/s
D_ca = Constant(1.0E-7) # m^2/s
mu_an = Constant(3.9607E-6) # m^2/s*V
mu_ca = Constant(3.9607E-6) # m^2/s*V
z_an = Constant(-1.0)
z_ca = Constant(1.0)
z_fc = Constant(-1.0)
Farad = Constant(9.6487E4) # C/mol
eps0 = Constant(8.854E-12) # As/Vm
epsR = Constant(100.0)
Temp = Constant(293.0)
R = Constant(8.3143)

################################## mesh part ##################################

mesh = RectangleMesh(Point(0.0, 0.0), Point(15.0E-3, 15.0E-3), 160, 160)

plot(mesh, title = "Original Mesh")
#File('mesh_orig.pvd') << mesh
plt.show()

center = [7.5E-3, 7.5E-3]
refinement_cycles = 5
for _ in range(refinement_cycles):
    refine_cells = MeshFunction("bool", mesh, 2)
    refine_cells.set_all(False)
    for cell in cells(mesh):
        mp = cell.midpoint()
        loc = sqrt((mp.x() - center[0])**2. + (mp.y() - center[1])**2.)
        if loc < 2.7E-3 and loc > 2.3E-3:
            refine_cells[cell] = True
    plot(refine_cells, title = "Refined Cells")
    plt.show()        
    mesh = refine(mesh, refine_cells)
    plot(mesh, title = "Refined Mesh")
    plt.show()
    File('mesh_refine.pvd') << mesh

def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0.0)
def right_boundary(x, on_boundary):
    return on_boundary and near(x[0], 15.0E-3)
def bottom_boundary(x, on_boundary):
    return on_boundary and near(x[1], 0.0)
def top_boundary(x, on_boundary):
    return on_boundary and near(x[1], 15.0E-3) 

################################### FE part ###################################   
    
P1 = FiniteElement('P', 'triangle', 2)
element = MixedElement([P1, P1, P1])
ME = FunctionSpace(mesh, element)

subdomain = CompiledSubDomain('(pow((x[0] - 7.5E-3), 2) + pow((x[1] - 7.5E-3), 2)) <= pow(2.5E-3, 2)')
subdomains = MeshFunction("size_t", mesh, 2)
subdomains.set_all(0)        
subdomain.mark(subdomains, 1)

fc = Constant(2.0) # FCD
V0_r = FunctionSpace(mesh, 'DG', 0)
fc_function = Function(V0_r)
fc_val = [0.0, fc]

help = np.asarray(subdomains.array(), dtype = np.int32)
fc_function.vector()[:] = np.choose(help, fc_val)
zeroth = plot(fc_function, title = "Fixed charge density, $c^f$")
plt.colorbar(zeroth)
plot(fc_function)
plt.xlim(0.0, 0.015)
plt.xticks([0.0, 0.005, 0.01, 0.015])
plt.ylim(0.0, 0.015)
plt.yticks([0.0, 0.005, 0.01, 0.015])
plt.show()

Sol_c = Constant(1.0)
Poten = 50.0E-3
l_bc_an = DirichletBC(ME.sub(0), Sol_c, left_boundary)
r_bc_an = DirichletBC(ME.sub(0), Sol_c, right_boundary)
l_bc_ca = DirichletBC(ME.sub(1), Sol_c, left_boundary)
r_bc_ca = DirichletBC(ME.sub(1), Sol_c, right_boundary)
l_bc_psi = DirichletBC(ME.sub(2), Constant(-Poten), left_boundary)
r_bc_psi = DirichletBC(ME.sub(2), Constant(Poten), right_boundary)
bcs = [l_bc_an, r_bc_an, l_bc_ca, r_bc_ca, l_bc_psi, r_bc_psi]

u = Function(ME)

V = FunctionSpace(mesh, P1)
an_int = interpolate(Expression('((pow((x[0] - 7.5E-3), 2) + pow((x[1] - 7.5E-3), 2)) <= pow(2.5E-3, 2)) ? 0.4142 : 1.0', degree = 1), V)
ca_int = interpolate(Expression('((pow((x[0] - 7.5E-3), 2) + pow((x[1] - 7.5E-3), 2)) <= pow(2.5E-3, 2)) ? 2.4142 : 1.0', degree = 1), V)
psi_int = interpolate(Expression('((pow((x[0] - 7.5E-3), 2) + pow((x[1] - 7.5E-3), 2)) <= pow(2.5E-3, 2)) ? -22.252E-3 : 0.0', degree = 1), V)
assign(u.sub(0), an_int)
assign(u.sub(1), ca_int)
assign(u.sub(2), psi_int)

an, ca, psi = split(u)
van, vca, vpsi = TestFunctions(ME)

Fan = D_an*(-inner(grad(an), grad(van))*dx - Farad / R / Temp * z_an*an*inner(grad(psi), grad(van))*dx)
Fca = D_ca*(-inner(grad(ca), grad(vca))*dx - Farad / R / Temp * z_ca*ca*inner(grad(psi), grad(vca))*dx)
Fpsi = inner(grad(psi), grad(vpsi))*dx - (Farad/(eps0*epsR))*(z_an*an + z_ca*ca + z_fc*fc_function)*vpsi*dx

F = Fpsi + Fan + Fca

J = derivative(F, u)
problem = NonlinearVariationalProblem(F, u, bcs = bcs, J = J)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters
#print(solver.default_parameters().items())
prm["newton_solver"]["linear_solver"] = 'mumps'
solver.solve()

#########################  Post - Processing  #################################
an, ca, psi = u.split()

aftersolveT = timer() 

first = plot(an, title = "Anion concentration, $c^-$")
plt.xlim(0.0, 0.015)
plt.xticks([0.0, 0.005, 0.01, 0.015])
plt.ylim(0.0, 0.015)
plt.yticks([0.0, 0.005, 0.01, 0.015])
plt.colorbar(first)
plt.show()

second = plot(ca, title = "Cation concentration, $c^+$")
plt.xlim(0.0, 0.015)
plt.xticks([0.0, 0.005, 0.01, 0.015])
plt.ylim(0.0, 0.015)
plt.yticks([0.0, 0.005, 0.01, 0.015])
plt.colorbar(second)
plt.show()

third = plot(psi, title = "Electric potential, $\psi$")
plt.xlim(0.0, 0.015)
plt.xticks([0.0, 0.005, 0.01, 0.015])
plt.ylim(0.0, 0.015)
plt.yticks([0.0, 0.005, 0.01, 0.015])
plt.colorbar(third)
plt.show()

File('an_ES.pvd') << an
File('ca_ES.pvd') << ca
File('psi_ES.pvd') << psi

totime = aftersolveT - startime
#print("Start time is : " + str(round(startime, 2)))
#print("After solve time : " + str(round(aftersolveT, 2)))
print("Number of DOFs: {}".format(ME.dim()))
print("Total time for Simulation : " + str(round(totime)) + "s")
