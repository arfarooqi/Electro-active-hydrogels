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
from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab

plt.rc('font', family='serif')
import matplotlib
matplotlib.font_manager._rebuild()
#plt.rcParams["mathtext.fontset"] = "dejavuserif"

dt = 10
D_an = Constant(1.0E-7) # m^2/s
D_ca = Constant(1.0E-7) # m^2/s
mu_an = 3.9607E-6 # m^2/s*V
mu_ca = 3.9607E-6 # m^2/s*V
z_an = Constant(-1.0)
z_ca = Constant(1.0)
z_fc = Constant(-1.0)
Farad = Constant(9.6487E4) # C/mol
eps0 = Constant(8.854E-12) # As/Vm
epsR = Constant(100.0)
Temp = Constant(293.0)
R = Constant(8.3143)
E = Constant(1.0)
k = Constant(10.0)   #m/M

fc = Constant(2.0) # FCD

mesh = IntervalMesh(1000000, 0.0, 15.0E-3)

refinement_cycles = 2
for _ in range(refinement_cycles):
    refine_cells = MeshFunction("bool", mesh, 1)
    refine_cells.set_all(False)
    for cell in cells(mesh):
        if abs(cell.distance(Point(5.0e-3))) < 2.0e-3:
            refine_cells[cell] = True
        elif abs(cell.distance(Point(10.0e-3))) < 2.0e-3:
            refine_cells[cell] = True
            
    mesh1 = refine(mesh, refine_cells)

P1 = FiniteElement('P', interval, 1)
element = MixedElement([P1, P1, P1])
ME = FunctionSpace(mesh1, element)

def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0.0)
def right_boundary(x, on_boundary):
    return on_boundary and near(x[0], 15.0E-3)

subdomain = CompiledSubDomain("x[0] >= 5.0E-3 && x[0] <= 10.0E-3")
subdomains = MeshFunction("size_t", mesh1, 1)
subdomains.set_all(0)        
subdomain.mark(subdomains, 1)

fc = Constant(2.0) # FCD
V0_r = FunctionSpace(mesh1, 'DG', 0)
fc_function = Function(V0_r)
fc_val = [0.0, fc]
help = np.asarray(subdomains.array(), dtype = np.int32)
fc_function.vector()[:] = np.choose(help, fc_val)

############################# Chemical Stimulation ############################
Poten = 0.0E-3
Sol_c = Constant(1.0)
l_bc_an = DirichletBC(ME.sub(0), Constant(Sol_c), left_boundary)
r_bc_an = DirichletBC(ME.sub(0), Constant(Sol_c), right_boundary)
l_bc_ca = DirichletBC(ME.sub(1), Constant(Sol_c), left_boundary)
r_bc_ca = DirichletBC(ME.sub(1), Constant(Sol_c), right_boundary)
l_bc_psi = DirichletBC(ME.sub(2), Constant(-Poten), left_boundary)
r_bc_psi = DirichletBC(ME.sub(2), Constant(Poten), right_boundary)
bcs = [l_bc_an, r_bc_an, l_bc_ca, r_bc_ca, l_bc_psi, r_bc_psi]

u = Function(ME)
class InitialConditions(UserExpression):
 def eval(self, values, x):
     if((x[0] >= 5.0E-3) and (x[0] <= 10.0E-3)):
         values[0] = 0.0
         values[1] = 1.0
         values[2] = 0.0
     else:
         values[0] = 1.0
         values[1] = 1.0
         values[2] = 0.0
 def value_shape(self):
   return (3,)
u.interpolate(InitialConditions(degree=1))
an, ca, psi = split(u)

du = TrialFunction(ME)
van, vca, vpsi = TestFunctions(ME)

Fan = D_an*(inner(grad(an), grad(van))*dx + (Farad / R / Temp * z_an*an)*inner(grad(psi), grad(van))*dx)
Fca = D_ca*(inner(grad(ca), grad(vca))*dx + (Farad / R / Temp * z_ca*ca)*inner(grad(psi), grad(vca))*dx)
Fpsi = inner(grad(psi), grad(vpsi))*dx - (Farad/(eps0*epsR))*(z_an*an + z_ca*ca + z_fc*fc_function)*vpsi*dx

F = Fpsi + Fan + Fca
J = derivative(F, u)
problem = NonlinearVariationalProblem(F, u, bcs = bcs, J = J)
solver = NonlinearVariationalSolver(problem)
solver.parameters["newton_solver"]["linear_solver"] = "mumps"

solver.solve()
an, ca, psi = u.split()

y1 = np.linspace(0, 0.015, 1000)
#points = [(0,y_) for y_ in y]
ca_line1 = np.array([ca(point) for point in y1])
fc_line1 = np.array([fc_function(point) for point in y1])
an_line1 = np.array([an(point) for point in y1])
psi_line1 = np.array([psi(point) for point in y1])
plt.plot(y1, an_line1, 'k')
#plt.plot(y1, ca_line1,'b--', linewidth=1.5)
############################## Electrical Stimulation #########################
mesh1 = IntervalMesh(10000, 0.0, 15.0E-3)

P1 = FiniteElement('P', interval, 1)
element = MixedElement([P1, P1, P1])
ME = FunctionSpace(mesh1, element)

def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0.0)
def right_boundary(x, on_boundary):
    return on_boundary and near(x[0], 15.0E-3)

subdomain = CompiledSubDomain("x[0] >= 5.0E-3 && x[0] <= 10.0E-3")
subdomains = MeshFunction("size_t", mesh1, 1)
subdomains.set_all(0)        
subdomain.mark(subdomains, 1)

fc = Constant(2.0) # FCD
V0_r = FunctionSpace(mesh1, 'DG', 0)
fc_function = Function(V0_r)
fc_val = [0.0, fc]
help = np.asarray(subdomains.array(), dtype = np.int32)
fc_function.vector()[:] = np.choose(help, fc_val)

Sol_c = 1.0
l_bc_an = DirichletBC(ME.sub(0), Constant(Sol_c), left_boundary)
r_bc_an = DirichletBC(ME.sub(0), Constant(Sol_c), right_boundary)
l_bc_ca = DirichletBC(ME.sub(1), Constant(Sol_c), left_boundary)
r_bc_ca = DirichletBC(ME.sub(1), Constant(Sol_c), right_boundary)
l_bc_psi = DirichletBC(ME.sub(2), Constant(-50.0E-3), left_boundary)
r_bc_psi = DirichletBC(ME.sub(2), Constant(50.0E-3), right_boundary)

bcs = [l_bc_an, r_bc_an, l_bc_ca, r_bc_ca, l_bc_psi, r_bc_psi]

u = Function(ME)
u_n = Function(ME)
class InitialConditions(UserExpression):
 def eval(self, values, x):
     if((x[0] >= 5.0E-3) and (x[0] <= 10.0E-3)):
         values[0] = 0.414
         values[1] = 2.414
         values[2] = -22.252E-3
     else:
         values[0] = 1.0
         values[1] = 1.0
         values[2] = 0.0
 def value_shape(self):
   return (3,)
u.interpolate(InitialConditions(degree=1))
u_n.interpolate(InitialConditions(degree=1))

an, ca, psi = split(u)
an_prev, ca_prev, psi_prev = split(u_n)

van, vca, vpsi = TestFunctions(ME)

Fan = an*van*dx- an_prev*van*dx - dt*D_an*(-inner(grad(an), grad(van))*dx - Farad / R / Temp * z_an*an*inner(grad(psi), grad(van))*dx)
Fca = ca*vca*dx - ca_prev*vca*dx - dt*D_ca*(-inner(grad(ca), grad(vca))*dx - Farad / R / Temp * z_ca*ca*inner(grad(psi), grad(vca))*dx)
Fpsi = inner(grad(psi), grad(vpsi))*dx - (Farad/(eps0*epsR))*(z_an*an + z_ca*ca + z_fc*fc_function)*vpsi*dx
F = Fan + Fca + Fpsi
J = derivative(F, u)

t = 0.0
num_steps = 100*dt
y2 = np.linspace(0, 0.015, 1000)
while t <= num_steps:

    if(t == 10):
        problem = NonlinearVariationalProblem(F, u, bcs=bcs, J=J)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["newton_solver"]["linear_solver"] = "mumps"
        solver.parameters["newton_solver"]["absolute_tolerance"] = 5.0e-6
        solver.parameters['newton_solver']['maximum_iterations'] = 20
        solver.solve() 
        an1, ca1, psi1 = u.split()
        an_line2 = np.array([an1(point) for point in y2])
        plt.plot(y2, an_line2, 'b')
    if(t == 20):
        problem = NonlinearVariationalProblem(F, u, bcs=bcs, J=J)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["newton_solver"]["linear_solver"] = "mumps"
        solver.parameters["newton_solver"]["absolute_tolerance"] = 5.0e-6
        solver.parameters['newton_solver']['maximum_iterations'] = 20
        solver.solve() 
        an1, ca1, psi1 = u.split()
        an_line2 = np.array([an1(point) for point in y2])
        plt.plot(y2, an_line2, 'g')
    if(t == 100):
        problem = NonlinearVariationalProblem(F, u, bcs=bcs, J=J)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["newton_solver"]["linear_solver"] = "mumps"
        solver.parameters["newton_solver"]["absolute_tolerance"] = 5.0e-6
        solver.parameters['newton_solver']['maximum_iterations'] = 20
        solver.solve() 
        an1, ca1, psi1 = u.split()
        an_line2 = np.array([an1(point) for point in y2])
        plt.plot(y2, an_line2, 'r')
    if(t == 800):
        problem = NonlinearVariationalProblem(F, u, bcs=bcs, J=J)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["newton_solver"]["linear_solver"] = "mumps"
        solver.parameters["newton_solver"]["absolute_tolerance"] = 5.0e-6
        solver.parameters['newton_solver']['maximum_iterations'] = 20
        solver.solve() 
        an1, ca1, psi1 = u.split()
        an_line2 = np.array([an1(point) for point in y2])
        plt.plot(y2, an_line2, 'm')
      
    u_n.assign(u)
    t+=dt
    print("value of t", t)

#plt.plot(y1, ca_line1,'b--', linewidth=1.5)
plt.legend(['$t = 0$s','$t = 10$s','$t = 20$s','$t = 100$s','$t = 800$s','$t = 800$s'], fontsize = 11)
plt.xlabel('$x$ (m)', fontsize = 14)
#plt.title("Anion concentration")
plt.ylabel('Anion concentration (mM)', fontsize = 14)
plt.xlim(0, 0.015)
plt.xticks([0, 0.005, 0.010, 0.015], fontsize = 12)
plt.yticks(fontsize = 12)
plt.grid(color = 'k', linestyle = '-', linewidth = 0.1)
#plt.legend(['$\psi$'], loc='best', markerfirst = False, prop={'size': 15})
#plt.savefig('anion_t', dpi=1000, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, \
#            bbox_inches='tight', pad_inches=0.1, frameon=None)
plt.show()
