#define S_roll 5
#define M_roll 6
#define Mesh_r 3

// Mukesh V
// ME18B156
// Jul-Nov 2022 FEA Project

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

using namespace dealii;

class Boundary_values : public Function<2>
{
public:
  virtual double value(const Point<2> &p,
                       const unsigned int component = 0) const override;
};
double Boundary_values::value(const Point<2> &p, const unsigned int) const
{
  return (S_roll + M_roll) * 100;
}


class Right_side : public Function<2>
{
public:
  virtual double value(const Point<2> &p,
                       const unsigned int component = 0) const override;
};
double Right_side::value(const Point<2> &p, const unsigned int) const
{
  return std::pow((S_roll * p(0) - M_roll * p(1)), 2);
}


class fea
{
public:
  fea();

  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  Triangulation<2> mesh;
  FE_Q<2> fe;
  DoFHandler<2> dof_handler;
  AffineConstraints<double> constraints;
  SparsityPattern sparsity_pattern;
  SparseMatrix<double> K_matrix;

  Vector<double> d_vector;
  Vector<double> F_vector;
};

fea::fea()
    : fe(1), dof_handler(mesh)
{
}

void fea::make_grid()
{
  const Point<2> center(0, 0);
  GridGenerator::hyper_shell(mesh, center, 2 * (S_roll + M_roll), 4 * (S_roll + M_roll), 25);
  mesh.refine_global(Mesh_r);

  std::cout << "# of active cells: " << mesh.n_active_cells() << std::endl;
}

void fea::setup_system()
{
  dof_handler.distribute_dofs(fe);
  std::cout << "# of dof: " << dof_handler.n_dofs() << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  K_matrix.reinit(sparsity_pattern);
  d_vector.reinit(dof_handler.n_dofs());
  F_vector.reinit(dof_handler.n_dofs());
}

void fea::assemble_system()
{
  QGauss<2> quadrature_formula(fe.degree + 1);
  Boundary_values bv;
  Right_side rhs;
  FEValues<2> fe_values(fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> e_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> e_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);

    e_matrix = 0, e_rhs = 0;
    for (const unsigned int q : fe_values.quadrature_point_indices())
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
        {
          e_matrix(i, j) += (fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * fe_values.JxW(q));

          const auto &x_q = fe_values.quadrature_point(q);
          e_rhs(i) += (fe_values.shape_value(i, q) * rhs.value(x_q) * fe_values.JxW(q));
        }

    cell->get_dof_indices(local_dof_indices);
    for (const unsigned int i : fe_values.dof_indices())
    {
      for (const unsigned int j : fe_values.dof_indices())
        K_matrix.add(local_dof_indices[i], local_dof_indices[j], e_matrix(i, j));
      F_vector(local_dof_indices[i]) += e_rhs(i);
    }
  }

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0, bv, boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, K_matrix, d_vector, F_vector);
}

void fea::solve()
{
  SolverControl solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(K_matrix, d_vector, F_vector, PreconditionIdentity());

  std::cout << "   " << solver_control.last_step() << " CG iterations needed to obtain convergence." << std::endl;
}

void fea::output_results() const
{
  DataOut<2> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(d_vector, "solution");

  data_out.build_patches();

  std::ofstream output("solution-2d.vtk");
  data_out.write_vtk(output);
}

void fea::run()
{
  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}

int main()
{
  deallog.depth_console(2);
  fea code;
  code.run();

  return 0;
}
