#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
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
 
  Triangulation<2> triangulation;
  FE_Q<2>          fe;
  DoFHandler<2>    dof_handler;
  AffineConstraints<double> constraints;
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
 
  Vector<double> solution;
  Vector<double> system_rhs;
};
 
 
fea::fea()
  : fe(1)
  , dof_handler(triangulation)
{}
 
 
 
void fea::make_grid()
{

}
 
 
 
 
void fea::setup_system()
{

}
 
 
 
void fea::assemble_system()
{

}
 
 
 
void fea::solve()
{

}
 
 
 
void fea::output_results() const
{

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

 
  return 0;
}
