#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

int main(){
    Triangulation<2> mesh;
    GridGenerator::hyper_cube(mesh, 0,1);
    mesh.refine_global(5);
    std::ofstream out("grid.svg");
    GridOut        grid_out;
    grid_out.write_svg(mesh, out);
    std::cout << "Grid written to grid1.svg" << std::endl;
}
