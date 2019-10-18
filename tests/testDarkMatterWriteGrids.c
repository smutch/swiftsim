#include <assert.h>
#include <hdf5.h>
#include "darkmatter_write_grids.h"

int main(int argc, char *argv[])
{
  const size_t Npart = 10;
  const double dim[3] = {100., 100., 100.};
  const int n_threads = 1;

  MPI_Init(&argc, &argv);

  int i_rank, n_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks); 

  struct unit_system internal_units, snapshot_units;
  units_init(&internal_units 1.988480e+43, 3.085678e+24, 3.085678e+19,
      1.000000e+00, 1.000000e+00);
  units_init(&snapshot_units 1.988480e+43, 3.085678e+24, 3.085678e+19,
      1.000000e+00, 1.000000e+00);

  struct engine engine;
  struct space space;
  engine.space = &space;
  engine.snapshot_grid_dim = 16;
  engine.snapshot_grid_method = {"NGP"};
  threadpool_init(&engine.threadpool, n_threads);

  space.dim = dim;
  struct gpart gparts[Npart];
  space.gparts = gparts;

  for(int ii=0; ii<Npart; ++ii) {
    gparts[ii].x = {0., 0., 0.};
    gparts[ii].v_full = {1., 2., 4.};
    gparts[ii].mass = 10.;
  }
  
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t h_file = H5Fcreate("testDarkMatterWriteGrids.h5", H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  H5Pclose(plist_id);

  darkmatter_write_grids(&engine, Npart, h_file, &internal_units, &snapshot_units);

  threadpool_clean(&engine.threadpool);
  H5Fclose(h_file);
  MPI_Finalize();

  return EXIT_SUCCESS;
}
