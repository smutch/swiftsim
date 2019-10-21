#include <hdf5.h>
#include <hdf5_hl.h>

#include "darkmatter_write_grids.h"

static char fname[] = {"testDarkMatterWriteGrids.h5"};
static int grid_dim = 16;

static int isclose(float x, float y) {
  return (fabs(x - y) < 1e-8);
}

static hid_t create_output_file(void) {
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t h_file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);
    hid_t h_grp = H5Gcreate(h_file, "PartType1", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Gclose(h_grp);

    return h_file;
}

static void read_results(const char ds_name[], float *data) {

    hid_t h_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

    char ds[256];
    snprintf(ds, 256, "/PartType1/Grids/%s", ds_name);

    H5LTread_dataset_float(h_file, ds, data);
}

int main(int argc, char *argv[])
{
  const int Npart = 10;
  const double dim[3] = {100., 100., 100.};
  const int n_threads = 1;
  const int n_cells = grid_dim * grid_dim * grid_dim;

  MPI_Init(&argc, &argv);

  int i_rank, n_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks); 

  struct unit_system internal_units, snapshot_units;
  units_init(&internal_units, 1.988480e+43, 3.085678e+24, 3.085678e+19,
      1.000000e+00, 1.000000e+00);
  units_init(&snapshot_units, 1.988480e+43, 3.085678e+24, 3.085678e+19,
      1.000000e+00, 1.000000e+00);

  struct engine engine;
  struct space space;
  engine.s = &space;
  engine.snapshot_grid_dim = grid_dim;
  threadpool_init(&engine.threadpool, n_threads);

  for(int ii=0; ii<3; ++ii) {
      space.dim[ii] = dim[ii];
  }
  struct gpart* gparts = calloc(Npart, sizeof(struct gpart));
  space.gparts = (struct gpart*)gparts;

  float *res;
  if(i_rank == 0) {
    res = calloc(grid_dim*grid_dim*grid_dim, sizeof(float));
  }

  /* TEST CASE 1 */
  {
      snprintf(engine.snapshot_grid_method, 4, "NGP");

      for(int ii=0; ii<Npart; ++ii) {
          for(int jj=0; jj<3; ++jj) {
              gparts[ii].x[jj] = 0.;
              gparts[ii].v_full[jj] = 2. * jj;
          }
          gparts[ii].mass = 10.;
      }

      hid_t h_file = create_output_file();
      darkmatter_write_grids(&engine, (size_t)Npart, h_file, &internal_units, &snapshot_units);
      H5Fclose(h_file);

      if (i_rank == 0) {
          read_results("Density", res);

          assert(isclose(res[0], 0.8192f));
          for(int ii = 1; ii < n_cells; ++ii) {
              assert(isclose(res[ii], 0.0f));
          }

          read_results("Vx", res);
          for(int ii = 0; ii < n_cells; ++ii) {
              assert(isclose(res[ii], 0.0f));
          }

          read_results("Vy", res);
          assert(isclose(res[0], 2.0f));
          for(int ii = 1; ii < n_cells; ++ii) {
              assert(isclose(res[ii], 0.0f));
          }

          read_results("Vz", res);
          assert(isclose(res[0], 4.0f));
          for(int ii = 1; ii < n_cells; ++ii) {
              assert(isclose(res[ii], 0.0f));
          }
      }   
  }
          
  if(i_rank == 0) {
      free(res);
  }

  free(gparts);
  threadpool_clean(&engine.threadpool);
  MPI_Finalize();

  return EXIT_SUCCESS;
}
