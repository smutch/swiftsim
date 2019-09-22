/* Config parameters. */
#include "../config.h"

/* This object's header. */
#include "darkmatter_write_grids.h"

/* Local includes. */
#include <mpi.h>
#include "error.h"
#include "threadpool.h"

#define DS_NAME_SIZE 8

// TODO(smutch): Make this available from `mesh_gravity.c`
/**
 * @brief Returns 1D index of a 3D NxNxN array using row-major style.
 *
 * Wraps around in the corresponding dimension if any of the 3 indices is >= N
 * or < 0.
 *
 * @param i Index along x.
 * @param j Index along y.
 * @param k Index along z.
 * @param N Size of the array along one axis.
 */
__attribute__((always_inline)) INLINE static int row_major_id_periodic(int i,
    int j,
    int k,
    int N) {
  return (((i + N) % N) * N * N + ((j + N) % N) * N + ((k + N) % N));
}


__attribute__((always_inline)) INLINE static int part_to_grid_index(const struct gpart* gp, const double cell_size[3], const double grid_dim, const int n_grid_points) {
  int coord[3] = {-1, -1, -1};

  for(int jj=0; jj < 3; ++jj) {
    coord[jj] = (int)round(gp->x[jj] / cell_size[jj]);
  }

  int index = row_major_id_periodic(coord[0], coord[1], coord[2], grid_dim);
  assert((0 <= index) && (index < n_grid_points));

  return index;
}

struct gridding_extra_data {
  struct gpart* gparts;
  double cell_size[3];
  int grid_dim;
  int n_grid_points;
};


void increment_point_count_mapper(void* restrict temp_point_counts, int N, void* restrict temp_extra_data) {

  int* point_counts = (int *)temp_point_counts;
  const struct gridding_extra_data extra_data = *((const struct gridding_extra_data*)temp_extra_data);
  const struct gpart* gparts = extra_data.gparts;
  const double* cell_size = extra_data.cell_size;
  const int grid_dim = extra_data.grid_dim;
  const int n_grid_points = extra_data.n_grid_points;

  for(int ii=0; ii < N; ++ii) {
    const struct gpart* gp = &gparts[ii];
    int index = part_to_grid_index(gp, cell_size, grid_dim, n_grid_points);
    // Assumption here is that all particles have the same mass.
    atomic_inc(&(point_counts[index]));  // note that this can be reused as required for other gridded properties
  }
}


// TODO(smutch): Make this use threads (see `mesh_gravity.c` for an example)
// TODO(smutch): Remove asserts?
void darkmatter_write_grids(struct engine* e, const size_t Npart, const hid_t h_file, 
    const struct unit_system* internal_units,
    const struct unit_system* snapshot_units) {

  struct gpart* gparts = e->s->gparts;
  const int grid_dim = 16;  // TODO(smutch): Make this a variable
  const int n_grid_points = grid_dim * grid_dim * grid_dim;
  const double *box_size = e->s->dim;
  char dataset_name[DS_NAME_SIZE] = "";

  double cell_size[3] = {0, 0, 0};
  for(int ii=0; ii < 3; ++ii) {
    cell_size[ii] = box_size[ii] / (double)grid_dim;
  }
  
  double* grid = NULL;
  if (swift_memalign("writegrid", (void**)&grid, IO_BUFFER_ALIGNMENT,
                     n_grid_points * sizeof(double)) != 0)
      error("Failed to allocate output DM grids!");
  for(int ii = 0; ii < n_grid_points; ++ii) {
    grid[ii] = 0.0;
  }

  int* point_counts = NULL;
  if (swift_memalign("countgrid", (void**)&point_counts, IO_BUFFER_ALIGNMENT,
                     n_grid_points * sizeof(int)) != 0)
      error("Failed to allocate point counts grids!");
  for(int ii = 0; ii < n_grid_points; ++ii) {
    point_counts[ii] = 0;
  }

  
  // Calculate information for the write that is not dependent on the property being written

  // create the group to store the results
  hid_t h_grp = H5Gcreate(h_file, "/PartType1/Grids", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (h_grp < 0)
    error("Error while creating dark matter grids group.");

  int i_rank, n_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

  // split the write into slabs on the x axis
  {
    // TODO(smutch): Deal with case when this isn't true.
    int tmp = grid_dim % n_ranks;
    assert(tmp == 0);
  }
  int local_slab_size = grid_dim / n_ranks;
  int local_offset = local_slab_size * i_rank;
  if (i_rank == n_ranks - 1) {
    local_slab_size = grid_dim - local_slab_size;
  }

  // create the filespace
  hsize_t dims[3] = { grid_dim, grid_dim, grid_dim };
  hid_t fspace_id = H5Screate_simple(3, dims, NULL);

  // set the dataset creation property list to use chunking along x-axis
  hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(dcpl_id, 3, (hsize_t[3]) { 1, grid_dim, grid_dim });

  // create the property list
  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);

  // select a hyperslab in the filespace
  hsize_t start[3] = { local_offset, 0, 0 };
  hsize_t count[3] = { local_slab_size, grid_dim, grid_dim };
  H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, start, NULL, count, NULL);
  
  // create the memspace
  hsize_t mem_dims[3] = { local_slab_size, grid_dim, grid_dim };
  hid_t memspace_id = H5Screate_simple(3, mem_dims, NULL);

  struct gridding_extra_data extra_data;
  extra_data.gparts = gparts;
  memcpy(extra_data.cell_size, cell_size, sizeof(double)*3);
  extra_data.grid_dim = grid_dim;
  extra_data.n_grid_points = n_grid_points;

  enum grid_types {
    DENSITY,
    VELOCITY_X,
    VELOCITY_Y,
    VELOCITY_Z
  };

  for(enum grid_types grid_type=DENSITY; grid_type <= VELOCITY_Z; ++grid_type) {
    // TODO(smutch): Hoping this will be faster than switch statement in the loop (possible the compiler would fix that anyway though)
    /* Loop through all particles and assign to the grid. */
    switch (grid_type) {
      case DENSITY:
        threadpool_map((struct threadpool*)&e->threadpool, increment_point_count_mapper, point_counts, Npart, grid_dim*grid_dim, 0, (void*)&extra_data);
        break;

      case VELOCITY_X:
        for(size_t ii=0; ii < Npart; ++ii) {
          const struct gpart* gp = &gparts[ii];
          int index = part_to_grid_index(gp, cell_size, grid_dim, n_grid_points);
          grid[index] += gp->v_full[0];
        }
        break;

      case VELOCITY_Y:
        for(size_t ii=0; ii < Npart; ++ii) {
          const struct gpart* gp = &gparts[ii];
          int index = part_to_grid_index(gp, cell_size, grid_dim, n_grid_points);
          grid[index] += gp->v_full[1];
        }
        break;

      case VELOCITY_Z:
        for(size_t ii=0; ii < Npart; ++ii) {
          const struct gpart* gp = &gparts[ii];
          int index = part_to_grid_index(gp, cell_size, grid_dim, n_grid_points);
          grid[index] += gp->v_full[2];
        }
        break;
    }

    /* reduce the grids */
    MPI_Allreduce(MPI_IN_PLACE, grid, n_grid_points, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, point_counts, n_grid_points, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // swift_declare_aligned_ptr(double, temp_d, (double*)temp,
    //     IO_BUFFER_ALIGNMENT);

    /* Do any necessary conversions */
    switch (grid_type) {
      double n_to_density;
      double unit_conv_factor;
      case DENSITY:
        /* convert n_particles to density */
        unit_conv_factor = units_conversion_factor(internal_units, snapshot_units, UNIT_CONV_DENSITY);
        n_to_density = gparts[0].mass * unit_conv_factor / (cell_size[0] * cell_size[1] * cell_size[2]);
        for(int ii=0; ii < n_grid_points; ++ii) {
          grid[ii] = n_to_density * (double)point_counts[ii];
        }
        break;

      case VELOCITY_X:
      case VELOCITY_Y:
      case VELOCITY_Z:
        /* take the mean */
        unit_conv_factor = units_conversion_factor(internal_units, snapshot_units, UNIT_CONV_VELOCITY);
        for(int ii=0; ii < n_grid_points; ++ii) {
          grid[ii] *= unit_conv_factor / (double)point_counts[ii];
        }
        break;
    }

    switch (grid_type) {
      case DENSITY:
        snprintf(dataset_name, DS_NAME_SIZE, "Density");
        break;

      case VELOCITY_X:
        snprintf(dataset_name, DS_NAME_SIZE, "Vx");
        break;

      case VELOCITY_Y:
        snprintf(dataset_name, DS_NAME_SIZE, "Vy");
        break;

      case VELOCITY_Z:
        snprintf(dataset_name, DS_NAME_SIZE, "Vz");
        break;
    }

    // create the dataset
    hid_t dset_id = H5Dcreate(h_grp, dataset_name, H5T_NATIVE_DOUBLE, fspace_id,
        H5P_DEFAULT, dcpl_id, H5P_DEFAULT);

    // write and close the dataset
    H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace_id, fspace_id, plist_id, &grid[row_major_id_periodic(local_offset, 0, 0, grid_dim)]);
    H5Dclose(dset_id);

    // reset the grid if needed
    if (grid_type != VELOCITY_Z) {
      bzero(grid, n_grid_points * sizeof(double));
    }

  } 

  H5Sclose(memspace_id);
  H5Pclose(plist_id);
  H5Pclose(dcpl_id);
  H5Sclose(fspace_id);
  H5Gclose(h_grp);

  /* free the grids */
  if (point_counts) swift_free("countgrid", point_counts);
  if (grid) swift_free("writegrid", grid);
}
