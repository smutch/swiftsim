/* Config parameters. */
#include "../config.h"

/* This object's header. */
#include "darkmatter_write_grids.h"

/* Local includes. */
#include "error.h"


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


// TODO(smutch): Make this use threads (see `mesh_gravity.c` for an example)
void darkmatter_write_grids(struct engine* e, const size_t Npart, const hid_t h_file) {

    struct gpart* gparts = e->s->gparts;
    const int grid_dim = 128;  // TODO(smutch): Make this a variable
    const int n_grid_cells = grid_dim * grid_dim * grid_dim;
    const double *box_size = e->s->dim;

    double cell_size[3] = {0, 0, 0};
    for(int ii=0; ii < 3; ++ii) {
      cell_size[ii] = box_size[ii] / (double)grid_dim;
    }

    //TODO(smutch): Error checking
    double* grid = calloc(n_grid_cells, sizeof(double));

    /* Loop through all particles and assign to the grid. */
    for(size_t ii=0; ii < Npart; ++ii) {
      const struct gpart* gp = &gparts[ii];
      int coord[3] = {-1, -1, -1};

      for(int jj=0; jj < 3; ++jj) {
        coord[jj] = (int)roundf(gp->x[jj] / cell_size[jj]);
      }

      int index = row_major_id_periodic(coord[0], coord[1], coord[2], grid_dim);

      grid[index] += gp->mass;
    }

    /* reduce the grid */
    MPI_Allreduce(MPI_IN_PLACE, grid, n_grid_cells, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    /* write the result */
    hid_t h_grp = H5Gopen(h_file, "/MassTest", H5P_DEFAULT);
    if (h_grp < 0)
      error("Error while opening MassTest group.");

    int i_rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // split the write into slabs on the x axis
    int local_slab_size = (int)(grid_dim / n_ranks);
    int local_offset = i_rank > 0 ? local_slab_size * (n_ranks - 1) : 0;
    if (i_rank == n_ranks - 1) {
        local_slab_size = grid_dim - local_slab_size;
    }

    // create the filespace
    hsize_t dims[3] = { grid_dim, grid_dim, grid_dim };
    hid_t fspace_id = H5Screate_simple(3, dims, NULL);

    // create the memspace
    hsize_t mem_dims[3] = { local_slab_size, grid_dim, grid_dim };
    hid_t memspace_id = H5Screate_simple(3, mem_dims, NULL);

    // set the dataset creation property list to use chunking along x-axis
    hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl_id, 3, (hsize_t[3]) { 1, grid_dim, grid_dim });

    // create the dataset
    hid_t dset_id = H5Dcreate(h_file, "data", H5T_NATIVE_FLOAT, fspace_id,
        H5P_DEFAULT, dcpl_id, H5P_DEFAULT);

    H5Pclose(dcpl_id);

    // create the property list
    hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
    
    // select a hyperslab in the filespace
    hsize_t start[3] = { local_offset, 0, 0 };
    hsize_t count[3] = { local_slab_size, grid_dim, grid_dim };
    H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, start, NULL, count, NULL);

    // write the dataset
    H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace_id, fspace_id, plist_id, grid);

    H5Pclose(plist_id);
    H5Dclose(dset_id);
    H5Sclose(memspace_id);
    H5Sclose(fspace_id);
    H5Gclose(h_grp);

    /* free the grid */
    if (grid) swift_free("grid", grid);
}
