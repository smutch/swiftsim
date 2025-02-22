/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2012 Pedro Gonnet (pedro.gonnet@durham.ac.uk),
 *                    Matthieu Schaller (matthieu.schaller@durham.ac.uk).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

/* Config parameters. */
#include "../config.h"

/* This object's header. */
#include "common_io.h"

/* Pre-inclusion as needed in other headers */
#include "engine.h"

/* Local includes. */
#include "black_holes_io.h"
#include "chemistry_io.h"
#include "const.h"
#include "cooling_io.h"
#include "error.h"
#include "fof_io.h"
#include "gravity_io.h"
#include "hydro.h"
#include "hydro_io.h"
#include "io_properties.h"
#include "kernel_hydro.h"
#include "part.h"
#include "part_type.h"
#include "star_formation_io.h"
#include "stars_io.h"
#include "threadpool.h"
#include "tracers_io.h"
#include "units.h"
#include "velociraptor_io.h"
#include "version.h"

/* Some standard headers. */
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(HAVE_HDF5)

#include <hdf5.h>

/* MPI headers. */
#ifdef WITH_MPI
#include <mpi.h>
#endif

/**
 * @brief Converts a C data type to the HDF5 equivalent.
 *
 * This function is a trivial wrapper around the HDF5 types but allows
 * to change the exact storage types matching the code types in a transparent
 *way.
 */
hid_t io_hdf5_type(enum IO_DATA_TYPE type) {

  switch (type) {
    case INT:
      return H5T_NATIVE_INT;
    case UINT:
      return H5T_NATIVE_UINT;
    case LONG:
      return H5T_NATIVE_LONG;
    case ULONG:
      return H5T_NATIVE_ULONG;
    case LONGLONG:
      return H5T_NATIVE_LLONG;
    case ULONGLONG:
      return H5T_NATIVE_ULLONG;
    case FLOAT:
      return H5T_NATIVE_FLOAT;
    case DOUBLE:
      return H5T_NATIVE_DOUBLE;
    case CHAR:
      return H5T_NATIVE_CHAR;
    default:
      error("Unknown type");
      return 0;
  }
}

/**
 * @brief Return 1 if the type has double precision
 *
 * Returns an error if the type is not FLOAT or DOUBLE
 */
int io_is_double_precision(enum IO_DATA_TYPE type) {

  switch (type) {
    case FLOAT:
      return 0;
    case DOUBLE:
      return 1;
    default:
      error("Invalid type");
      return 0;
  }
}

/**
 * @brief Reads an attribute from a given HDF5 group.
 *
 * @param grp The group from which to read.
 * @param name The name of the attribute to read.
 * @param type The #IO_DATA_TYPE of the attribute.
 * @param data (output) The attribute read from the HDF5 group.
 *
 * Calls #error() if an error occurs.
 */
void io_read_attribute(hid_t grp, const char* name, enum IO_DATA_TYPE type,
                       void* data) {

  const hid_t h_attr = H5Aopen(grp, name, H5P_DEFAULT);
  if (h_attr < 0) error("Error while opening attribute '%s'", name);

  const hid_t h_err = H5Aread(h_attr, io_hdf5_type(type), data);
  if (h_err < 0) error("Error while reading attribute '%s'", name);

  H5Aclose(h_attr);
}

/**
 * @brief Reads an attribute from a given HDF5 group.
 *
 * @param grp The group from which to read.
 * @param name The name of the attribute to read.
 * @param type The #IO_DATA_TYPE of the attribute.
 * @param data (output) The attribute read from the HDF5 group.
 *
 * Exits gracefully (i.e. does not read the attribute at all) if
 * it is not present, unless debugging checks are activated. If they are,
 * and the read fails, we print a warning.
 */
void io_read_attribute_graceful(hid_t grp, const char* name,
                                enum IO_DATA_TYPE type, void* data) {

  /* First, we need to check if this attribute exists to avoid raising errors
   * within the HDF5 library if we attempt to access an attribute that does
   * not exist. */
  const htri_t h_exists = H5Aexists(grp, name);

  if (h_exists <= 0) {
  /* Attribute either does not exist (0) or function failed (-ve) */
#ifdef SWIFT_DEBUG_CHECKS
    message("WARNING: attribute '%s' does not exist.", name);
#endif
  } else {
    /* Ok, now we know that it exists we can read it. */
    const hid_t h_attr = H5Aopen(grp, name, H5P_DEFAULT);

    if (h_attr >= 0) {
      const hid_t h_err = H5Aread(h_attr, io_hdf5_type(type), data);
      if (h_err < 0) {
      /* Explicitly do nothing unless debugging checks are activated */
#ifdef SWIFT_DEBUG_CHECKS
        message("WARNING: unable to read attribute '%s'", name);
#endif
      }
    } else {
#ifdef SWIFT_DEBUG_CHECKS
      if (h_attr < 0) {
        message("WARNING: was unable to open attribute '%s'", name);
      }
#endif
    }

    H5Aclose(h_attr);
  }
}

/**
 * @brief Asserts that the redshift in the initial conditions and the one
 *        specified by the parameter file match.
 *
 * @param h_grp The Header group from the ICs
 * @param a Current scale factor as specified by parameter file
 */
void io_assert_valid_header_cosmology(hid_t h_grp, double a) {

  double redshift_from_snapshot = -1.0;
  io_read_attribute_graceful(h_grp, "Redshift", DOUBLE,
                             &redshift_from_snapshot);

  /* If the Header/Redshift value is not present, then we skip this check */
  if (redshift_from_snapshot == -1.0) {
    return;
  }

  const double current_redshift = 1.0 / a - 1.0;
  const double redshift_fractional_difference =
      fabs(redshift_from_snapshot - current_redshift) / current_redshift;

  if (redshift_fractional_difference >= io_redshift_tolerance) {
    error(
        "Initial redshift specified in parameter file (%lf) and redshift "
        "read from initial conditions (%lf) are inconsistent.",
        current_redshift, redshift_from_snapshot);
  }
}

/**
 * @brief Write an attribute to a given HDF5 group.
 *
 * @param grp The group in which to write.
 * @param name The name of the attribute to write.
 * @param type The #IO_DATA_TYPE of the attribute.
 * @param data The attribute to write.
 * @param num The number of elements to write
 *
 * Calls #error() if an error occurs.
 */
void io_write_attribute(hid_t grp, const char* name, enum IO_DATA_TYPE type,
                        const void* data, int num) {

  const hid_t h_space = H5Screate(H5S_SIMPLE);
  if (h_space < 0)
    error("Error while creating dataspace for attribute '%s'.", name);

  hsize_t dim[1] = {(hsize_t)num};
  const hid_t h_err = H5Sset_extent_simple(h_space, 1, dim, NULL);
  if (h_err < 0)
    error("Error while changing dataspace shape for attribute '%s'.", name);

  const hid_t h_attr =
      H5Acreate1(grp, name, io_hdf5_type(type), h_space, H5P_DEFAULT);
  if (h_attr < 0) error("Error while creating attribute '%s'.", name);

  const hid_t h_err2 = H5Awrite(h_attr, io_hdf5_type(type), data);
  if (h_err2 < 0) error("Error while reading attribute '%s'.", name);

  H5Sclose(h_space);
  H5Aclose(h_attr);
}

/**
 * @brief Write a string as an attribute to a given HDF5 group.
 *
 * @param grp The group in which to write.
 * @param name The name of the attribute to write.
 * @param str The string to write.
 * @param length The length of the string
 *
 * Calls #error() if an error occurs.
 */
void io_writeStringAttribute(hid_t grp, const char* name, const char* str,
                             int length) {

  const hid_t h_space = H5Screate(H5S_SCALAR);
  if (h_space < 0)
    error("Error while creating dataspace for attribute '%s'.", name);

  const hid_t h_type = H5Tcopy(H5T_C_S1);
  if (h_type < 0) error("Error while copying datatype 'H5T_C_S1'.");

  const hid_t h_err = H5Tset_size(h_type, length);
  if (h_err < 0) error("Error while resizing attribute type to '%i'.", length);

  const hid_t h_attr = H5Acreate1(grp, name, h_type, h_space, H5P_DEFAULT);
  if (h_attr < 0) error("Error while creating attribute '%s'.", name);

  const hid_t h_err2 = H5Awrite(h_attr, h_type, str);
  if (h_err2 < 0) error("Error while reading attribute '%s'.", name);

  H5Tclose(h_type);
  H5Sclose(h_space);
  H5Aclose(h_attr);
}

/**
 * @brief Writes a double value as an attribute
 * @param grp The group in which to write
 * @param name The name of the attribute
 * @param data The value to write
 */
void io_write_attribute_d(hid_t grp, const char* name, double data) {
  io_write_attribute(grp, name, DOUBLE, &data, 1);
}

/**
 * @brief Writes a float value as an attribute
 * @param grp The group in which to write
 * @param name The name of the attribute
 * @param data The value to write
 */
void io_write_attribute_f(hid_t grp, const char* name, float data) {
  io_write_attribute(grp, name, FLOAT, &data, 1);
}

/**
 * @brief Writes an int value as an attribute
 * @param grp The group in which to write
 * @param name The name of the attribute
 * @param data The value to write
 */
void io_write_attribute_i(hid_t grp, const char* name, int data) {
  io_write_attribute(grp, name, INT, &data, 1);
}

/**
 * @brief Writes a long value as an attribute
 * @param grp The group in which to write
 * @param name The name of the attribute
 * @param data The value to write
 */
void io_write_attribute_l(hid_t grp, const char* name, long data) {
  io_write_attribute(grp, name, LONG, &data, 1);
}

/**
 * @brief Writes a string value as an attribute
 * @param grp The group in which to write
 * @param name The name of the attribute
 * @param str The string to write
 */
void io_write_attribute_s(hid_t grp, const char* name, const char* str) {
  io_writeStringAttribute(grp, name, str, strlen(str));
}

/**
 * @brief Reads the Unit System from an IC file.
 *
 * If the 'Units' group does not exist in the ICs, we will use the internal
 * system of units.
 *
 * @param h_file The (opened) HDF5 file from which to read.
 * @param ic_units The unit_system to fill.
 * @param internal_units The internal system of units to copy if needed.
 * @param mpi_rank The MPI rank we are on.
 */
void io_read_unit_system(hid_t h_file, struct unit_system* ic_units,
                         const struct unit_system* internal_units,
                         int mpi_rank) {

  /* First check if it exists as this is *not* required. */
  const htri_t exists = H5Lexists(h_file, "/Units", H5P_DEFAULT);

  if (exists == 0) {

    if (mpi_rank == 0)
      message("'Units' group not found in ICs. Assuming internal unit system.");

    units_copy(ic_units, internal_units);

    return;

  } else if (exists < 0) {
    error("Serious problem with 'Units' group in ICs. H5Lexists gives %d",
          exists);
  }

  if (mpi_rank == 0) message("Reading IC units from ICs.");
  hid_t h_grp = H5Gopen(h_file, "/Units", H5P_DEFAULT);

  /* Ok, Read the damn thing */
  io_read_attribute(h_grp, "Unit length in cgs (U_L)", DOUBLE,
                    &ic_units->UnitLength_in_cgs);
  io_read_attribute(h_grp, "Unit mass in cgs (U_M)", DOUBLE,
                    &ic_units->UnitMass_in_cgs);
  io_read_attribute(h_grp, "Unit time in cgs (U_t)", DOUBLE,
                    &ic_units->UnitTime_in_cgs);
  io_read_attribute(h_grp, "Unit current in cgs (U_I)", DOUBLE,
                    &ic_units->UnitCurrent_in_cgs);
  io_read_attribute(h_grp, "Unit temperature in cgs (U_T)", DOUBLE,
                    &ic_units->UnitTemperature_in_cgs);

  /* Clean up */
  H5Gclose(h_grp);
}

/**
 * @brief Writes the current Unit System
 * @param h_file The (opened) HDF5 file in which to write
 * @param us The unit_system to dump
 * @param groupName The name of the HDF5 group to write to
 */
void io_write_unit_system(hid_t h_file, const struct unit_system* us,
                          const char* groupName) {

  const hid_t h_grpunit = H5Gcreate1(h_file, groupName, 0);
  if (h_grpunit < 0) error("Error while creating Unit System group");

  io_write_attribute_d(h_grpunit, "Unit mass in cgs (U_M)",
                       units_get_base_unit(us, UNIT_MASS));
  io_write_attribute_d(h_grpunit, "Unit length in cgs (U_L)",
                       units_get_base_unit(us, UNIT_LENGTH));
  io_write_attribute_d(h_grpunit, "Unit time in cgs (U_t)",
                       units_get_base_unit(us, UNIT_TIME));
  io_write_attribute_d(h_grpunit, "Unit current in cgs (U_I)",
                       units_get_base_unit(us, UNIT_CURRENT));
  io_write_attribute_d(h_grpunit, "Unit temperature in cgs (U_T)",
                       units_get_base_unit(us, UNIT_TEMPERATURE));

  H5Gclose(h_grpunit);
}

/**
 * @brief Writes the code version to the file
 * @param h_file The (opened) HDF5 file in which to write
 */
void io_write_code_description(hid_t h_file) {

  const hid_t h_grpcode = H5Gcreate1(h_file, "/Code", 0);
  if (h_grpcode < 0) error("Error while creating code group");

  io_write_attribute_s(h_grpcode, "Code", "SWIFT");
  io_write_attribute_s(h_grpcode, "Code Version", package_version());
  io_write_attribute_s(h_grpcode, "Compiler Name", compiler_name());
  io_write_attribute_s(h_grpcode, "Compiler Version", compiler_version());
  io_write_attribute_s(h_grpcode, "Git Branch", git_branch());
  io_write_attribute_s(h_grpcode, "Git Revision", git_revision());
  io_write_attribute_s(h_grpcode, "Git Date", git_date());
  io_write_attribute_s(h_grpcode, "Configuration options",
                       configuration_options());
  io_write_attribute_s(h_grpcode, "CFLAGS", compilation_cflags());
  io_write_attribute_s(h_grpcode, "HDF5 library version", hdf5_version());
  io_write_attribute_s(h_grpcode, "Thread barriers", thread_barrier_version());
  io_write_attribute_s(h_grpcode, "Allocators", allocator_version());
#ifdef HAVE_FFTW
  io_write_attribute_s(h_grpcode, "FFTW library version", fftw3_version());
#endif
#ifdef HAVE_LIBGSL
  io_write_attribute_s(h_grpcode, "GSL library version", libgsl_version());
#endif
#ifdef WITH_MPI
  io_write_attribute_s(h_grpcode, "MPI library", mpi_version());
#ifdef HAVE_METIS
  io_write_attribute_s(h_grpcode, "METIS library version", metis_version());
#endif
#ifdef HAVE_PARMETIS
  io_write_attribute_s(h_grpcode, "ParMETIS library version",
                       parmetis_version());
#endif
#else
  io_write_attribute_s(h_grpcode, "MPI library", "Non-MPI version of SWIFT");
#endif
  H5Gclose(h_grpcode);
}

/**
 * @brief Write the #engine policy to the file.
 * @param h_file File to write to.
 * @param e The #engine to read the policy from.
 */
void io_write_engine_policy(hid_t h_file, const struct engine* e) {

  const hid_t h_grp = H5Gcreate1(h_file, "/Policy", 0);
  if (h_grp < 0) error("Error while creating policy group");

  for (int i = 1; i < engine_maxpolicy; ++i)
    if (e->policy & (1 << i))
      io_write_attribute_i(h_grp, engine_policy_names[i + 1], 1);
    else
      io_write_attribute_i(h_grp, engine_policy_names[i + 1], 0);

  H5Gclose(h_grp);
}

static long long cell_count_non_inhibited_gas(const struct cell* c) {
  const int total_count = c->hydro.count;
  struct part* parts = c->hydro.parts;
  long long count = 0;
  for (int i = 0; i < total_count; ++i) {
    if ((parts[i].time_bin != time_bin_inhibited) &&
        (parts[i].time_bin != time_bin_not_created)) {
      ++count;
    }
  }
  return count;
}

static long long cell_count_non_inhibited_dark_matter(const struct cell* c) {
  const int total_count = c->grav.count;
  struct gpart* gparts = c->grav.parts;
  long long count = 0;
  for (int i = 0; i < total_count; ++i) {
    if ((gparts[i].time_bin != time_bin_inhibited) &&
        (gparts[i].time_bin != time_bin_not_created) &&
        (gparts[i].type == swift_type_dark_matter)) {
      ++count;
    }
  }
  return count;
}

static long long cell_count_non_inhibited_background_dark_matter(
    const struct cell* c) {
  const int total_count = c->grav.count;
  struct gpart* gparts = c->grav.parts;
  long long count = 0;
  for (int i = 0; i < total_count; ++i) {
    if ((gparts[i].time_bin != time_bin_inhibited) &&
        (gparts[i].time_bin != time_bin_not_created) &&
        (gparts[i].type == swift_type_dark_matter_background)) {
      ++count;
    }
  }
  return count;
}

static long long cell_count_non_inhibited_stars(const struct cell* c) {
  const int total_count = c->stars.count;
  struct spart* sparts = c->stars.parts;
  long long count = 0;
  for (int i = 0; i < total_count; ++i) {
    if ((sparts[i].time_bin != time_bin_inhibited) &&
        (sparts[i].time_bin != time_bin_not_created)) {
      ++count;
    }
  }
  return count;
}

static long long cell_count_non_inhibited_black_holes(const struct cell* c) {
  const int total_count = c->black_holes.count;
  struct bpart* bparts = c->black_holes.parts;
  long long count = 0;
  for (int i = 0; i < total_count; ++i) {
    if ((bparts[i].time_bin != time_bin_inhibited) &&
        (bparts[i].time_bin != time_bin_not_created)) {
      ++count;
    }
  }
  return count;
}

void io_write_cell_offsets(hid_t h_grp, const int cdim[3],
                           const struct cell* cells_top, const int nr_cells,
                           const double width[3], const int nodeID,
                           const long long global_counts[swift_type_count],
                           const long long global_offsets[swift_type_count],
                           const struct unit_system* internal_units,
                           const struct unit_system* snapshot_units) {

  double cell_width[3] = {width[0], width[1], width[2]};

  /* Temporary memory for the cell-by-cell information */
  double* centres = NULL;
  centres = (double*)malloc(3 * nr_cells * sizeof(double));

  /* Count of particles in each cell */
  long long *count_part = NULL, *count_gpart = NULL,
            *count_background_gpart = NULL, *count_spart = NULL,
            *count_bpart = NULL;
  count_part = (long long*)malloc(nr_cells * sizeof(long long));
  count_gpart = (long long*)malloc(nr_cells * sizeof(long long));
  count_background_gpart = (long long*)malloc(nr_cells * sizeof(long long));
  count_spart = (long long*)malloc(nr_cells * sizeof(long long));
  count_bpart = (long long*)malloc(nr_cells * sizeof(long long));

  /* Global offsets of particles in each cell */
  long long *offset_part = NULL, *offset_gpart = NULL,
            *offset_background_gpart = NULL, *offset_spart = NULL,
            *offset_bpart = NULL;
  offset_part = (long long*)malloc(nr_cells * sizeof(long long));
  offset_gpart = (long long*)malloc(nr_cells * sizeof(long long));
  offset_background_gpart = (long long*)malloc(nr_cells * sizeof(long long));
  offset_spart = (long long*)malloc(nr_cells * sizeof(long long));
  offset_bpart = (long long*)malloc(nr_cells * sizeof(long long));

  /* Offsets of the 0^th element */
  offset_part[0] = 0;
  offset_gpart[0] = 0;
  offset_background_gpart[0] = 0;
  offset_spart[0] = 0;
  offset_bpart[0] = 0;

  /* Collect the cell information of *local* cells */
  long long local_offset_part = 0;
  long long local_offset_gpart = 0;
  long long local_offset_background_gpart = 0;
  long long local_offset_spart = 0;
  long long local_offset_bpart = 0;
  for (int i = 0; i < nr_cells; ++i) {

    if (cells_top[i].nodeID == nodeID) {

      /* Centre of each cell */
      centres[i * 3 + 0] = cells_top[i].loc[0] + cell_width[0] * 0.5;
      centres[i * 3 + 1] = cells_top[i].loc[1] + cell_width[1] * 0.5;
      centres[i * 3 + 2] = cells_top[i].loc[2] + cell_width[2] * 0.5;

      /* Count real particles that will be written */
      count_part[i] = cell_count_non_inhibited_gas(&cells_top[i]);
      count_gpart[i] = cell_count_non_inhibited_dark_matter(&cells_top[i]);
      count_background_gpart[i] =
          cell_count_non_inhibited_background_dark_matter(&cells_top[i]);
      count_spart[i] = cell_count_non_inhibited_stars(&cells_top[i]);
      count_bpart[i] = cell_count_non_inhibited_black_holes(&cells_top[i]);

      /* Offsets including the global offset of all particles on this MPI rank
       */
      offset_part[i] = local_offset_part + global_offsets[swift_type_gas];
      offset_gpart[i] =
          local_offset_gpart + global_offsets[swift_type_dark_matter];
      offset_background_gpart[i] =
          local_offset_background_gpart +
          global_offsets[swift_type_dark_matter_background];
      offset_spart[i] = local_offset_spart + global_offsets[swift_type_stars];
      offset_bpart[i] =
          local_offset_bpart + global_offsets[swift_type_black_hole];

      local_offset_part += count_part[i];
      local_offset_gpart += count_gpart[i];
      local_offset_background_gpart += count_background_gpart[i];
      local_offset_spart += count_spart[i];
      local_offset_bpart += count_bpart[i];

    } else {

      /* Just zero everything for the foregin cells */

      centres[i * 3 + 0] = 0.;
      centres[i * 3 + 1] = 0.;
      centres[i * 3 + 2] = 0.;

      count_part[i] = 0;
      count_gpart[i] = 0;
      count_background_gpart[i] = 0;
      count_spart[i] = 0;
      count_bpart[i] = 0;

      offset_part[i] = 0;
      offset_gpart[i] = 0;
      offset_background_gpart[i] = 0;
      offset_spart[i] = 0;
      offset_bpart[i] = 0;
    }
  }

#ifdef WITH_MPI
  /* Now, reduce all the arrays. Note that we use a bit-wise OR here. This
     is safe as we made sure only local cells have non-zero values. */
  if (nodeID == 0) {
    MPI_Reduce(MPI_IN_PLACE, count_part, nr_cells, MPI_LONG_LONG_INT, MPI_BOR,
               0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(count_part, NULL, nr_cells, MPI_LONG_LONG_INT, MPI_BOR, 0,
               MPI_COMM_WORLD);
  }
  if (nodeID == 0) {
    MPI_Reduce(MPI_IN_PLACE, count_gpart, nr_cells, MPI_LONG_LONG_INT, MPI_BOR,
               0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(count_gpart, NULL, nr_cells, MPI_LONG_LONG_INT, MPI_BOR, 0,
               MPI_COMM_WORLD);
  }
  if (nodeID == 0) {
    MPI_Reduce(MPI_IN_PLACE, count_background_gpart, nr_cells,
               MPI_LONG_LONG_INT, MPI_BOR, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(count_background_gpart, NULL, nr_cells, MPI_LONG_LONG_INT,
               MPI_BOR, 0, MPI_COMM_WORLD);
  }
  if (nodeID == 0) {
    MPI_Reduce(MPI_IN_PLACE, count_spart, nr_cells, MPI_LONG_LONG_INT, MPI_BOR,
               0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(count_spart, NULL, nr_cells, MPI_LONG_LONG_INT, MPI_BOR, 0,
               MPI_COMM_WORLD);
  }
  if (nodeID == 0) {
    MPI_Reduce(MPI_IN_PLACE, count_bpart, nr_cells, MPI_LONG_LONG_INT, MPI_BOR,
               0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(count_bpart, NULL, nr_cells, MPI_LONG_LONG_INT, MPI_BOR, 0,
               MPI_COMM_WORLD);
  }

  if (nodeID == 0) {
    MPI_Reduce(MPI_IN_PLACE, offset_part, nr_cells, MPI_LONG_LONG_INT, MPI_BOR,
               0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(offset_part, NULL, nr_cells, MPI_LONG_LONG_INT, MPI_BOR, 0,
               MPI_COMM_WORLD);
  }
  if (nodeID == 0) {
    MPI_Reduce(MPI_IN_PLACE, offset_gpart, nr_cells, MPI_LONG_LONG_INT, MPI_BOR,
               0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(offset_gpart, NULL, nr_cells, MPI_LONG_LONG_INT, MPI_BOR, 0,
               MPI_COMM_WORLD);
  }
  if (nodeID == 0) {
    MPI_Reduce(MPI_IN_PLACE, offset_background_gpart, nr_cells,
               MPI_LONG_LONG_INT, MPI_BOR, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(offset_background_gpart, NULL, nr_cells, MPI_LONG_LONG_INT,
               MPI_BOR, 0, MPI_COMM_WORLD);
  }
  if (nodeID == 0) {
    MPI_Reduce(MPI_IN_PLACE, offset_spart, nr_cells, MPI_LONG_LONG_INT, MPI_BOR,
               0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(offset_spart, NULL, nr_cells, MPI_LONG_LONG_INT, MPI_BOR, 0,
               MPI_COMM_WORLD);
  }
  if (nodeID == 0) {
    MPI_Reduce(MPI_IN_PLACE, offset_bpart, nr_cells, MPI_LONG_LONG_INT, MPI_BOR,
               0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(offset_bpart, NULL, nr_cells, MPI_LONG_LONG_INT, MPI_BOR, 0,
               MPI_COMM_WORLD);
  }

  /* For the centres we use a sum as MPI does not like bit-wise operations
     on floating point numbers */
  if (nodeID == 0) {
    MPI_Reduce(MPI_IN_PLACE, centres, 3 * nr_cells, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(centres, NULL, 3 * nr_cells, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
  }
#endif

  /* Only rank 0 actually writes */
  if (nodeID == 0) {

    /* Unit conversion if necessary */
    const double factor = units_conversion_factor(
        internal_units, snapshot_units, UNIT_CONV_LENGTH);
    if (factor != 1.) {

      /* Convert the cell centres */
      for (int i = 0; i < nr_cells; ++i) {
        centres[i * 3 + 0] *= factor;
        centres[i * 3 + 1] *= factor;
        centres[i * 3 + 2] *= factor;
      }

      /* Convert the cell widths */
      cell_width[0] *= factor;
      cell_width[1] *= factor;
      cell_width[2] *= factor;
    }

    /* Write some meta-information first */
    hid_t h_subgrp =
        H5Gcreate(h_grp, "Meta-data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (h_subgrp < 0) error("Error while creating meta-data sub-group");
    io_write_attribute(h_subgrp, "nr_cells", INT, &nr_cells, 1);
    io_write_attribute(h_subgrp, "size", DOUBLE, cell_width, 3);
    io_write_attribute(h_subgrp, "dimension", INT, cdim, 3);
    H5Gclose(h_subgrp);

    /* Write the centres to the group */
    hsize_t shape[2] = {(hsize_t)nr_cells, 3};
    hid_t h_space = H5Screate(H5S_SIMPLE);
    if (h_space < 0) error("Error while creating data space for cell centres");
    hid_t h_err = H5Sset_extent_simple(h_space, 2, shape, shape);
    if (h_err < 0)
      error("Error while changing shape of gas offsets data space.");
    hid_t h_data = H5Dcreate(h_grp, "Centres", io_hdf5_type(DOUBLE), h_space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (h_data < 0) error("Error while creating dataspace for gas offsets.");
    h_err = H5Dwrite(h_data, io_hdf5_type(DOUBLE), h_space, H5S_ALL,
                     H5P_DEFAULT, centres);
    if (h_err < 0) error("Error while writing centres.");
    H5Dclose(h_data);
    H5Sclose(h_space);

    /* Group containing the offsets for each particle type */
    h_subgrp =
        H5Gcreate(h_grp, "Offsets", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (h_subgrp < 0) error("Error while creating offsets sub-group");

    if (global_counts[swift_type_gas] > 0) {

      shape[0] = nr_cells;
      shape[1] = 1;
      h_space = H5Screate(H5S_SIMPLE);
      if (h_space < 0) error("Error while creating data space for gas offsets");
      h_err = H5Sset_extent_simple(h_space, 1, shape, shape);
      if (h_err < 0)
        error("Error while changing shape of gas offsets data space.");
      h_data = H5Dcreate(h_subgrp, "PartType0", io_hdf5_type(LONGLONG), h_space,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (h_data < 0) error("Error while creating dataspace for gas offsets.");
      h_err = H5Dwrite(h_data, io_hdf5_type(LONGLONG), h_space, H5S_ALL,
                       H5P_DEFAULT, offset_part);
      if (h_err < 0) error("Error while writing gas offsets.");
      H5Dclose(h_data);
      H5Sclose(h_space);
    }

    if (global_counts[swift_type_dark_matter] > 0) {

      shape[0] = nr_cells;
      shape[1] = 1;
      h_space = H5Screate(H5S_SIMPLE);
      if (h_space < 0) error("Error while creating data space for DM offsets");
      h_err = H5Sset_extent_simple(h_space, 1, shape, shape);
      if (h_err < 0)
        error("Error while changing shape of DM offsets data space.");
      h_data = H5Dcreate(h_subgrp, "PartType1", io_hdf5_type(LONGLONG), h_space,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (h_data < 0) error("Error while creating dataspace for DM offsets.");
      h_err = H5Dwrite(h_data, io_hdf5_type(LONGLONG), h_space, H5S_ALL,
                       H5P_DEFAULT, offset_gpart);
      if (h_err < 0) error("Error while writing DM offsets.");
      H5Dclose(h_data);
      H5Sclose(h_space);
    }

    if (global_counts[swift_type_dark_matter_background] > 0) {

      shape[0] = nr_cells;
      shape[1] = 1;
      h_space = H5Screate(H5S_SIMPLE);
      if (h_space < 0)
        error("Error while creating data space for background DM offsets");
      h_err = H5Sset_extent_simple(h_space, 1, shape, shape);
      if (h_err < 0)
        error(
            "Error while changing shape of background DM offsets data space.");
      h_data = H5Dcreate(h_subgrp, "PartType2", io_hdf5_type(LONGLONG), h_space,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (h_data < 0)
        error("Error while creating dataspace for background DM offsets.");
      h_err = H5Dwrite(h_data, io_hdf5_type(LONGLONG), h_space, H5S_ALL,
                       H5P_DEFAULT, offset_background_gpart);
      if (h_err < 0) error("Error while writing background DM offsets.");
      H5Dclose(h_data);
      H5Sclose(h_space);
    }

    if (global_counts[swift_type_stars] > 0) {

      shape[0] = nr_cells;
      shape[1] = 1;
      h_space = H5Screate(H5S_SIMPLE);
      if (h_space < 0)
        error("Error while creating data space for stars offsets");
      h_err = H5Sset_extent_simple(h_space, 1, shape, shape);
      if (h_err < 0)
        error("Error while changing shape of stars offsets data space.");
      h_data = H5Dcreate(h_subgrp, "PartType4", io_hdf5_type(LONGLONG), h_space,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (h_data < 0) error("Error while creating dataspace for star offsets.");
      h_err = H5Dwrite(h_data, io_hdf5_type(LONGLONG), h_space, H5S_ALL,
                       H5P_DEFAULT, offset_spart);
      if (h_err < 0) error("Error while writing star offsets.");
      H5Dclose(h_data);
      H5Sclose(h_space);
    }

    if (global_counts[swift_type_black_hole] > 0) {

      shape[0] = nr_cells;
      shape[1] = 1;
      h_space = H5Screate(H5S_SIMPLE);
      if (h_space < 0)
        error("Error while creating data space for black hole offsets");
      h_err = H5Sset_extent_simple(h_space, 1, shape, shape);
      if (h_err < 0)
        error("Error while changing shape of black hole offsets data space.");
      h_data = H5Dcreate(h_subgrp, "PartType5", io_hdf5_type(LONGLONG), h_space,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (h_data < 0)
        error("Error while creating dataspace for black hole offsets.");
      h_err = H5Dwrite(h_data, io_hdf5_type(LONGLONG), h_space, H5S_ALL,
                       H5P_DEFAULT, offset_bpart);
      if (h_err < 0) error("Error while writing black hole offsets.");
      H5Dclose(h_data);
      H5Sclose(h_space);
    }

    H5Gclose(h_subgrp);

    /* Group containing the counts for each particle type */
    h_subgrp =
        H5Gcreate(h_grp, "Counts", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (h_subgrp < 0) error("Error while creating counts sub-group");

    if (global_counts[swift_type_gas] > 0) {

      shape[0] = nr_cells;
      shape[1] = 1;
      h_space = H5Screate(H5S_SIMPLE);
      if (h_space < 0) error("Error while creating data space for gas counts");
      h_err = H5Sset_extent_simple(h_space, 1, shape, shape);
      if (h_err < 0)
        error("Error while changing shape of gas counts data space.");
      h_data = H5Dcreate(h_subgrp, "PartType0", io_hdf5_type(LONGLONG), h_space,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (h_data < 0) error("Error while creating dataspace for gas counts.");
      h_err = H5Dwrite(h_data, io_hdf5_type(LONGLONG), h_space, H5S_ALL,
                       H5P_DEFAULT, count_part);
      if (h_err < 0) error("Error while writing gas counts.");
      H5Dclose(h_data);
      H5Sclose(h_space);
    }

    if (global_counts[swift_type_dark_matter] > 0) {

      shape[0] = nr_cells;
      shape[1] = 1;
      h_space = H5Screate(H5S_SIMPLE);
      if (h_space < 0) error("Error while creating data space for DM counts");
      h_err = H5Sset_extent_simple(h_space, 1, shape, shape);
      if (h_err < 0)
        error("Error while changing shape of DM counts data space.");
      h_data = H5Dcreate(h_subgrp, "PartType1", io_hdf5_type(LONGLONG), h_space,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (h_data < 0) error("Error while creating dataspace for DM counts.");
      h_err = H5Dwrite(h_data, io_hdf5_type(LONGLONG), h_space, H5S_ALL,
                       H5P_DEFAULT, count_gpart);
      if (h_err < 0) error("Error while writing DM counts.");
      H5Dclose(h_data);
      H5Sclose(h_space);
    }

    if (global_counts[swift_type_dark_matter_background] > 0) {

      shape[0] = nr_cells;
      shape[1] = 1;
      h_space = H5Screate(H5S_SIMPLE);
      if (h_space < 0)
        error("Error while creating data space for background DM counts");
      h_err = H5Sset_extent_simple(h_space, 1, shape, shape);
      if (h_err < 0)
        error("Error while changing shape of background DM counts data space.");
      h_data = H5Dcreate(h_subgrp, "PartType2", io_hdf5_type(LONGLONG), h_space,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (h_data < 0)
        error("Error while creating dataspace for background DM counts.");
      h_err = H5Dwrite(h_data, io_hdf5_type(LONGLONG), h_space, H5S_ALL,
                       H5P_DEFAULT, count_background_gpart);
      if (h_err < 0) error("Error while writing background DM counts.");
      H5Dclose(h_data);
      H5Sclose(h_space);
    }

    if (global_counts[swift_type_stars] > 0) {

      shape[0] = nr_cells;
      shape[1] = 1;
      h_space = H5Screate(H5S_SIMPLE);
      if (h_space < 0)
        error("Error while creating data space for stars counts");
      h_err = H5Sset_extent_simple(h_space, 1, shape, shape);
      if (h_err < 0)
        error("Error while changing shape of stars counts data space.");
      h_data = H5Dcreate(h_subgrp, "PartType4", io_hdf5_type(LONGLONG), h_space,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (h_data < 0) error("Error while creating dataspace for star counts.");
      h_err = H5Dwrite(h_data, io_hdf5_type(LONGLONG), h_space, H5S_ALL,
                       H5P_DEFAULT, count_spart);
      if (h_err < 0) error("Error while writing star counts.");
      H5Dclose(h_data);
      H5Sclose(h_space);
    }

    if (global_counts[swift_type_black_hole] > 0) {

      shape[0] = nr_cells;
      shape[1] = 1;
      h_space = H5Screate(H5S_SIMPLE);
      if (h_space < 0)
        error("Error while creating data space for black hole counts");
      h_err = H5Sset_extent_simple(h_space, 1, shape, shape);
      if (h_err < 0)
        error("Error while changing shape of black hole counts data space.");
      h_data = H5Dcreate(h_subgrp, "PartType5", io_hdf5_type(LONGLONG), h_space,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (h_data < 0)
        error("Error while creating dataspace for black hole counts.");
      h_err = H5Dwrite(h_data, io_hdf5_type(LONGLONG), h_space, H5S_ALL,
                       H5P_DEFAULT, count_bpart);
      if (h_err < 0) error("Error while writing black hole counts.");
      H5Dclose(h_data);
      H5Sclose(h_space);
    }

    H5Gclose(h_subgrp);
  }

  /* Free everything we allocated */
  free(centres);
  free(count_part);
  free(count_gpart);
  free(count_background_gpart);
  free(count_spart);
  free(count_bpart);
  free(offset_part);
  free(offset_gpart);
  free(offset_background_gpart);
  free(offset_spart);
  free(offset_bpart);
}

#endif /* HAVE_HDF5 */

/**
 * @brief Returns the memory size of the data type
 */
size_t io_sizeof_type(enum IO_DATA_TYPE type) {

  switch (type) {
    case INT:
      return sizeof(int);
    case UINT:
      return sizeof(unsigned int);
    case LONG:
      return sizeof(long);
    case ULONG:
      return sizeof(unsigned long);
    case LONGLONG:
      return sizeof(long long);
    case ULONGLONG:
      return sizeof(unsigned long long);
    case FLOAT:
      return sizeof(float);
    case DOUBLE:
      return sizeof(double);
    case CHAR:
      return sizeof(char);
    default:
      error("Unknown type");
      return 0;
  }
}

/**
 * @brief Mapper function to copy #part or #gpart fields into a buffer.
 */
void io_copy_mapper(void* restrict temp, int N, void* restrict extra_data) {

  const struct io_props props = *((const struct io_props*)extra_data);
  const size_t typeSize = io_sizeof_type(props.type);
  const size_t copySize = typeSize * props.dimension;

  /* How far are we with this chunk? */
  char* restrict temp_c = (char*)temp;
  const ptrdiff_t delta = (temp_c - props.start_temp_c) / copySize;

  for (int k = 0; k < N; k++) {
    memcpy(&temp_c[k * copySize], props.field + (delta + k) * props.partSize,
           copySize);
  }
}

/**
 * @brief Mapper function to copy #part into a buffer of floats using a
 * conversion function.
 */
void io_convert_part_f_mapper(void* restrict temp, int N,
                              void* restrict extra_data) {

  const struct io_props props = *((const struct io_props*)extra_data);
  const struct part* restrict parts = props.parts;
  const struct xpart* restrict xparts = props.xparts;
  const struct engine* e = props.e;
  const size_t dim = props.dimension;

  /* How far are we with this chunk? */
  float* restrict temp_f = (float*)temp;
  const ptrdiff_t delta = (temp_f - props.start_temp_f) / dim;

  for (int i = 0; i < N; i++)
    props.convert_part_f(e, parts + delta + i, xparts + delta + i,
                         &temp_f[i * dim]);
}

/**
 * @brief Mapper function to copy #part into a buffer of ints using a
 * conversion function.
 */
void io_convert_part_i_mapper(void* restrict temp, int N,
                              void* restrict extra_data) {

  const struct io_props props = *((const struct io_props*)extra_data);
  const struct part* restrict parts = props.parts;
  const struct xpart* restrict xparts = props.xparts;
  const struct engine* e = props.e;
  const size_t dim = props.dimension;

  /* How far are we with this chunk? */
  int* restrict temp_i = (int*)temp;
  const ptrdiff_t delta = (temp_i - props.start_temp_i) / dim;

  for (int i = 0; i < N; i++)
    props.convert_part_i(e, parts + delta + i, xparts + delta + i,
                         &temp_i[i * dim]);
}

/**
 * @brief Mapper function to copy #part into a buffer of doubles using a
 * conversion function.
 */
void io_convert_part_d_mapper(void* restrict temp, int N,
                              void* restrict extra_data) {

  const struct io_props props = *((const struct io_props*)extra_data);
  const struct part* restrict parts = props.parts;
  const struct xpart* restrict xparts = props.xparts;
  const struct engine* e = props.e;
  const size_t dim = props.dimension;

  /* How far are we with this chunk? */
  double* restrict temp_d = (double*)temp;
  const ptrdiff_t delta = (temp_d - props.start_temp_d) / dim;

  for (int i = 0; i < N; i++)
    props.convert_part_d(e, parts + delta + i, xparts + delta + i,
                         &temp_d[i * dim]);
}

/**
 * @brief Mapper function to copy #part into a buffer of doubles using a
 * conversion function.
 */
void io_convert_part_l_mapper(void* restrict temp, int N,
                              void* restrict extra_data) {

  const struct io_props props = *((const struct io_props*)extra_data);
  const struct part* restrict parts = props.parts;
  const struct xpart* restrict xparts = props.xparts;
  const struct engine* e = props.e;
  const size_t dim = props.dimension;

  /* How far are we with this chunk? */
  long long* restrict temp_l = (long long*)temp;
  const ptrdiff_t delta = (temp_l - props.start_temp_l) / dim;

  for (int i = 0; i < N; i++)
    props.convert_part_l(e, parts + delta + i, xparts + delta + i,
                         &temp_l[i * dim]);
}

/**
 * @brief Mapper function to copy #gpart into a buffer of floats using a
 * conversion function.
 */
void io_convert_gpart_f_mapper(void* restrict temp, int N,
                               void* restrict extra_data) {

  const struct io_props props = *((const struct io_props*)extra_data);
  const struct gpart* restrict gparts = props.gparts;
  const struct engine* e = props.e;
  const size_t dim = props.dimension;

  /* How far are we with this chunk? */
  float* restrict temp_f = (float*)temp;
  const ptrdiff_t delta = (temp_f - props.start_temp_f) / dim;

  for (int i = 0; i < N; i++)
    props.convert_gpart_f(e, gparts + delta + i, &temp_f[i * dim]);
}

/**
 * @brief Mapper function to copy #gpart into a buffer of ints using a
 * conversion function.
 */
void io_convert_gpart_i_mapper(void* restrict temp, int N,
                               void* restrict extra_data) {

  const struct io_props props = *((const struct io_props*)extra_data);
  const struct gpart* restrict gparts = props.gparts;
  const struct engine* e = props.e;
  const size_t dim = props.dimension;

  /* How far are we with this chunk? */
  int* restrict temp_i = (int*)temp;
  const ptrdiff_t delta = (temp_i - props.start_temp_i) / dim;

  for (int i = 0; i < N; i++)
    props.convert_gpart_i(e, gparts + delta + i, &temp_i[i * dim]);
}

/**
 * @brief Mapper function to copy #gpart into a buffer of doubles using a
 * conversion function.
 */
void io_convert_gpart_d_mapper(void* restrict temp, int N,
                               void* restrict extra_data) {

  const struct io_props props = *((const struct io_props*)extra_data);
  const struct gpart* restrict gparts = props.gparts;
  const struct engine* e = props.e;
  const size_t dim = props.dimension;

  /* How far are we with this chunk? */
  double* restrict temp_d = (double*)temp;
  const ptrdiff_t delta = (temp_d - props.start_temp_d) / dim;

  for (int i = 0; i < N; i++)
    props.convert_gpart_d(e, gparts + delta + i, &temp_d[i * dim]);
}

/**
 * @brief Mapper function to copy #gpart into a buffer of doubles using a
 * conversion function.
 */
void io_convert_gpart_l_mapper(void* restrict temp, int N,
                               void* restrict extra_data) {

  const struct io_props props = *((const struct io_props*)extra_data);
  const struct gpart* restrict gparts = props.gparts;
  const struct engine* e = props.e;
  const size_t dim = props.dimension;

  /* How far are we with this chunk? */
  long long* restrict temp_l = (long long*)temp;
  const ptrdiff_t delta = (temp_l - props.start_temp_l) / dim;

  for (int i = 0; i < N; i++)
    props.convert_gpart_l(e, gparts + delta + i, &temp_l[i * dim]);
}

/**
 * @brief Mapper function to copy #spart into a buffer of floats using a
 * conversion function.
 */
void io_convert_spart_f_mapper(void* restrict temp, int N,
                               void* restrict extra_data) {

  const struct io_props props = *((const struct io_props*)extra_data);
  const struct spart* restrict sparts = props.sparts;
  const struct engine* e = props.e;
  const size_t dim = props.dimension;

  /* How far are we with this chunk? */
  float* restrict temp_f = (float*)temp;
  const ptrdiff_t delta = (temp_f - props.start_temp_f) / dim;

  for (int i = 0; i < N; i++)
    props.convert_spart_f(e, sparts + delta + i, &temp_f[i * dim]);
}

/**
 * @brief Mapper function to copy #spart into a buffer of ints using a
 * conversion function.
 */
void io_convert_spart_i_mapper(void* restrict temp, int N,
                               void* restrict extra_data) {

  const struct io_props props = *((const struct io_props*)extra_data);
  const struct spart* restrict sparts = props.sparts;
  const struct engine* e = props.e;
  const size_t dim = props.dimension;

  /* How far are we with this chunk? */
  int* restrict temp_i = (int*)temp;
  const ptrdiff_t delta = (temp_i - props.start_temp_i) / dim;

  for (int i = 0; i < N; i++)
    props.convert_spart_i(e, sparts + delta + i, &temp_i[i * dim]);
}

/**
 * @brief Mapper function to copy #spart into a buffer of doubles using a
 * conversion function.
 */
void io_convert_spart_d_mapper(void* restrict temp, int N,
                               void* restrict extra_data) {

  const struct io_props props = *((const struct io_props*)extra_data);
  const struct spart* restrict sparts = props.sparts;
  const struct engine* e = props.e;
  const size_t dim = props.dimension;

  /* How far are we with this chunk? */
  double* restrict temp_d = (double*)temp;
  const ptrdiff_t delta = (temp_d - props.start_temp_d) / dim;

  for (int i = 0; i < N; i++)
    props.convert_spart_d(e, sparts + delta + i, &temp_d[i * dim]);
}

/**
 * @brief Mapper function to copy #spart into a buffer of doubles using a
 * conversion function.
 */
void io_convert_spart_l_mapper(void* restrict temp, int N,
                               void* restrict extra_data) {

  const struct io_props props = *((const struct io_props*)extra_data);
  const struct spart* restrict sparts = props.sparts;
  const struct engine* e = props.e;
  const size_t dim = props.dimension;

  /* How far are we with this chunk? */
  long long* restrict temp_l = (long long*)temp;
  const ptrdiff_t delta = (temp_l - props.start_temp_l) / dim;

  for (int i = 0; i < N; i++)
    props.convert_spart_l(e, sparts + delta + i, &temp_l[i * dim]);
}

/**
 * @brief Mapper function to copy #bpart into a buffer of floats using a
 * conversion function.
 */
void io_convert_bpart_f_mapper(void* restrict temp, int N,
                               void* restrict extra_data) {

  const struct io_props props = *((const struct io_props*)extra_data);
  const struct bpart* restrict bparts = props.bparts;
  const struct engine* e = props.e;
  const size_t dim = props.dimension;

  /* How far are we with this chunk? */
  float* restrict temp_f = (float*)temp;
  const ptrdiff_t delta = (temp_f - props.start_temp_f) / dim;

  for (int i = 0; i < N; i++)
    props.convert_bpart_f(e, bparts + delta + i, &temp_f[i * dim]);
}

/**
 * @brief Mapper function to copy #bpart into a buffer of ints using a
 * conversion function.
 */
void io_convert_bpart_i_mapper(void* restrict temp, int N,
                               void* restrict extra_data) {

  const struct io_props props = *((const struct io_props*)extra_data);
  const struct bpart* restrict bparts = props.bparts;
  const struct engine* e = props.e;
  const size_t dim = props.dimension;

  /* How far are we with this chunk? */
  int* restrict temp_i = (int*)temp;
  const ptrdiff_t delta = (temp_i - props.start_temp_i) / dim;

  for (int i = 0; i < N; i++)
    props.convert_bpart_i(e, bparts + delta + i, &temp_i[i * dim]);
}

/**
 * @brief Mapper function to copy #bpart into a buffer of doubles using a
 * conversion function.
 */
void io_convert_bpart_d_mapper(void* restrict temp, int N,
                               void* restrict extra_data) {

  const struct io_props props = *((const struct io_props*)extra_data);
  const struct bpart* restrict bparts = props.bparts;
  const struct engine* e = props.e;
  const size_t dim = props.dimension;

  /* How far are we with this chunk? */
  double* restrict temp_d = (double*)temp;
  const ptrdiff_t delta = (temp_d - props.start_temp_d) / dim;

  for (int i = 0; i < N; i++)
    props.convert_bpart_d(e, bparts + delta + i, &temp_d[i * dim]);
}

/**
 * @brief Mapper function to copy #bpart into a buffer of doubles using a
 * conversion function.
 */
void io_convert_bpart_l_mapper(void* restrict temp, int N,
                               void* restrict extra_data) {

  const struct io_props props = *((const struct io_props*)extra_data);
  const struct bpart* restrict bparts = props.bparts;
  const struct engine* e = props.e;
  const size_t dim = props.dimension;

  /* How far are we with this chunk? */
  long long* restrict temp_l = (long long*)temp;
  const ptrdiff_t delta = (temp_l - props.start_temp_l) / dim;

  for (int i = 0; i < N; i++)
    props.convert_bpart_l(e, bparts + delta + i, &temp_l[i * dim]);
}

/**
 * @brief Copy the particle data into a temporary buffer ready for i/o.
 *
 * @param temp The buffer to be filled. Must be allocated and aligned properly.
 * @param e The #engine.
 * @param props The #io_props corresponding to the particle field we are
 * copying.
 * @param N The number of particles to copy
 * @param internal_units The system of units used internally.
 * @param snapshot_units The system of units used for the snapshots.
 */
void io_copy_temp_buffer(void* temp, const struct engine* e,
                         struct io_props props, size_t N,
                         const struct unit_system* internal_units,
                         const struct unit_system* snapshot_units) {

  const size_t typeSize = io_sizeof_type(props.type);
  const size_t copySize = typeSize * props.dimension;
  const size_t num_elements = N * props.dimension;

  /* Copy particle data to temporary buffer */
  if (props.conversion == 0) { /* No conversion */

    /* Prepare some parameters */
    char* temp_c = (char*)temp;
    props.start_temp_c = temp_c;

    /* Copy the whole thing into a buffer */
    threadpool_map((struct threadpool*)&e->threadpool, io_copy_mapper, temp_c,
                   N, copySize, 0, (void*)&props);

  } else { /* Converting particle to data */

    if (props.convert_part_f != NULL) {

      /* Prepare some parameters */
      float* temp_f = (float*)temp;
      props.start_temp_f = (float*)temp;
      props.e = e;

      /* Copy the whole thing into a buffer */
      threadpool_map((struct threadpool*)&e->threadpool,
                     io_convert_part_f_mapper, temp_f, N, copySize, 0,
                     (void*)&props);

    } else if (props.convert_part_i != NULL) {

      /* Prepare some parameters */
      int* temp_i = (int*)temp;
      props.start_temp_i = (int*)temp;
      props.e = e;

      /* Copy the whole thing into a buffer */
      threadpool_map((struct threadpool*)&e->threadpool,
                     io_convert_part_i_mapper, temp_i, N, copySize, 0,
                     (void*)&props);

    } else if (props.convert_part_d != NULL) {

      /* Prepare some parameters */
      double* temp_d = (double*)temp;
      props.start_temp_d = (double*)temp;
      props.e = e;

      /* Copy the whole thing into a buffer */
      threadpool_map((struct threadpool*)&e->threadpool,
                     io_convert_part_d_mapper, temp_d, N, copySize, 0,
                     (void*)&props);

    } else if (props.convert_part_l != NULL) {

      /* Prepare some parameters */
      long long* temp_l = (long long*)temp;
      props.start_temp_l = (long long*)temp;
      props.e = e;

      /* Copy the whole thing into a buffer */
      threadpool_map((struct threadpool*)&e->threadpool,
                     io_convert_part_l_mapper, temp_l, N, copySize, 0,
                     (void*)&props);

    } else if (props.convert_gpart_f != NULL) {

      /* Prepare some parameters */
      float* temp_f = (float*)temp;
      props.start_temp_f = (float*)temp;
      props.e = e;

      /* Copy the whole thing into a buffer */
      threadpool_map((struct threadpool*)&e->threadpool,
                     io_convert_gpart_f_mapper, temp_f, N, copySize, 0,
                     (void*)&props);

    } else if (props.convert_gpart_i != NULL) {

      /* Prepare some parameters */
      int* temp_i = (int*)temp;
      props.start_temp_i = (int*)temp;
      props.e = e;

      /* Copy the whole thing into a buffer */
      threadpool_map((struct threadpool*)&e->threadpool,
                     io_convert_gpart_i_mapper, temp_i, N, copySize, 0,
                     (void*)&props);

    } else if (props.convert_gpart_d != NULL) {

      /* Prepare some parameters */
      double* temp_d = (double*)temp;
      props.start_temp_d = (double*)temp;
      props.e = e;

      /* Copy the whole thing into a buffer */
      threadpool_map((struct threadpool*)&e->threadpool,
                     io_convert_gpart_d_mapper, temp_d, N, copySize, 0,
                     (void*)&props);

    } else if (props.convert_gpart_l != NULL) {

      /* Prepare some parameters */
      long long* temp_l = (long long*)temp;
      props.start_temp_l = (long long*)temp;
      props.e = e;

      /* Copy the whole thing into a buffer */
      threadpool_map((struct threadpool*)&e->threadpool,
                     io_convert_gpart_l_mapper, temp_l, N, copySize, 0,
                     (void*)&props);

    } else if (props.convert_spart_f != NULL) {

      /* Prepare some parameters */
      float* temp_f = (float*)temp;
      props.start_temp_f = (float*)temp;
      props.e = e;

      /* Copy the whole thing into a buffer */
      threadpool_map((struct threadpool*)&e->threadpool,
                     io_convert_spart_f_mapper, temp_f, N, copySize, 0,
                     (void*)&props);

    } else if (props.convert_spart_i != NULL) {

      /* Prepare some parameters */
      int* temp_i = (int*)temp;
      props.start_temp_i = (int*)temp;
      props.e = e;

      /* Copy the whole thing into a buffer */
      threadpool_map((struct threadpool*)&e->threadpool,
                     io_convert_spart_i_mapper, temp_i, N, copySize, 0,
                     (void*)&props);

    } else if (props.convert_spart_d != NULL) {

      /* Prepare some parameters */
      double* temp_d = (double*)temp;
      props.start_temp_d = (double*)temp;
      props.e = e;

      /* Copy the whole thing into a buffer */
      threadpool_map((struct threadpool*)&e->threadpool,
                     io_convert_spart_d_mapper, temp_d, N, copySize, 0,
                     (void*)&props);

    } else if (props.convert_spart_l != NULL) {

      /* Prepare some parameters */
      long long* temp_l = (long long*)temp;
      props.start_temp_l = (long long*)temp;
      props.e = e;

      /* Copy the whole thing into a buffer */
      threadpool_map((struct threadpool*)&e->threadpool,
                     io_convert_spart_l_mapper, temp_l, N, copySize, 0,
                     (void*)&props);

    } else if (props.convert_bpart_f != NULL) {

      /* Prepare some parameters */
      float* temp_f = (float*)temp;
      props.start_temp_f = (float*)temp;
      props.e = e;

      /* Copy the whole thing into a buffer */
      threadpool_map((struct threadpool*)&e->threadpool,
                     io_convert_bpart_f_mapper, temp_f, N, copySize, 0,
                     (void*)&props);

    } else if (props.convert_bpart_i != NULL) {

      /* Prepare some parameters */
      int* temp_i = (int*)temp;
      props.start_temp_i = (int*)temp;
      props.e = e;

      /* Copy the whole thing into a buffer */
      threadpool_map((struct threadpool*)&e->threadpool,
                     io_convert_bpart_i_mapper, temp_i, N, copySize, 0,
                     (void*)&props);

    } else if (props.convert_bpart_d != NULL) {

      /* Prepare some parameters */
      double* temp_d = (double*)temp;
      props.start_temp_d = (double*)temp;
      props.e = e;

      /* Copy the whole thing into a buffer */
      threadpool_map((struct threadpool*)&e->threadpool,
                     io_convert_bpart_d_mapper, temp_d, N, copySize, 0,
                     (void*)&props);

    } else if (props.convert_bpart_l != NULL) {

      /* Prepare some parameters */
      long long* temp_l = (long long*)temp;
      props.start_temp_l = (long long*)temp;
      props.e = e;

      /* Copy the whole thing into a buffer */
      threadpool_map((struct threadpool*)&e->threadpool,
                     io_convert_bpart_l_mapper, temp_l, N, copySize, 0,
                     (void*)&props);

    } else {
      error("Missing conversion function");
    }
  }

  /* Unit conversion if necessary */
  const double factor =
      units_conversion_factor(internal_units, snapshot_units, props.units);
  if (factor != 1.) {

    /* message("Converting ! factor=%e", factor); */

    if (io_is_double_precision(props.type)) {
      swift_declare_aligned_ptr(double, temp_d, (double*)temp,
                                IO_BUFFER_ALIGNMENT);
      for (size_t i = 0; i < num_elements; ++i) temp_d[i] *= factor;
    } else {
      swift_declare_aligned_ptr(float, temp_f, (float*)temp,
                                IO_BUFFER_ALIGNMENT);
      for (size_t i = 0; i < num_elements; ++i) temp_f[i] *= factor;
    }
  }
}

void io_prepare_dm_gparts_mapper(void* restrict data, int Ndm, void* dummy) {

  struct gpart* restrict gparts = (struct gpart*)data;

  /* Let's give all these gparts a negative id */
  for (int i = 0; i < Ndm; ++i) {

    /* Negative ids are not allowed */
    if (gparts[i].id_or_neg_offset < 0)
      error("Negative ID for DM particle %i: ID=%lld", i,
            gparts[i].id_or_neg_offset);

    /* Set gpart type */
    gparts[i].type = swift_type_dark_matter;
  }
}

/**
 * @brief Prepare the DM particles (in gparts) read in for the addition of the
 * other particle types
 *
 * This function assumes that the DM particles are all at the start of the
 * gparts array
 *
 * @param tp The current #threadpool.
 * @param gparts The array of #gpart freshly read in.
 * @param Ndm The number of DM particles read in.
 */
void io_prepare_dm_gparts(struct threadpool* tp, struct gpart* const gparts,
                          size_t Ndm) {

  threadpool_map(tp, io_prepare_dm_gparts_mapper, gparts, Ndm,
                 sizeof(struct gpart), 0, NULL);
}

void io_prepare_dm_background_gparts_mapper(void* restrict data, int Ndm,
                                            void* dummy) {

  struct gpart* restrict gparts = (struct gpart*)data;

  /* Let's give all these gparts a negative id */
  for (int i = 0; i < Ndm; ++i) {

    /* Negative ids are not allowed */
    if (gparts[i].id_or_neg_offset < 0)
      error("Negative ID for DM particle %i: ID=%lld", i,
            gparts[i].id_or_neg_offset);

    /* Set gpart type */
    gparts[i].type = swift_type_dark_matter_background;
  }
}

/**
 * @brief Prepare the DM backgorund particles (in gparts) read in
 * for the addition of the other particle types
 *
 * This function assumes that the DM particles are all at the start of the
 * gparts array and that the background particles directly follow them.
 *
 * @param tp The current #threadpool.
 * @param gparts The array of #gpart freshly read in.
 * @param Ndm The number of DM particles read in.
 */
void io_prepare_dm_background_gparts(struct threadpool* tp,
                                     struct gpart* const gparts, size_t Ndm) {

  threadpool_map(tp, io_prepare_dm_background_gparts_mapper, gparts, Ndm,
                 sizeof(struct gpart), 0, NULL);
}

size_t io_count_dm_background_gparts(const struct gpart* const gparts,
                                     const size_t Ndm) {

  swift_declare_aligned_ptr(const struct gpart, gparts_array, gparts,
                            SWIFT_STRUCT_ALIGNMENT);

  size_t count = 0;
  for (size_t i = 0; i < Ndm; ++i) {
    if (gparts_array[i].type == swift_type_dark_matter_background) ++count;
  }

  return count;
}

struct duplication_data {

  struct part* parts;
  struct gpart* gparts;
  struct spart* sparts;
  struct bpart* bparts;
  int Ndm;
  int Ngas;
  int Nstars;
  int Nblackholes;
};

void io_duplicate_hydro_gparts_mapper(void* restrict data, int Ngas,
                                      void* restrict extra_data) {

  struct duplication_data* temp = (struct duplication_data*)extra_data;
  const int Ndm = temp->Ndm;
  struct part* parts = (struct part*)data;
  const ptrdiff_t offset = parts - temp->parts;
  struct gpart* gparts = temp->gparts + offset;

  for (int i = 0; i < Ngas; ++i) {

    /* Duplicate the crucial information */
    gparts[i + Ndm].x[0] = parts[i].x[0];
    gparts[i + Ndm].x[1] = parts[i].x[1];
    gparts[i + Ndm].x[2] = parts[i].x[2];

    gparts[i + Ndm].v_full[0] = parts[i].v[0];
    gparts[i + Ndm].v_full[1] = parts[i].v[1];
    gparts[i + Ndm].v_full[2] = parts[i].v[2];

    gparts[i + Ndm].mass = hydro_get_mass(&parts[i]);

    /* Set gpart type */
    gparts[i + Ndm].type = swift_type_gas;

    /* Link the particles */
    gparts[i + Ndm].id_or_neg_offset = -(long long)(offset + i);
    parts[i].gpart = &gparts[i + Ndm];
  }
}

/**
 * @brief Copy every #part into the corresponding #gpart and link them.
 *
 * This function assumes that the DM particles are all at the start of the
 * gparts array and adds the hydro particles afterwards
 *
 * @param tp The current #threadpool.
 * @param parts The array of #part freshly read in.
 * @param gparts The array of #gpart freshly read in with all the DM particles
 * at the start
 * @param Ngas The number of gas particles read in.
 * @param Ndm The number of DM particles read in.
 */
void io_duplicate_hydro_gparts(struct threadpool* tp, struct part* const parts,
                               struct gpart* const gparts, size_t Ngas,
                               size_t Ndm) {
  struct duplication_data data;
  data.parts = parts;
  data.gparts = gparts;
  data.Ndm = Ndm;

  threadpool_map(tp, io_duplicate_hydro_gparts_mapper, parts, Ngas,
                 sizeof(struct part), 0, &data);
}

void io_duplicate_stars_gparts_mapper(void* restrict data, int Nstars,
                                      void* restrict extra_data) {

  struct duplication_data* temp = (struct duplication_data*)extra_data;
  const int Ndm = temp->Ndm;
  struct spart* sparts = (struct spart*)data;
  const ptrdiff_t offset = sparts - temp->sparts;
  struct gpart* gparts = temp->gparts + offset;

  for (int i = 0; i < Nstars; ++i) {

    /* Duplicate the crucial information */
    gparts[i + Ndm].x[0] = sparts[i].x[0];
    gparts[i + Ndm].x[1] = sparts[i].x[1];
    gparts[i + Ndm].x[2] = sparts[i].x[2];

    gparts[i + Ndm].v_full[0] = sparts[i].v[0];
    gparts[i + Ndm].v_full[1] = sparts[i].v[1];
    gparts[i + Ndm].v_full[2] = sparts[i].v[2];

    gparts[i + Ndm].mass = sparts[i].mass;

    /* Set gpart type */
    gparts[i + Ndm].type = swift_type_stars;

    /* Link the particles */
    gparts[i + Ndm].id_or_neg_offset = -(long long)(offset + i);
    sparts[i].gpart = &gparts[i + Ndm];
  }
}

/**
 * @brief Copy every #spart into the corresponding #gpart and link them.
 *
 * This function assumes that the DM particles and gas particles are all at
 * the start of the gparts array and adds the star particles afterwards
 *
 * @param tp The current #threadpool.
 * @param sparts The array of #spart freshly read in.
 * @param gparts The array of #gpart freshly read in with all the DM and gas
 * particles at the start.
 * @param Nstars The number of stars particles read in.
 * @param Ndm The number of DM and gas particles read in.
 */
void io_duplicate_stars_gparts(struct threadpool* tp,
                               struct spart* const sparts,
                               struct gpart* const gparts, size_t Nstars,
                               size_t Ndm) {

  struct duplication_data data;
  data.gparts = gparts;
  data.sparts = sparts;
  data.Ndm = Ndm;

  threadpool_map(tp, io_duplicate_stars_gparts_mapper, sparts, Nstars,
                 sizeof(struct spart), 0, &data);
}

void io_duplicate_black_holes_gparts_mapper(void* restrict data,
                                            int Nblackholes,
                                            void* restrict extra_data) {

  struct duplication_data* temp = (struct duplication_data*)extra_data;
  const int Ndm = temp->Ndm;
  struct bpart* bparts = (struct bpart*)data;
  const ptrdiff_t offset = bparts - temp->bparts;
  struct gpart* gparts = temp->gparts + offset;

  for (int i = 0; i < Nblackholes; ++i) {

    /* Duplicate the crucial information */
    gparts[i + Ndm].x[0] = bparts[i].x[0];
    gparts[i + Ndm].x[1] = bparts[i].x[1];
    gparts[i + Ndm].x[2] = bparts[i].x[2];

    gparts[i + Ndm].v_full[0] = bparts[i].v[0];
    gparts[i + Ndm].v_full[1] = bparts[i].v[1];
    gparts[i + Ndm].v_full[2] = bparts[i].v[2];

    gparts[i + Ndm].mass = bparts[i].mass;

    /* Set gpart type */
    gparts[i + Ndm].type = swift_type_black_hole;

    /* Link the particles */
    gparts[i + Ndm].id_or_neg_offset = -(long long)(offset + i);
    bparts[i].gpart = &gparts[i + Ndm];
  }
}

/**
 * @brief Copy every #bpart into the corresponding #gpart and link them.
 *
 * This function assumes that the DM particles, gas particles and star particles
 * are all at the start of the gparts array and adds the black hole particles
 * afterwards
 *
 * @param tp The current #threadpool.
 * @param bparts The array of #bpart freshly read in.
 * @param gparts The array of #gpart freshly read in with all the DM, gas
 * and star particles at the start.
 * @param Nblackholes The number of blackholes particles read in.
 * @param Ndm The number of DM, gas and star particles read in.
 */
void io_duplicate_black_holes_gparts(struct threadpool* tp,
                                     struct bpart* const bparts,
                                     struct gpart* const gparts,
                                     size_t Nblackholes, size_t Ndm) {

  struct duplication_data data;
  data.gparts = gparts;
  data.bparts = bparts;
  data.Ndm = Ndm;

  threadpool_map(tp, io_duplicate_black_holes_gparts_mapper, bparts,
                 Nblackholes, sizeof(struct bpart), 0, &data);
}

/**
 * @brief Copy every non-inhibited #part into the parts_written array.
 *
 * @param parts The array of #part containing all particles.
 * @param xparts The array of #xpart containing all particles.
 * @param parts_written The array of #part to fill with particles we want to
 * write.
 * @param xparts_written The array of #xpart  to fill with particles we want to
 * write.
 * @param Nparts The total number of #part.
 * @param Nparts_written The total number of #part to write.
 */
void io_collect_parts_to_write(const struct part* restrict parts,
                               const struct xpart* restrict xparts,
                               struct part* restrict parts_written,
                               struct xpart* restrict xparts_written,
                               const size_t Nparts,
                               const size_t Nparts_written) {

  size_t count = 0;

  /* Loop over all parts */
  for (size_t i = 0; i < Nparts; ++i) {

    /* And collect the ones that have not been removed */
    if (parts[i].time_bin != time_bin_inhibited &&
        parts[i].time_bin != time_bin_not_created) {

      parts_written[count] = parts[i];
      xparts_written[count] = xparts[i];
      count++;
    }
  }

  /* Check that everything is fine */
  if (count != Nparts_written)
    error("Collected the wrong number of particles (%zu vs. %zu expected)",
          count, Nparts_written);
}

/**
 * @brief Copy every non-inhibited #spart into the sparts_written array.
 *
 * @param sparts The array of #spart containing all particles.
 * @param sparts_written The array of #spart to fill with particles we want to
 * write.
 * @param Nsparts The total number of #part.
 * @param Nsparts_written The total number of #part to write.
 */
void io_collect_sparts_to_write(const struct spart* restrict sparts,
                                struct spart* restrict sparts_written,
                                const size_t Nsparts,
                                const size_t Nsparts_written) {

  size_t count = 0;

  /* Loop over all parts */
  for (size_t i = 0; i < Nsparts; ++i) {

    /* And collect the ones that have not been removed */
    if (sparts[i].time_bin != time_bin_inhibited &&
        sparts[i].time_bin != time_bin_not_created) {

      sparts_written[count] = sparts[i];
      count++;
    }
  }

  /* Check that everything is fine */
  if (count != Nsparts_written)
    error("Collected the wrong number of s-particles (%zu vs. %zu expected)",
          count, Nsparts_written);
}

/**
 * @brief Copy every non-inhibited #bpart into the bparts_written array.
 *
 * @param bparts The array of #bpart containing all particles.
 * @param bparts_written The array of #bpart to fill with particles we want to
 * write.
 * @param Nbparts The total number of #part.
 * @param Nbparts_written The total number of #part to write.
 */
void io_collect_bparts_to_write(const struct bpart* restrict bparts,
                                struct bpart* restrict bparts_written,
                                const size_t Nbparts,
                                const size_t Nbparts_written) {

  size_t count = 0;

  /* Loop over all parts */
  for (size_t i = 0; i < Nbparts; ++i) {

    /* And collect the ones that have not been removed */
    if (bparts[i].time_bin != time_bin_inhibited &&
        bparts[i].time_bin != time_bin_not_created) {

      bparts_written[count] = bparts[i];
      count++;
    }
  }

  /* Check that everything is fine */
  if (count != Nbparts_written)
    error("Collected the wrong number of s-particles (%zu vs. %zu expected)",
          count, Nbparts_written);
}

/**
 * @brief Copy every non-inhibited regulat DM #gpart into the gparts_written
 * array.
 *
 * @param gparts The array of #gpart containing all particles.
 * @param vr_data The array of gpart-related VELOCIraptor output.
 * @param gparts_written The array of #gpart to fill with particles we want to
 * write.
 * @param vr_data_written The array of gpart-related VELOCIraptor with particles
 * we want to write.
 * @param Ngparts The total number of #part.
 * @param Ngparts_written The total number of #part to write.
 * @param with_stf Are we running with STF? i.e. do we want to collect vr data?
 */
void io_collect_gparts_to_write(
    const struct gpart* restrict gparts,
    const struct velociraptor_gpart_data* restrict vr_data,
    struct gpart* restrict gparts_written,
    struct velociraptor_gpart_data* restrict vr_data_written,
    const size_t Ngparts, const size_t Ngparts_written, const int with_stf) {

  size_t count = 0;

  /* Loop over all parts */
  for (size_t i = 0; i < Ngparts; ++i) {

    /* And collect the ones that have not been removed */
    if ((gparts[i].time_bin != time_bin_inhibited) &&
        (gparts[i].time_bin != time_bin_not_created) &&
        (gparts[i].type == swift_type_dark_matter)) {

      if (with_stf) vr_data_written[count] = vr_data[i];

      gparts_written[count] = gparts[i];
      count++;
    }
  }

  /* Check that everything is fine */
  if (count != Ngparts_written)
    error("Collected the wrong number of g-particles (%zu vs. %zu expected)",
          count, Ngparts_written);
}

/**
 * @brief Copy every non-inhibited background DM #gpart into the gparts_written
 * array.
 *
 * @param gparts The array of #gpart containing all particles.
 * @param vr_data The array of gpart-related VELOCIraptor output.
 * @param gparts_written The array of #gpart to fill with particles we want to
 * write.
 * @param vr_data_written The array of gpart-related VELOCIraptor with particles
 * we want to write.
 * @param Ngparts The total number of #part.
 * @param Ngparts_written The total number of #part to write.
 * @param with_stf Are we running with STF? i.e. do we want to collect vr data?
 */
void io_collect_gparts_background_to_write(
    const struct gpart* restrict gparts,
    const struct velociraptor_gpart_data* restrict vr_data,
    struct gpart* restrict gparts_written,
    struct velociraptor_gpart_data* restrict vr_data_written,
    const size_t Ngparts, const size_t Ngparts_written, const int with_stf) {

  size_t count = 0;

  /* Loop over all parts */
  for (size_t i = 0; i < Ngparts; ++i) {

    /* And collect the ones that have not been removed */
    if ((gparts[i].time_bin != time_bin_inhibited) &&
        (gparts[i].time_bin != time_bin_not_created) &&
        (gparts[i].type == swift_type_dark_matter_background)) {

      if (with_stf) vr_data_written[count] = vr_data[i];

      gparts_written[count] = gparts[i];
      count++;
    }
  }

  /* Check that everything is fine */
  if (count != Ngparts_written)
    error("Collected the wrong number of g-particles (%zu vs. %zu expected)",
          count, Ngparts_written);
}

/**
 * @brief Verify the io parameter file
 *
 * @param params The #swift_params
 * @param N_total The total number of each particle type.
 */
void io_check_output_fields(const struct swift_params* params,
                            const long long N_total[swift_type_count]) {

  /* Loop over all particle types to check the fields */
  for (int ptype = 0; ptype < swift_type_count; ptype++) {

    int num_fields = 0;
    struct io_props list[100];

    /* Don't do anything if no particle of this kind */
    if (N_total[ptype] == 0) continue;

    /* Gather particle fields from the particle structures */
    switch (ptype) {

      case swift_type_gas:
        hydro_write_particles(NULL, NULL, list, &num_fields);
        num_fields += chemistry_write_particles(NULL, list + num_fields);
        num_fields +=
            cooling_write_particles(NULL, NULL, list + num_fields, NULL);
        num_fields += tracers_write_particles(NULL, NULL, list + num_fields,
                                              /*with_cosmology=*/1);
        num_fields +=
            star_formation_write_particles(NULL, NULL, list + num_fields);
        num_fields += fof_write_parts(NULL, NULL, list + num_fields);
        num_fields += velociraptor_write_parts(NULL, NULL, list + num_fields);
        break;

      case swift_type_dark_matter:
        darkmatter_write_particles(NULL, list, &num_fields);
        num_fields += fof_write_gparts(NULL, list + num_fields);
        num_fields += velociraptor_write_gparts(NULL, list + num_fields);
        break;

      case swift_type_dark_matter_background:
        darkmatter_write_particles(NULL, list, &num_fields);
        num_fields += fof_write_gparts(NULL, list + num_fields);
        num_fields += velociraptor_write_gparts(NULL, list + num_fields);
        break;

      case swift_type_stars:
        stars_write_particles(NULL, list, &num_fields, /*with_cosmology=*/1);
        num_fields += chemistry_write_sparticles(NULL, list + num_fields);
        num_fields += tracers_write_sparticles(NULL, list + num_fields,
                                               /*with_cosmology=*/1);
        num_fields += fof_write_sparts(NULL, list + num_fields);
        num_fields += velociraptor_write_sparts(NULL, list + num_fields);
        break;

      case swift_type_black_hole:
        black_holes_write_particles(NULL, list, &num_fields,
                                    /*with_cosmology=*/1);
        num_fields += chemistry_write_bparticles(NULL, list + num_fields);
        num_fields += fof_write_bparts(NULL, list + num_fields);
        num_fields += velociraptor_write_bparts(NULL, list + num_fields);
        break;

      default:
        error("Particle Type %d not yet supported. Aborting", ptype);
    }

    /* loop over each parameter */
    for (int param_id = 0; param_id < params->paramCount; param_id++) {
      const char* param_name = params->data[param_id].name;

      char section_name[PARSER_MAX_LINE_SIZE];

      /* Skip if wrong section */
      sprintf(section_name, "SelectOutput:");
      if (strstr(param_name, section_name) == NULL) continue;

      /* Skip if wrong particle type */
      sprintf(section_name, "_%s", part_type_names[ptype]);
      if (strstr(param_name, section_name) == NULL) continue;

      int found = 0;

      /* loop over each possible output field */
      for (int field_id = 0; field_id < num_fields; field_id++) {
        char field_name[PARSER_MAX_LINE_SIZE];
        sprintf(field_name, "SelectOutput:%.*s_%s", FIELD_BUFFER_SIZE,
                list[field_id].name, part_type_names[ptype]);

        if (strcmp(param_name, field_name) == 0) {
          found = 1;
          /* check if correct input */
          int retParam = 0;
          char str[PARSER_MAX_LINE_SIZE];
          sscanf(params->data[param_id].value, "%d%s", &retParam, str);

          /* Check that we have a 0 or 1 */
          if (retParam != 0 && retParam != 1)
            message(
                "WARNING: Unexpected input for %s. Received %i but expect 0 or "
                "1. ",
                field_name, retParam);

          /* Found it, so move to the next one. */
          break;
        }
      }
      if (!found)
        message(
            "WARNING: Trying to dump particle field '%s' (read from '%s') that "
            "does not exist.",
            param_name, params->fileName);
    }
  }
}

/**
 * @brief Write the output field parameters file
 *
 * @param filename The file to write.
 */
void io_write_output_field_parameter(const char* filename) {

  FILE* file = fopen(filename, "w");
  if (file == NULL) error("Error opening file '%s'", filename);

  /* Create a fake unit system for the snapshots */
  struct unit_system snapshot_units;
  units_init_cgs(&snapshot_units);

  /* Loop over all particle types */
  fprintf(file, "SelectOutput:\n");
  for (int ptype = 0; ptype < swift_type_count; ptype++) {

    int num_fields = 0;
    struct io_props list[100];

    /* Write particle fields from the particle structure */
    switch (ptype) {

      case swift_type_gas:
        hydro_write_particles(NULL, NULL, list, &num_fields);
        num_fields += chemistry_write_particles(NULL, list + num_fields);
        num_fields +=
            cooling_write_particles(NULL, NULL, list + num_fields, NULL);
        num_fields += tracers_write_particles(NULL, NULL, list + num_fields,
                                              /*with_cosmology=*/1);
        num_fields +=
            star_formation_write_particles(NULL, NULL, list + num_fields);
        num_fields += fof_write_parts(NULL, NULL, list + num_fields);
        num_fields += velociraptor_write_parts(NULL, NULL, list + num_fields);
        break;

      case swift_type_dark_matter:
        darkmatter_write_particles(NULL, list, &num_fields);
        num_fields += fof_write_gparts(NULL, list + num_fields);
        num_fields += velociraptor_write_gparts(NULL, list + num_fields);
        break;

      case swift_type_dark_matter_background:
        darkmatter_write_particles(NULL, list, &num_fields);
        num_fields += fof_write_gparts(NULL, list + num_fields);
        num_fields += velociraptor_write_gparts(NULL, list + num_fields);
        break;

      case swift_type_stars:
        stars_write_particles(NULL, list, &num_fields, /*with_cosmology=*/1);
        num_fields += chemistry_write_sparticles(NULL, list + num_fields);
        num_fields += tracers_write_sparticles(NULL, list + num_fields,
                                               /*with_cosmology=*/1);
        num_fields += fof_write_sparts(NULL, list + num_fields);
        num_fields += velociraptor_write_sparts(NULL, list + num_fields);
        break;

      case swift_type_black_hole:
        black_holes_write_particles(NULL, list, &num_fields,
                                    /*with_cosmology=*/1);
        num_fields += chemistry_write_bparticles(NULL, list + num_fields);
        num_fields += fof_write_bparts(NULL, list + num_fields);
        num_fields += velociraptor_write_bparts(NULL, list + num_fields);
        break;

      default:
        break;
    }

    if (num_fields == 0) continue;

    /* Output a header for that particle type */
    fprintf(file, "  # Particle Type %s\n", part_type_names[ptype]);

    /* Write all the fields of this particle type */
    for (int i = 0; i < num_fields; ++i) {

      char buffer[FIELD_BUFFER_SIZE] = {0};
      units_cgs_conversion_string(buffer, &snapshot_units, list[i].units,
                                  list[i].scale_factor_exponent);

      fprintf(file,
              "  %s_%s: %*d \t # %s. ::: Conversion to physical CGS: %s\n",
              list[i].name, part_type_names[ptype],
              (int)(28 - strlen(list[i].name)), 1, list[i].description, buffer);
    }

    fprintf(file, "\n");
  }

  fclose(file);

  printf(
      "List of valid ouput fields for the particle in snapshots dumped in "
      "'%s'.\n",
      filename);
}
