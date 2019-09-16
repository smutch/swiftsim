#ifndef SWIFT_DARKMATTER_WRITE_GRIDS_H
#define SWIFT_DARKMATTER_WRITE_GRIDS_H

/* Config parameters. */
#include "../config.h"

#include "engine.h"

void darkmatter_write_grids(struct engine* e, const size_t Npart, const hid_t h_file);

#endif   /* SWIFT_DARKMATTER_WRITE_GRIDS_H */
