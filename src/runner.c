/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2012 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 *                    Matthieu Schaller (matthieu.schaller@durham.ac.uk)
 *               2015 Peter W. Draper (p.w.draper@durham.ac.uk)
 *               2016 John A. Regan (john.a.regan@durham.ac.uk)
 *                    Tom Theuns (tom.theuns@durham.ac.uk)
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

/* Some standard headers. */
#include <float.h>
#include <limits.h>
#include <stdlib.h>

/* MPI headers. */
#ifdef WITH_MPI
#include <mpi.h>
#endif

/* This object's header. */
#include "runner.h"

/* Local headers. */
#include "active.h"
#include "approx_math.h"
#include "atomic.h"
#include "black_holes.h"
#include "black_holes_properties.h"
#include "cell.h"
#include "chemistry.h"
#include "const.h"
#include "cooling.h"
#include "debug.h"
#include "drift.h"
#include "engine.h"
#include "entropy_floor.h"
#include "error.h"
#include "feedback.h"
#include "gravity.h"
#include "hydro.h"
#include "hydro_properties.h"
#include "kick.h"
#include "logger.h"
#include "memuse.h"
#include "minmax.h"
#include "pressure_floor.h"
#include "pressure_floor_iact.h"
#include "scheduler.h"
#include "sort_part.h"
#include "space.h"
#include "space_getsid.h"
#include "star_formation.h"
#include "star_formation_logger.h"
#include "stars.h"
#include "task.h"
#include "timers.h"
#include "timestep.h"
#include "timestep_limiter.h"
#include "tracers.h"

/**
 * @brief Calculate gravity acceleration from external potential
 *
 * @param r runner task
 * @param c cell
 * @param timer 1 if the time is to be recorded.
 */
void runner_do_grav_external(struct runner *r, struct cell *c, int timer) {

  struct gpart *restrict gparts = c->grav.parts;
  const int gcount = c->grav.count;
  const struct engine *e = r->e;
  const struct external_potential *potential = e->external_potential;
  const struct phys_const *constants = e->physical_constants;
  const double time = r->e->time;

  TIMER_TIC;

  /* Anything to do here? */
  if (!cell_is_active_gravity(c, e)) return;

  /* Recurse? */
  if (c->split) {
    for (int k = 0; k < 8; k++)
      if (c->progeny[k] != NULL) runner_do_grav_external(r, c->progeny[k], 0);
  } else {

    /* Loop over the gparts in this cell. */
    for (int i = 0; i < gcount; i++) {

      /* Get a direct pointer on the part. */
      struct gpart *restrict gp = &gparts[i];

      /* Is this part within the time step? */
      if (gpart_is_active(gp, e)) {
        external_gravity_acceleration(time, potential, constants, gp);
      }
    }
  }

  if (timer) TIMER_TOC(timer_dograv_external);
}

/**
 * @brief Calculate gravity accelerations from the periodic mesh
 *
 * @param r runner task
 * @param c cell
 * @param timer 1 if the time is to be recorded.
 */
void runner_do_grav_mesh(struct runner *r, struct cell *c, int timer) {

  struct gpart *restrict gparts = c->grav.parts;
  const int gcount = c->grav.count;
  const struct engine *e = r->e;

#ifdef SWIFT_DEBUG_CHECKS
  if (!e->s->periodic) error("Calling mesh forces in non-periodic mode.");
#endif

  TIMER_TIC;

  /* Anything to do here? */
  if (!cell_is_active_gravity(c, e)) return;

  /* Recurse? */
  if (c->split) {
    for (int k = 0; k < 8; k++)
      if (c->progeny[k] != NULL) runner_do_grav_mesh(r, c->progeny[k], 0);
  } else {

    /* Get the forces from the gravity mesh */
    pm_mesh_interpolate_forces(e->mesh, e, gparts, gcount);
  }

  if (timer) TIMER_TOC(timer_dograv_mesh);
}

/**
 * @brief Calculate change in thermal state of particles induced
 * by radiative cooling and heating.
 *
 * @param r runner task
 * @param c cell
 * @param timer 1 if the time is to be recorded.
 */
void runner_do_cooling(struct runner *r, struct cell *c, int timer) {

  const struct engine *e = r->e;
  const struct cosmology *cosmo = e->cosmology;
  const int with_cosmology = (e->policy & engine_policy_cosmology);
  const struct cooling_function_data *cooling_func = e->cooling_func;
  const struct phys_const *constants = e->physical_constants;
  const struct unit_system *us = e->internal_units;
  const struct hydro_props *hydro_props = e->hydro_properties;
  const struct entropy_floor_properties *entropy_floor_props = e->entropy_floor;
  const double time_base = e->time_base;
  const integertime_t ti_current = e->ti_current;
  struct part *restrict parts = c->hydro.parts;
  struct xpart *restrict xparts = c->hydro.xparts;
  const int count = c->hydro.count;

  TIMER_TIC;

  /* Anything to do here? */
  if (!cell_is_active_hydro(c, e)) return;

  /* Recurse? */
  if (c->split) {
    for (int k = 0; k < 8; k++)
      if (c->progeny[k] != NULL) runner_do_cooling(r, c->progeny[k], 0);
  } else {

    /* Loop over the parts in this cell. */
    for (int i = 0; i < count; i++) {

      /* Get a direct pointer on the part. */
      struct part *restrict p = &parts[i];
      struct xpart *restrict xp = &xparts[i];

      if (part_is_active(p, e)) {

        double dt_cool, dt_therm;
        if (with_cosmology) {
          const integertime_t ti_step = get_integer_timestep(p->time_bin);
          const integertime_t ti_begin =
              get_integer_time_begin(ti_current - 1, p->time_bin);

          dt_cool =
              cosmology_get_delta_time(cosmo, ti_begin, ti_begin + ti_step);
          dt_therm = cosmology_get_therm_kick_factor(e->cosmology, ti_begin,
                                                     ti_begin + ti_step);

        } else {
          dt_cool = get_timestep(p->time_bin, time_base);
          dt_therm = get_timestep(p->time_bin, time_base);
        }

        /* Let's cool ! */
        cooling_cool_part(constants, us, cosmo, hydro_props,
                          entropy_floor_props, cooling_func, p, xp, dt_cool,
                          dt_therm);
      }
    }
  }

  if (timer) TIMER_TOC(timer_do_cooling);
}

/**
 *
 */
void runner_do_star_formation(struct runner *r, struct cell *c, int timer) {

  struct engine *e = r->e;
  const struct cosmology *cosmo = e->cosmology;
  const struct star_formation *sf_props = e->star_formation;
  const struct phys_const *phys_const = e->physical_constants;
  const int count = c->hydro.count;
  struct part *restrict parts = c->hydro.parts;
  struct xpart *restrict xparts = c->hydro.xparts;
  const int with_cosmology = (e->policy & engine_policy_cosmology);
  const int with_feedback = (e->policy & engine_policy_feedback);
  const struct hydro_props *restrict hydro_props = e->hydro_properties;
  const struct unit_system *restrict us = e->internal_units;
  struct cooling_function_data *restrict cooling = e->cooling_func;
  const struct entropy_floor_properties *entropy_floor = e->entropy_floor;
  const double time_base = e->time_base;
  const integertime_t ti_current = e->ti_current;
  const int current_stars_count = c->stars.count;

  TIMER_TIC;

#ifdef SWIFT_DEBUG_CHECKS
  if (c->nodeID != e->nodeID)
    error("Running star formation task on a foreign node!");
#endif

  /* Anything to do here? */
  if (c->hydro.count == 0 || !cell_is_active_hydro(c, e)) {
    star_formation_logger_log_inactive_cell(&c->stars.sfh);
    return;
  }

  /* Reset the SFR */
  star_formation_logger_init(&c->stars.sfh);

  /* Recurse? */
  if (c->split) {
    for (int k = 0; k < 8; k++)
      if (c->progeny[k] != NULL) {
        /* Load the child cell */
        struct cell *restrict cp = c->progeny[k];

        /* Do the recursion */
        runner_do_star_formation(r, cp, 0);

        /* Update current cell using child cells */
        star_formation_logger_add(&c->stars.sfh, &cp->stars.sfh);
      }
  } else {

    /* Loop over the gas particles in this cell. */
    for (int k = 0; k < count; k++) {

      /* Get a handle on the part. */
      struct part *restrict p = &parts[k];
      struct xpart *restrict xp = &xparts[k];

      /* Only work on active particles */
      if (part_is_active(p, e)) {

        /* Is this particle star forming? */
        if (star_formation_is_star_forming(p, xp, sf_props, phys_const, cosmo,
                                           hydro_props, us, cooling,
                                           entropy_floor)) {

          /* Time-step size for this particle */
          double dt_star;
          if (with_cosmology) {
            const integertime_t ti_step = get_integer_timestep(p->time_bin);
            const integertime_t ti_begin =
                get_integer_time_begin(ti_current - 1, p->time_bin);

            dt_star =
                cosmology_get_delta_time(cosmo, ti_begin, ti_begin + ti_step);

          } else {
            dt_star = get_timestep(p->time_bin, time_base);
          }

          /* Compute the SF rate of the particle */
          star_formation_compute_SFR(p, xp, sf_props, phys_const, cosmo,
                                     dt_star);

          /* Add the SFR and SFR*dt to the SFH struct of this cell */
          star_formation_logger_log_active_part(p, xp, &c->stars.sfh, dt_star);

          /* Are we forming a star particle from this SF rate? */
          if (star_formation_should_convert_to_star(p, xp, sf_props, e,
                                                    dt_star)) {

            /* Convert the gas particle to a star particle */
            struct spart *sp = cell_convert_part_to_spart(e, c, p, xp);

            /* Did we get a star? (Or did we run out of spare ones?) */
            if (sp != NULL) {

              /* message("We formed a star id=%lld cellID=%d", sp->id,
               * c->cellID); */

              /* Copy the properties of the gas particle to the star particle */
              star_formation_copy_properties(p, xp, sp, e, sf_props, cosmo,
                                             with_cosmology, phys_const,
                                             hydro_props, us, cooling);

              /* Update the Star formation history */
              star_formation_logger_log_new_spart(sp, &c->stars.sfh);
            }
          }

        } else { /* Are we not star-forming? */

          /* Update the particle to flag it as not star-forming */
          star_formation_update_part_not_SFR(p, xp, e, sf_props,
                                             with_cosmology);

        } /* Not Star-forming? */

      } else { /* is active? */

        /* Check if the particle is not inhibited */
        if (!part_is_inhibited(p, e)) {
          star_formation_logger_log_inactive_part(p, xp, &c->stars.sfh);
        }
      }
    } /* Loop over particles */
  }

  /* If we formed any stars, the star sorts are now invalid. We need to
   * re-compute them. */
  if (with_feedback && (c == c->top) &&
      (current_stars_count != c->stars.count)) {
    cell_set_star_resort_flag(c);
  }

  if (timer) TIMER_TOC(timer_do_star_formation);
}

/**
 * @brief Sorts again all the stars in a given cell hierarchy.
 *
 * This is intended to be used after the star formation task has been run
 * to get the cells back into a state where self/pair star tasks can be run.
 *
 * @param r The thread #runner.
 * @param c The top-level cell to run on.
 * @param timer Are we timing this?
 */
void runner_do_stars_resort(struct runner *r, struct cell *c, const int timer) {

#ifdef SWIFT_DEBUG_CHECKS
  if (c->nodeID != r->e->nodeID) error("Task must be run locally!");
#endif

  TIMER_TIC;

  /* Did we demand a recalculation of the stars'sorts? */
  if (cell_get_flag(c, cell_flag_do_stars_resort)) {
    runner_do_all_stars_sort(r, c);
    cell_clear_flag(c, cell_flag_do_stars_resort);
  }

  if (timer) TIMER_TOC(timer_do_stars_resort);
}

/**
 * @brief Sort the entries in ascending order using QuickSort.
 *
 * @param sort The entries
 * @param N The number of entries.
 */
void runner_do_sort_ascending(struct sort_entry *sort, int N) {

  struct {
    short int lo, hi;
  } qstack[10];
  int qpos, i, j, lo, hi, imin;
  struct sort_entry temp;
  float pivot;

  /* Sort parts in cell_i in decreasing order with quicksort */
  qstack[0].lo = 0;
  qstack[0].hi = N - 1;
  qpos = 0;
  while (qpos >= 0) {
    lo = qstack[qpos].lo;
    hi = qstack[qpos].hi;
    qpos -= 1;
    if (hi - lo < 15) {
      for (i = lo; i < hi; i++) {
        imin = i;
        for (j = i + 1; j <= hi; j++)
          if (sort[j].d < sort[imin].d) imin = j;
        if (imin != i) {
          temp = sort[imin];
          sort[imin] = sort[i];
          sort[i] = temp;
        }
      }
    } else {
      pivot = sort[(lo + hi) / 2].d;
      i = lo;
      j = hi;
      while (i <= j) {
        while (sort[i].d < pivot) i++;
        while (sort[j].d > pivot) j--;
        if (i <= j) {
          if (i < j) {
            temp = sort[i];
            sort[i] = sort[j];
            sort[j] = temp;
          }
          i += 1;
          j -= 1;
        }
      }
      if (j > (lo + hi) / 2) {
        if (lo < j) {
          qpos += 1;
          qstack[qpos].lo = lo;
          qstack[qpos].hi = j;
        }
        if (i < hi) {
          qpos += 1;
          qstack[qpos].lo = i;
          qstack[qpos].hi = hi;
        }
      } else {
        if (i < hi) {
          qpos += 1;
          qstack[qpos].lo = i;
          qstack[qpos].hi = hi;
        }
        if (lo < j) {
          qpos += 1;
          qstack[qpos].lo = lo;
          qstack[qpos].hi = j;
        }
      }
    }
  }
}

#ifdef SWIFT_DEBUG_CHECKS
/**
 * @brief Recursively checks that the flags are consistent in a cell hierarchy.
 *
 * Debugging function. Exists in two flavours: hydro & stars.
 */
#define RUNNER_CHECK_SORTS(TYPE)                                               \
  void runner_check_sorts_##TYPE(struct cell *c, int flags) {                  \
                                                                               \
    if (flags & ~c->TYPE.sorted) error("Inconsistent sort flags (downward)!"); \
    if (c->split)                                                              \
      for (int k = 0; k < 8; k++)                                              \
        if (c->progeny[k] != NULL && c->progeny[k]->TYPE.count > 0)            \
          runner_check_sorts_##TYPE(c->progeny[k], c->TYPE.sorted);            \
  }
#else
#define RUNNER_CHECK_SORTS(TYPE)                                       \
  void runner_check_sorts_##TYPE(struct cell *c, int flags) {          \
    error("Calling debugging code without debugging flag activated."); \
  }
#endif

RUNNER_CHECK_SORTS(hydro)
RUNNER_CHECK_SORTS(stars)

/**
 * @brief Sort the particles in the given cell along all cardinal directions.
 *
 * @param r The #runner.
 * @param c The #cell.
 * @param flags Cell flag.
 * @param cleanup If true, re-build the sorts for the selected flags instead
 *        of just adding them.
 * @param clock Flag indicating whether to record the timing or not, needed
 *      for recursive calls.
 */
void runner_do_hydro_sort(struct runner *r, struct cell *c, int flags,
                          int cleanup, int clock) {

  struct sort_entry *fingers[8];
  const int count = c->hydro.count;
  const struct part *parts = c->hydro.parts;
  struct xpart *xparts = c->hydro.xparts;
  float buff[8];

  TIMER_TIC;

#ifdef SWIFT_DEBUG_CHECKS
  if (c->hydro.super == NULL) error("Task called above the super level!!!");
#endif

  /* We need to do the local sorts plus whatever was requested further up. */
  flags |= c->hydro.do_sort;
  if (cleanup) {
    c->hydro.sorted = 0;
  } else {
    flags &= ~c->hydro.sorted;
  }
  if (flags == 0 && !cell_get_flag(c, cell_flag_do_hydro_sub_sort)) return;

  /* Check that the particles have been moved to the current time */
  if (flags && !cell_are_part_drifted(c, r->e))
    error("Sorting un-drifted cell c->nodeID=%d", c->nodeID);

#ifdef SWIFT_DEBUG_CHECKS
  /* Make sure the sort flags are consistent (downward). */
  runner_check_sorts_hydro(c, c->hydro.sorted);

  /* Make sure the sort flags are consistent (upard). */
  for (struct cell *finger = c->parent; finger != NULL;
       finger = finger->parent) {
    if (finger->hydro.sorted & ~c->hydro.sorted)
      error("Inconsistent sort flags (upward).");
  }

  /* Update the sort timer which represents the last time the sorts
     were re-set. */
  if (c->hydro.sorted == 0) c->hydro.ti_sort = r->e->ti_current;
#endif

  /* Allocate memory for sorting. */
  cell_malloc_hydro_sorts(c, flags);

  /* Does this cell have any progeny? */
  if (c->split) {

    /* Fill in the gaps within the progeny. */
    float dx_max_sort = 0.0f;
    float dx_max_sort_old = 0.0f;
    for (int k = 0; k < 8; k++) {
      if (c->progeny[k] != NULL) {

        if (c->progeny[k]->hydro.count > 0) {

          /* Only propagate cleanup if the progeny is stale. */
          runner_do_hydro_sort(
              r, c->progeny[k], flags,
              cleanup && (c->progeny[k]->hydro.dx_max_sort_old >
                          space_maxreldx * c->progeny[k]->dmin),
              0);
          dx_max_sort = max(dx_max_sort, c->progeny[k]->hydro.dx_max_sort);
          dx_max_sort_old =
              max(dx_max_sort_old, c->progeny[k]->hydro.dx_max_sort_old);
        } else {

          /* We need to clean up the unused flags that were in case the
             number of particles in the cell would change */
          cell_clear_hydro_sort_flags(c->progeny[k], /*clear_unused_flags=*/1);
        }
      }
    }
    c->hydro.dx_max_sort = dx_max_sort;
    c->hydro.dx_max_sort_old = dx_max_sort_old;

    /* Loop over the 13 different sort arrays. */
    for (int j = 0; j < 13; j++) {

      /* Has this sort array been flagged? */
      if (!(flags & (1 << j))) continue;

      /* Init the particle index offsets. */
      int off[8];
      off[0] = 0;
      for (int k = 1; k < 8; k++)
        if (c->progeny[k - 1] != NULL)
          off[k] = off[k - 1] + c->progeny[k - 1]->hydro.count;
        else
          off[k] = off[k - 1];

      /* Init the entries and indices. */
      int inds[8];
      for (int k = 0; k < 8; k++) {
        inds[k] = k;
        if (c->progeny[k] != NULL && c->progeny[k]->hydro.count > 0) {
          fingers[k] = c->progeny[k]->hydro.sort[j];
          buff[k] = fingers[k]->d;
          off[k] = off[k];
        } else
          buff[k] = FLT_MAX;
      }

      /* Sort the buffer. */
      for (int i = 0; i < 7; i++)
        for (int k = i + 1; k < 8; k++)
          if (buff[inds[k]] < buff[inds[i]]) {
            int temp_i = inds[i];
            inds[i] = inds[k];
            inds[k] = temp_i;
          }

      /* For each entry in the new sort list. */
      struct sort_entry *finger = c->hydro.sort[j];
      for (int ind = 0; ind < count; ind++) {

        /* Copy the minimum into the new sort array. */
        finger[ind].d = buff[inds[0]];
        finger[ind].i = fingers[inds[0]]->i + off[inds[0]];

        /* Update the buffer. */
        fingers[inds[0]] += 1;
        buff[inds[0]] = fingers[inds[0]]->d;

        /* Find the smallest entry. */
        for (int k = 1; k < 8 && buff[inds[k]] < buff[inds[k - 1]]; k++) {
          int temp_i = inds[k - 1];
          inds[k - 1] = inds[k];
          inds[k] = temp_i;
        }

      } /* Merge. */

      /* Add a sentinel. */
      c->hydro.sort[j][count].d = FLT_MAX;
      c->hydro.sort[j][count].i = 0;

      /* Mark as sorted. */
      atomic_or(&c->hydro.sorted, 1 << j);

    } /* loop over sort arrays. */

  } /* progeny? */

  /* Otherwise, just sort. */
  else {

    /* Reset the sort distance */
    if (c->hydro.sorted == 0) {
#ifdef SWIFT_DEBUG_CHECKS
      if (xparts != NULL && c->nodeID != engine_rank)
        error("Have non-NULL xparts in foreign cell");
#endif

      /* And the individual sort distances if we are a local cell */
      if (xparts != NULL) {
        for (int k = 0; k < count; k++) {
          xparts[k].x_diff_sort[0] = 0.0f;
          xparts[k].x_diff_sort[1] = 0.0f;
          xparts[k].x_diff_sort[2] = 0.0f;
        }
      }
      c->hydro.dx_max_sort_old = 0.f;
      c->hydro.dx_max_sort = 0.f;
    }

    /* Fill the sort array. */
    for (int k = 0; k < count; k++) {
      const double px[3] = {parts[k].x[0], parts[k].x[1], parts[k].x[2]};
      for (int j = 0; j < 13; j++)
        if (flags & (1 << j)) {
          c->hydro.sort[j][k].i = k;
          c->hydro.sort[j][k].d = px[0] * runner_shift[j][0] +
                                  px[1] * runner_shift[j][1] +
                                  px[2] * runner_shift[j][2];
        }
    }

    /* Add the sentinel and sort. */
    for (int j = 0; j < 13; j++)
      if (flags & (1 << j)) {
        c->hydro.sort[j][count].d = FLT_MAX;
        c->hydro.sort[j][count].i = 0;
        runner_do_sort_ascending(c->hydro.sort[j], count);
        atomic_or(&c->hydro.sorted, 1 << j);
      }
  }

#ifdef SWIFT_DEBUG_CHECKS
  /* Verify the sorting. */
  for (int j = 0; j < 13; j++) {
    if (!(flags & (1 << j))) continue;
    struct sort_entry *finger = c->hydro.sort[j];
    for (int k = 1; k < count; k++) {
      if (finger[k].d < finger[k - 1].d)
        error("Sorting failed, ascending array.");
      if (finger[k].i >= count) error("Sorting failed, indices borked.");
    }
  }

  /* Make sure the sort flags are consistent (downward). */
  runner_check_sorts_hydro(c, flags);

  /* Make sure the sort flags are consistent (upward). */
  for (struct cell *finger = c->parent; finger != NULL;
       finger = finger->parent) {
    if (finger->hydro.sorted & ~c->hydro.sorted)
      error("Inconsistent sort flags.");
  }
#endif

  /* Clear the cell's sort flags. */
  c->hydro.do_sort = 0;
  cell_clear_flag(c, cell_flag_do_hydro_sub_sort);
  c->hydro.requires_sorts = 0;

  if (clock) TIMER_TOC(timer_dosort);
}

/**
 * @brief Sort the stars particles in the given cell along all cardinal
 * directions.
 *
 * @param r The #runner.
 * @param c The #cell.
 * @param flags Cell flag.
 * @param cleanup If true, re-build the sorts for the selected flags instead
 *        of just adding them.
 * @param clock Flag indicating whether to record the timing or not, needed
 *      for recursive calls.
 */
void runner_do_stars_sort(struct runner *r, struct cell *c, int flags,
                          int cleanup, int clock) {

  struct sort_entry *fingers[8];
  const int count = c->stars.count;
  struct spart *sparts = c->stars.parts;
  float buff[8];

  TIMER_TIC;

#ifdef SWIFT_DEBUG_CHECKS
  if (c->hydro.super == NULL) error("Task called above the super level!!!");
#endif

  /* We need to do the local sorts plus whatever was requested further up. */
  flags |= c->stars.do_sort;
  if (cleanup) {
    c->stars.sorted = 0;
  } else {
    flags &= ~c->stars.sorted;
  }
  if (flags == 0 && !cell_get_flag(c, cell_flag_do_stars_sub_sort)) return;

  /* Check that the particles have been moved to the current time */
  if (flags && !cell_are_spart_drifted(c, r->e)) {
    error("Sorting un-drifted cell c->nodeID=%d", c->nodeID);
  }

#ifdef SWIFT_DEBUG_CHECKS
  /* Make sure the sort flags are consistent (downward). */
  runner_check_sorts_stars(c, c->stars.sorted);

  /* Make sure the sort flags are consistent (upward). */
  for (struct cell *finger = c->parent; finger != NULL;
       finger = finger->parent) {
    if (finger->stars.sorted & ~c->stars.sorted)
      error("Inconsistent sort flags (upward).");
  }

  /* Update the sort timer which represents the last time the sorts
     were re-set. */
  if (c->stars.sorted == 0) c->stars.ti_sort = r->e->ti_current;
#endif

  /* start by allocating the entry arrays in the requested dimensions. */
  cell_malloc_stars_sorts(c, flags);

  /* Does this cell have any progeny? */
  if (c->split) {

    /* Fill in the gaps within the progeny. */
    float dx_max_sort = 0.0f;
    float dx_max_sort_old = 0.0f;
    for (int k = 0; k < 8; k++) {
      if (c->progeny[k] != NULL) {

        if (c->progeny[k]->stars.count > 0) {

          /* Only propagate cleanup if the progeny is stale. */
          const int cleanup_prog =
              cleanup && (c->progeny[k]->stars.dx_max_sort_old >
                          space_maxreldx * c->progeny[k]->dmin);
          runner_do_stars_sort(r, c->progeny[k], flags, cleanup_prog, 0);
          dx_max_sort = max(dx_max_sort, c->progeny[k]->stars.dx_max_sort);
          dx_max_sort_old =
              max(dx_max_sort_old, c->progeny[k]->stars.dx_max_sort_old);
        } else {

          /* We need to clean up the unused flags that were in case the
             number of particles in the cell would change */
          cell_clear_stars_sort_flags(c->progeny[k], /*clear_unused_flags=*/1);
        }
      }
    }
    c->stars.dx_max_sort = dx_max_sort;
    c->stars.dx_max_sort_old = dx_max_sort_old;

    /* Loop over the 13 different sort arrays. */
    for (int j = 0; j < 13; j++) {

      /* Has this sort array been flagged? */
      if (!(flags & (1 << j))) continue;

      /* Init the particle index offsets. */
      int off[8];
      off[0] = 0;
      for (int k = 1; k < 8; k++)
        if (c->progeny[k - 1] != NULL)
          off[k] = off[k - 1] + c->progeny[k - 1]->stars.count;
        else
          off[k] = off[k - 1];

      /* Init the entries and indices. */
      int inds[8];
      for (int k = 0; k < 8; k++) {
        inds[k] = k;
        if (c->progeny[k] != NULL && c->progeny[k]->stars.count > 0) {
          fingers[k] = c->progeny[k]->stars.sort[j];
          buff[k] = fingers[k]->d;
          off[k] = off[k];
        } else
          buff[k] = FLT_MAX;
      }

      /* Sort the buffer. */
      for (int i = 0; i < 7; i++)
        for (int k = i + 1; k < 8; k++)
          if (buff[inds[k]] < buff[inds[i]]) {
            int temp_i = inds[i];
            inds[i] = inds[k];
            inds[k] = temp_i;
          }

      /* For each entry in the new sort list. */
      struct sort_entry *finger = c->stars.sort[j];
      for (int ind = 0; ind < count; ind++) {

        /* Copy the minimum into the new sort array. */
        finger[ind].d = buff[inds[0]];
        finger[ind].i = fingers[inds[0]]->i + off[inds[0]];

        /* Update the buffer. */
        fingers[inds[0]] += 1;
        buff[inds[0]] = fingers[inds[0]]->d;

        /* Find the smallest entry. */
        for (int k = 1; k < 8 && buff[inds[k]] < buff[inds[k - 1]]; k++) {
          int temp_i = inds[k - 1];
          inds[k - 1] = inds[k];
          inds[k] = temp_i;
        }

      } /* Merge. */

      /* Add a sentinel. */
      c->stars.sort[j][count].d = FLT_MAX;
      c->stars.sort[j][count].i = 0;

      /* Mark as sorted. */
      atomic_or(&c->stars.sorted, 1 << j);

    } /* loop over sort arrays. */

  } /* progeny? */

  /* Otherwise, just sort. */
  else {

    /* Reset the sort distance */
    if (c->stars.sorted == 0) {

      /* And the individual sort distances if we are a local cell */
      for (int k = 0; k < count; k++) {
        sparts[k].x_diff_sort[0] = 0.0f;
        sparts[k].x_diff_sort[1] = 0.0f;
        sparts[k].x_diff_sort[2] = 0.0f;
      }
      c->stars.dx_max_sort_old = 0.f;
      c->stars.dx_max_sort = 0.f;
    }

    /* Fill the sort array. */
    for (int k = 0; k < count; k++) {
      const double px[3] = {sparts[k].x[0], sparts[k].x[1], sparts[k].x[2]};
      for (int j = 0; j < 13; j++)
        if (flags & (1 << j)) {
          c->stars.sort[j][k].i = k;
          c->stars.sort[j][k].d = px[0] * runner_shift[j][0] +
                                  px[1] * runner_shift[j][1] +
                                  px[2] * runner_shift[j][2];
        }
    }

    /* Add the sentinel and sort. */
    for (int j = 0; j < 13; j++)
      if (flags & (1 << j)) {
        c->stars.sort[j][count].d = FLT_MAX;
        c->stars.sort[j][count].i = 0;
        runner_do_sort_ascending(c->stars.sort[j], count);
        atomic_or(&c->stars.sorted, 1 << j);
      }
  }

#ifdef SWIFT_DEBUG_CHECKS
  /* Verify the sorting. */
  for (int j = 0; j < 13; j++) {
    if (!(flags & (1 << j))) continue;
    struct sort_entry *finger = c->stars.sort[j];
    for (int k = 1; k < count; k++) {
      if (finger[k].d < finger[k - 1].d)
        error("Sorting failed, ascending array.");
      if (finger[k].i >= count) error("Sorting failed, indices borked.");
    }
  }

  /* Make sure the sort flags are consistent (downward). */
  runner_check_sorts_stars(c, flags);

  /* Make sure the sort flags are consistent (upward). */
  for (struct cell *finger = c->parent; finger != NULL;
       finger = finger->parent) {
    if (finger->stars.sorted & ~c->stars.sorted)
      error("Inconsistent sort flags.");
  }
#endif

  /* Clear the cell's sort flags. */
  c->stars.do_sort = 0;
  cell_clear_flag(c, cell_flag_do_stars_sub_sort);
  c->stars.requires_sorts = 0;

  if (clock) TIMER_TOC(timer_do_stars_sort);
}

/**
 * @brief Recurse into a cell until reaching the super level and call
 * the hydro sorting function there.
 *
 * This function must be called at or above the super level!
 *
 * This function will sort the particles in all 13 directions.
 *
 * @param r the #runner.
 * @param c the #cell.
 */
void runner_do_all_hydro_sort(struct runner *r, struct cell *c) {

#ifdef SWIFT_DEBUG_CHECKS
  if (c->nodeID != engine_rank) error("Function called on a foreign cell!");
#endif

  if (!cell_is_active_hydro(c, r->e)) return;

  /* Shall we sort at this level? */
  if (c->hydro.super == c) {

    /* Sort everything */
    runner_do_hydro_sort(r, c, 0x1FFF, /*cleanup=*/0, /*timer=*/0);

  } else {

#ifdef SWIFT_DEBUG_CHECKS
    if (c->hydro.super != NULL) error("Function called below the super level!");
#endif

    /* Ok, then, let's try lower */
    if (c->split) {
      for (int k = 0; k < 8; ++k) {
        if (c->progeny[k] != NULL) runner_do_all_hydro_sort(r, c->progeny[k]);
      }
    } else {
#ifdef SWIFT_DEBUG_CHECKS
      error("Reached a leaf without encountering a hydro super cell!");
#endif
    }
  }
}

/**
 * @brief Recurse into a cell until reaching the super level and call
 * the star sorting function there.
 *
 * This function must be called at or above the super level!
 *
 * This function will sort the particles in all 13 directions.
 *
 * @param r the #runner.
 * @param c the #cell.
 */
void runner_do_all_stars_sort(struct runner *r, struct cell *c) {

#ifdef SWIFT_DEBUG_CHECKS
  if (c->nodeID != engine_rank) error("Function called on a foreign cell!");
#endif

  if (!cell_is_active_stars(c, r->e) && !cell_is_active_hydro(c, r->e)) return;

  /* Shall we sort at this level? */
  if (c->hydro.super == c) {

    /* Sort everything */
    runner_do_stars_sort(r, c, 0x1FFF, /*cleanup=*/0, /*timer=*/0);

  } else {

#ifdef SWIFT_DEBUG_CHECKS
    if (c->hydro.super != NULL) error("Function called below the super level!");
#endif

    /* Ok, then, let's try lower */
    if (c->split) {
      for (int k = 0; k < 8; ++k) {
        if (c->progeny[k] != NULL) runner_do_all_stars_sort(r, c->progeny[k]);
      }
    } else {
#ifdef SWIFT_DEBUG_CHECKS
      error("Reached a leaf without encountering a hydro super cell!");
#endif
    }
  }
}

/**
 * @brief Initialize the multipoles before the gravity calculation.
 *
 * @param r The runner thread.
 * @param c The cell.
 * @param timer 1 if the time is to be recorded.
 */
void runner_do_init_grav(struct runner *r, struct cell *c, int timer) {

  const struct engine *e = r->e;

  TIMER_TIC;

#ifdef SWIFT_DEBUG_CHECKS
  if (!(e->policy & engine_policy_self_gravity))
    error("Grav-init task called outside of self-gravity calculation");
#endif

  /* Anything to do here? */
  if (!cell_is_active_gravity(c, e)) return;

  /* Reset the gravity acceleration tensors */
  gravity_field_tensors_init(&c->grav.multipole->pot, e->ti_current);

  /* Recurse? */
  if (c->split) {
    for (int k = 0; k < 8; k++) {
      if (c->progeny[k] != NULL) runner_do_init_grav(r, c->progeny[k], 0);
    }
  }

  if (timer) TIMER_TOC(timer_init_grav);
}

/**
 * @brief Drift all part in a cell.
 *
 * @param r The runner thread.
 * @param c The cell.
 * @param timer Are we timing this ?
 */
void runner_do_drift_part(struct runner *r, struct cell *c, int timer) {

  TIMER_TIC;

  cell_drift_part(c, r->e, 0);

  if (timer) TIMER_TOC(timer_drift_part);
}

/**
 * @brief Drift all gpart in a cell.
 *
 * @param r The runner thread.
 * @param c The cell.
 * @param timer Are we timing this ?
 */
void runner_do_drift_gpart(struct runner *r, struct cell *c, int timer) {

  TIMER_TIC;

  cell_drift_gpart(c, r->e, 0);

  if (timer) TIMER_TOC(timer_drift_gpart);
}

/**
 * @brief Drift all spart in a cell.
 *
 * @param r The runner thread.
 * @param c The cell.
 * @param timer Are we timing this ?
 */
void runner_do_drift_spart(struct runner *r, struct cell *c, int timer) {

  TIMER_TIC;

  cell_drift_spart(c, r->e, 0);

  if (timer) TIMER_TOC(timer_drift_spart);
}

/**
 * @brief Drift all bpart in a cell.
 *
 * @param r The runner thread.
 * @param c The cell.
 * @param timer Are we timing this ?
 */
void runner_do_drift_bpart(struct runner *r, struct cell *c, int timer) {

  TIMER_TIC;

  cell_drift_bpart(c, r->e, 0);

  if (timer) TIMER_TOC(timer_drift_bpart);
}

/**
 * @brief Perform the first half-kick on all the active particles in a cell.
 *
 * @param r The runner thread.
 * @param c The cell.
 * @param timer Are we timing this ?
 */
void runner_do_kick1(struct runner *r, struct cell *c, int timer) {

  const struct engine *e = r->e;
  const struct cosmology *cosmo = e->cosmology;
  const struct hydro_props *hydro_props = e->hydro_properties;
  const struct entropy_floor_properties *entropy_floor = e->entropy_floor;
  const int with_cosmology = (e->policy & engine_policy_cosmology);
  struct part *restrict parts = c->hydro.parts;
  struct xpart *restrict xparts = c->hydro.xparts;
  struct gpart *restrict gparts = c->grav.parts;
  struct spart *restrict sparts = c->stars.parts;
  const int count = c->hydro.count;
  const int gcount = c->grav.count;
  const int scount = c->stars.count;
  const integertime_t ti_current = e->ti_current;
  const double time_base = e->time_base;

  TIMER_TIC;

  /* Anything to do here? */
  if (!cell_is_starting_hydro(c, e) && !cell_is_starting_gravity(c, e) &&
      !cell_is_starting_stars(c, e) && !cell_is_starting_black_holes(c, e))
    return;

  /* Recurse? */
  if (c->split) {
    for (int k = 0; k < 8; k++)
      if (c->progeny[k] != NULL) runner_do_kick1(r, c->progeny[k], 0);
  } else {

    /* Loop over the parts in this cell. */
    for (int k = 0; k < count; k++) {

      /* Get a handle on the part. */
      struct part *restrict p = &parts[k];
      struct xpart *restrict xp = &xparts[k];

      /* If particle needs to be kicked */
      if (part_is_starting(p, e)) {

#ifdef SWIFT_DEBUG_CHECKS
        if (p->wakeup == time_bin_awake)
          error("Woken-up particle that has not been processed in kick1");
#endif

        /* Skip particles that have been woken up and treated by the limiter. */
        if (p->wakeup != time_bin_not_awake) continue;

        const integertime_t ti_step = get_integer_timestep(p->time_bin);
        const integertime_t ti_begin =
            get_integer_time_begin(ti_current + 1, p->time_bin);

#ifdef SWIFT_DEBUG_CHECKS
        const integertime_t ti_end = ti_begin + ti_step;

        if (ti_begin != ti_current)
          error(
              "Particle in wrong time-bin, ti_end=%lld, ti_begin=%lld, "
              "ti_step=%lld time_bin=%d wakeup=%d ti_current=%lld",
              ti_end, ti_begin, ti_step, p->time_bin, p->wakeup, ti_current);
#endif

        /* Time interval for this half-kick */
        double dt_kick_grav, dt_kick_hydro, dt_kick_therm, dt_kick_corr;
        if (with_cosmology) {
          dt_kick_hydro = cosmology_get_hydro_kick_factor(
              cosmo, ti_begin, ti_begin + ti_step / 2);
          dt_kick_grav = cosmology_get_grav_kick_factor(cosmo, ti_begin,
                                                        ti_begin + ti_step / 2);
          dt_kick_therm = cosmology_get_therm_kick_factor(
              cosmo, ti_begin, ti_begin + ti_step / 2);
          dt_kick_corr = cosmology_get_corr_kick_factor(cosmo, ti_begin,
                                                        ti_begin + ti_step / 2);
        } else {
          dt_kick_hydro = (ti_step / 2) * time_base;
          dt_kick_grav = (ti_step / 2) * time_base;
          dt_kick_therm = (ti_step / 2) * time_base;
          dt_kick_corr = (ti_step / 2) * time_base;
        }

        /* do the kick */
        kick_part(p, xp, dt_kick_hydro, dt_kick_grav, dt_kick_therm,
                  dt_kick_corr, cosmo, hydro_props, entropy_floor, ti_begin,
                  ti_begin + ti_step / 2);

        /* Update the accelerations to be used in the drift for hydro */
        if (p->gpart != NULL) {

          xp->a_grav[0] = p->gpart->a_grav[0];
          xp->a_grav[1] = p->gpart->a_grav[1];
          xp->a_grav[2] = p->gpart->a_grav[2];
        }
      }
    }

    /* Loop over the gparts in this cell. */
    for (int k = 0; k < gcount; k++) {

      /* Get a handle on the part. */
      struct gpart *restrict gp = &gparts[k];

      /* If the g-particle has no counterpart and needs to be kicked */
      if ((gp->type == swift_type_dark_matter ||
           gp->type == swift_type_dark_matter_background) &&
          gpart_is_starting(gp, e)) {

        const integertime_t ti_step = get_integer_timestep(gp->time_bin);
        const integertime_t ti_begin =
            get_integer_time_begin(ti_current + 1, gp->time_bin);

#ifdef SWIFT_DEBUG_CHECKS
        const integertime_t ti_end =
            get_integer_time_end(ti_current + 1, gp->time_bin);

        if (ti_begin != ti_current)
          error(
              "Particle in wrong time-bin, ti_end=%lld, ti_begin=%lld, "
              "ti_step=%lld time_bin=%d ti_current=%lld",
              ti_end, ti_begin, ti_step, gp->time_bin, ti_current);
#endif

        /* Time interval for this half-kick */
        double dt_kick_grav;
        if (with_cosmology) {
          dt_kick_grav = cosmology_get_grav_kick_factor(cosmo, ti_begin,
                                                        ti_begin + ti_step / 2);
        } else {
          dt_kick_grav = (ti_step / 2) * time_base;
        }

        /* do the kick */
        kick_gpart(gp, dt_kick_grav, ti_begin, ti_begin + ti_step / 2);
      }
    }

    /* Loop over the stars particles in this cell. */
    for (int k = 0; k < scount; k++) {

      /* Get a handle on the s-part. */
      struct spart *restrict sp = &sparts[k];

      /* If particle needs to be kicked */
      if (spart_is_starting(sp, e)) {

        const integertime_t ti_step = get_integer_timestep(sp->time_bin);
        const integertime_t ti_begin =
            get_integer_time_begin(ti_current + 1, sp->time_bin);

#ifdef SWIFT_DEBUG_CHECKS
        const integertime_t ti_end =
            get_integer_time_end(ti_current + 1, sp->time_bin);

        if (ti_begin != ti_current)
          error(
              "Particle in wrong time-bin, ti_end=%lld, ti_begin=%lld, "
              "ti_step=%lld time_bin=%d ti_current=%lld",
              ti_end, ti_begin, ti_step, sp->time_bin, ti_current);
#endif

        /* Time interval for this half-kick */
        double dt_kick_grav;
        if (with_cosmology) {
          dt_kick_grav = cosmology_get_grav_kick_factor(cosmo, ti_begin,
                                                        ti_begin + ti_step / 2);
        } else {
          dt_kick_grav = (ti_step / 2) * time_base;
        }

        /* do the kick */
        kick_spart(sp, dt_kick_grav, ti_begin, ti_begin + ti_step / 2);
      }
    }
  }

  if (timer) TIMER_TOC(timer_kick1);
}

/**
 * @brief Perform the second half-kick on all the active particles in a cell.
 *
 * Also prepares particles to be drifted.
 *
 * @param r The runner thread.
 * @param c The cell.
 * @param timer Are we timing this ?
 */
void runner_do_kick2(struct runner *r, struct cell *c, int timer) {

  const struct engine *e = r->e;
  const struct cosmology *cosmo = e->cosmology;
  const struct hydro_props *hydro_props = e->hydro_properties;
  const struct entropy_floor_properties *entropy_floor = e->entropy_floor;
  const int with_cosmology = (e->policy & engine_policy_cosmology);
  const int count = c->hydro.count;
  const int gcount = c->grav.count;
  const int scount = c->stars.count;
  struct part *restrict parts = c->hydro.parts;
  struct xpart *restrict xparts = c->hydro.xparts;
  struct gpart *restrict gparts = c->grav.parts;
  struct spart *restrict sparts = c->stars.parts;
  const integertime_t ti_current = e->ti_current;
  const double time_base = e->time_base;

  TIMER_TIC;

  /* Anything to do here? */
  if (!cell_is_active_hydro(c, e) && !cell_is_active_gravity(c, e) &&
      !cell_is_active_stars(c, e) && !cell_is_active_black_holes(c, e))
    return;

  /* Recurse? */
  if (c->split) {
    for (int k = 0; k < 8; k++)
      if (c->progeny[k] != NULL) runner_do_kick2(r, c->progeny[k], 0);
  } else {

    /* Loop over the particles in this cell. */
    for (int k = 0; k < count; k++) {

      /* Get a handle on the part. */
      struct part *restrict p = &parts[k];
      struct xpart *restrict xp = &xparts[k];

      /* If particle needs to be kicked */
      if (part_is_active(p, e)) {

        integertime_t ti_begin, ti_end, ti_step;

#ifdef SWIFT_DEBUG_CHECKS
        if (p->wakeup == time_bin_awake)
          error("Woken-up particle that has not been processed in kick1");
#endif

        if (p->wakeup == time_bin_not_awake) {

          /* Time-step from a regular kick */
          ti_step = get_integer_timestep(p->time_bin);
          ti_begin = get_integer_time_begin(ti_current, p->time_bin);
          ti_end = ti_begin + ti_step;

        } else {

          /* Time-step that follows a wake-up call */
          ti_begin = get_integer_time_begin(ti_current, p->wakeup);
          ti_end = get_integer_time_end(ti_current, p->time_bin);
          ti_step = ti_end - ti_begin;

          /* Reset the flag. Everything is back to normal from now on. */
          p->wakeup = time_bin_awake;
        }

#ifdef SWIFT_DEBUG_CHECKS
        if (ti_begin + ti_step != ti_current)
          error(
              "Particle in wrong time-bin, ti_begin=%lld, ti_step=%lld "
              "time_bin=%d wakeup=%d ti_current=%lld",
              ti_begin, ti_step, p->time_bin, p->wakeup, ti_current);
#endif
        /* Time interval for this half-kick */
        double dt_kick_grav, dt_kick_hydro, dt_kick_therm, dt_kick_corr;
        if (with_cosmology) {
          dt_kick_hydro = cosmology_get_hydro_kick_factor(
              cosmo, ti_begin + ti_step / 2, ti_end);
          dt_kick_grav = cosmology_get_grav_kick_factor(
              cosmo, ti_begin + ti_step / 2, ti_end);
          dt_kick_therm = cosmology_get_therm_kick_factor(
              cosmo, ti_begin + ti_step / 2, ti_end);
          dt_kick_corr = cosmology_get_corr_kick_factor(
              cosmo, ti_begin + ti_step / 2, ti_end);
        } else {
          dt_kick_hydro = (ti_end - (ti_begin + ti_step / 2)) * time_base;
          dt_kick_grav = (ti_end - (ti_begin + ti_step / 2)) * time_base;
          dt_kick_therm = (ti_end - (ti_begin + ti_step / 2)) * time_base;
          dt_kick_corr = (ti_end - (ti_begin + ti_step / 2)) * time_base;
        }

        /* Finish the time-step with a second half-kick */
        kick_part(p, xp, dt_kick_hydro, dt_kick_grav, dt_kick_therm,
                  dt_kick_corr, cosmo, hydro_props, entropy_floor,
                  ti_begin + ti_step / 2, ti_end);

#ifdef SWIFT_DEBUG_CHECKS
        /* Check that kick and the drift are synchronized */
        if (p->ti_drift != p->ti_kick) error("Error integrating part in time.");
#endif

        /* Prepare the values to be drifted */
        hydro_reset_predicted_values(p, xp, cosmo);
      }
    }

    /* Loop over the g-particles in this cell. */
    for (int k = 0; k < gcount; k++) {

      /* Get a handle on the part. */
      struct gpart *restrict gp = &gparts[k];

      /* If the g-particle has no counterpart and needs to be kicked */
      if ((gp->type == swift_type_dark_matter ||
           gp->type == swift_type_dark_matter_background) &&
          gpart_is_active(gp, e)) {

        const integertime_t ti_step = get_integer_timestep(gp->time_bin);
        const integertime_t ti_begin =
            get_integer_time_begin(ti_current, gp->time_bin);

#ifdef SWIFT_DEBUG_CHECKS
        if (ti_begin + ti_step != ti_current)
          error("Particle in wrong time-bin");
#endif

        /* Time interval for this half-kick */
        double dt_kick_grav;
        if (with_cosmology) {
          dt_kick_grav = cosmology_get_grav_kick_factor(
              cosmo, ti_begin + ti_step / 2, ti_begin + ti_step);
        } else {
          dt_kick_grav = (ti_step / 2) * time_base;
        }

        /* Finish the time-step with a second half-kick */
        kick_gpart(gp, dt_kick_grav, ti_begin + ti_step / 2,
                   ti_begin + ti_step);

#ifdef SWIFT_DEBUG_CHECKS
        /* Check that kick and the drift are synchronized */
        if (gp->ti_drift != gp->ti_kick)
          error("Error integrating g-part in time.");
#endif

        /* Prepare the values to be drifted */
        gravity_reset_predicted_values(gp);
      }
    }

    /* Loop over the particles in this cell. */
    for (int k = 0; k < scount; k++) {

      /* Get a handle on the part. */
      struct spart *restrict sp = &sparts[k];

      /* If particle needs to be kicked */
      if (spart_is_active(sp, e)) {

        const integertime_t ti_step = get_integer_timestep(sp->time_bin);
        const integertime_t ti_begin =
            get_integer_time_begin(ti_current, sp->time_bin);

#ifdef SWIFT_DEBUG_CHECKS
        if (ti_begin + ti_step != ti_current)
          error("Particle in wrong time-bin");
#endif

        /* Time interval for this half-kick */
        double dt_kick_grav;
        if (with_cosmology) {
          dt_kick_grav = cosmology_get_grav_kick_factor(
              cosmo, ti_begin + ti_step / 2, ti_begin + ti_step);
        } else {
          dt_kick_grav = (ti_step / 2) * time_base;
        }

        /* Finish the time-step with a second half-kick */
        kick_spart(sp, dt_kick_grav, ti_begin + ti_step / 2,
                   ti_begin + ti_step);

#ifdef SWIFT_DEBUG_CHECKS
        /* Check that kick and the drift are synchronized */
        if (sp->ti_drift != sp->ti_kick)
          error("Error integrating s-part in time.");
#endif

        /* Prepare the values to be drifted */
        stars_reset_predicted_values(sp);
      }
    }
  }
  if (timer) TIMER_TOC(timer_kick2);
}

/**
 * @brief Computes the next time-step of all active particles in this cell
 * and update the cell's statistics.
 *
 * @param r The runner thread.
 * @param c The cell.
 * @param timer Are we timing this ?
 */
void runner_do_timestep(struct runner *r, struct cell *c, int timer) {

  const struct engine *e = r->e;
  const integertime_t ti_current = e->ti_current;
  const int with_cosmology = (e->policy & engine_policy_cosmology);
  const int count = c->hydro.count;
  const int gcount = c->grav.count;
  const int scount = c->stars.count;
  const int bcount = c->black_holes.count;
  struct part *restrict parts = c->hydro.parts;
  struct xpart *restrict xparts = c->hydro.xparts;
  struct gpart *restrict gparts = c->grav.parts;
  struct spart *restrict sparts = c->stars.parts;
  struct bpart *restrict bparts = c->black_holes.parts;

  TIMER_TIC;

  /* Anything to do here? */
  if (!cell_is_active_hydro(c, e) && !cell_is_active_gravity(c, e) &&
      !cell_is_active_stars(c, e) && !cell_is_active_black_holes(c, e)) {
    c->hydro.updated = 0;
    c->grav.updated = 0;
    c->stars.updated = 0;
    c->black_holes.updated = 0;
    return;
  }

  int updated = 0, g_updated = 0, s_updated = 0, b_updated = 0;
  integertime_t ti_hydro_end_min = max_nr_timesteps, ti_hydro_end_max = 0,
                ti_hydro_beg_max = 0;
  integertime_t ti_gravity_end_min = max_nr_timesteps, ti_gravity_end_max = 0,
                ti_gravity_beg_max = 0;
  integertime_t ti_stars_end_min = max_nr_timesteps, ti_stars_end_max = 0,
                ti_stars_beg_max = 0;
  integertime_t ti_black_holes_end_min = max_nr_timesteps,
                ti_black_holes_end_max = 0, ti_black_holes_beg_max = 0;

  /* No children? */
  if (!c->split) {

    /* Loop over the particles in this cell. */
    for (int k = 0; k < count; k++) {

      /* Get a handle on the part. */
      struct part *restrict p = &parts[k];
      struct xpart *restrict xp = &xparts[k];

      /* If particle needs updating */
      if (part_is_active(p, e)) {

#ifdef SWIFT_DEBUG_CHECKS
        /* Current end of time-step */
        const integertime_t ti_end =
            get_integer_time_end(ti_current, p->time_bin);

        if (ti_end != ti_current)
          error("Computing time-step of rogue particle.");
#endif

        /* Get new time-step */
        const integertime_t ti_new_step = get_part_timestep(p, xp, e);

        /* Update particle */
        p->time_bin = get_time_bin(ti_new_step);
        if (p->gpart != NULL) p->gpart->time_bin = p->time_bin;

        /* Update the tracers properties */
        tracers_after_timestep(p, xp, e->internal_units, e->physical_constants,
                               with_cosmology, e->cosmology,
                               e->hydro_properties, e->cooling_func, e->time);

        /* Number of updated particles */
        updated++;
        if (p->gpart != NULL) g_updated++;

        /* What is the next sync-point ? */
        ti_hydro_end_min = min(ti_current + ti_new_step, ti_hydro_end_min);
        ti_hydro_end_max = max(ti_current + ti_new_step, ti_hydro_end_max);

        /* What is the next starting point for this cell ? */
        ti_hydro_beg_max = max(ti_current, ti_hydro_beg_max);

        if (p->gpart != NULL) {

          /* What is the next sync-point ? */
          ti_gravity_end_min =
              min(ti_current + ti_new_step, ti_gravity_end_min);
          ti_gravity_end_max =
              max(ti_current + ti_new_step, ti_gravity_end_max);

          /* What is the next starting point for this cell ? */
          ti_gravity_beg_max = max(ti_current, ti_gravity_beg_max);
        }
      }

      else { /* part is inactive */

        if (!part_is_inhibited(p, e)) {

          const integertime_t ti_end =
              get_integer_time_end(ti_current, p->time_bin);

          const integertime_t ti_beg =
              get_integer_time_begin(ti_current + 1, p->time_bin);

          /* What is the next sync-point ? */
          ti_hydro_end_min = min(ti_end, ti_hydro_end_min);
          ti_hydro_end_max = max(ti_end, ti_hydro_end_max);

          /* What is the next starting point for this cell ? */
          ti_hydro_beg_max = max(ti_beg, ti_hydro_beg_max);

          if (p->gpart != NULL) {

            /* What is the next sync-point ? */
            ti_gravity_end_min = min(ti_end, ti_gravity_end_min);
            ti_gravity_end_max = max(ti_end, ti_gravity_end_max);

            /* What is the next starting point for this cell ? */
            ti_gravity_beg_max = max(ti_beg, ti_gravity_beg_max);
          }
        }
      }
    }

    /* Loop over the g-particles in this cell. */
    for (int k = 0; k < gcount; k++) {

      /* Get a handle on the part. */
      struct gpart *restrict gp = &gparts[k];

      /* If the g-particle has no counterpart */
      if (gp->type == swift_type_dark_matter ||
          gp->type == swift_type_dark_matter_background) {

        /* need to be updated ? */
        if (gpart_is_active(gp, e)) {

#ifdef SWIFT_DEBUG_CHECKS
          /* Current end of time-step */
          const integertime_t ti_end =
              get_integer_time_end(ti_current, gp->time_bin);

          if (ti_end != ti_current)
            error("Computing time-step of rogue particle.");
#endif

          /* Get new time-step */
          const integertime_t ti_new_step = get_gpart_timestep(gp, e);

          /* Update particle */
          gp->time_bin = get_time_bin(ti_new_step);

          /* Number of updated g-particles */
          g_updated++;

          /* What is the next sync-point ? */
          ti_gravity_end_min =
              min(ti_current + ti_new_step, ti_gravity_end_min);
          ti_gravity_end_max =
              max(ti_current + ti_new_step, ti_gravity_end_max);

          /* What is the next starting point for this cell ? */
          ti_gravity_beg_max = max(ti_current, ti_gravity_beg_max);

        } else { /* gpart is inactive */

          if (!gpart_is_inhibited(gp, e)) {

            const integertime_t ti_end =
                get_integer_time_end(ti_current, gp->time_bin);

            /* What is the next sync-point ? */
            ti_gravity_end_min = min(ti_end, ti_gravity_end_min);
            ti_gravity_end_max = max(ti_end, ti_gravity_end_max);

            const integertime_t ti_beg =
                get_integer_time_begin(ti_current + 1, gp->time_bin);

            /* What is the next starting point for this cell ? */
            ti_gravity_beg_max = max(ti_beg, ti_gravity_beg_max);
          }
        }
      }
    }

    /* Loop over the star particles in this cell. */
    for (int k = 0; k < scount; k++) {

      /* Get a handle on the part. */
      struct spart *restrict sp = &sparts[k];

      /* need to be updated ? */
      if (spart_is_active(sp, e)) {

#ifdef SWIFT_DEBUG_CHECKS
        /* Current end of time-step */
        const integertime_t ti_end =
            get_integer_time_end(ti_current, sp->time_bin);

        if (ti_end != ti_current)
          error("Computing time-step of rogue particle.");
#endif
        /* Get new time-step */
        const integertime_t ti_new_step = get_spart_timestep(sp, e);

        /* Update particle */
        sp->time_bin = get_time_bin(ti_new_step);
        sp->gpart->time_bin = get_time_bin(ti_new_step);

        /* Number of updated s-particles */
        s_updated++;
        g_updated++;

        ti_stars_end_min = min(ti_current + ti_new_step, ti_stars_end_min);
        ti_stars_end_max = max(ti_current + ti_new_step, ti_stars_end_max);
        ti_gravity_end_min = min(ti_current + ti_new_step, ti_gravity_end_min);
        ti_gravity_end_max = max(ti_current + ti_new_step, ti_gravity_end_max);

        /* What is the next starting point for this cell ? */
        ti_stars_beg_max = max(ti_current, ti_stars_beg_max);
        ti_gravity_beg_max = max(ti_current, ti_gravity_beg_max);

        /* star particle is inactive but not inhibited */
      } else {

        if (!spart_is_inhibited(sp, e)) {

          const integertime_t ti_end =
              get_integer_time_end(ti_current, sp->time_bin);

          const integertime_t ti_beg =
              get_integer_time_begin(ti_current + 1, sp->time_bin);

          ti_stars_end_min = min(ti_end, ti_stars_end_min);
          ti_stars_end_max = max(ti_end, ti_stars_end_max);
          ti_gravity_end_min = min(ti_end, ti_gravity_end_min);
          ti_gravity_end_max = max(ti_end, ti_gravity_end_max);

          /* What is the next starting point for this cell ? */
          ti_stars_beg_max = max(ti_beg, ti_stars_beg_max);
          ti_gravity_beg_max = max(ti_beg, ti_gravity_beg_max);
        }
      }
    }

    /* Loop over the star particles in this cell. */
    for (int k = 0; k < bcount; k++) {

      /* Get a handle on the part. */
      struct bpart *restrict bp = &bparts[k];

      /* need to be updated ? */
      if (bpart_is_active(bp, e)) {

#ifdef SWIFT_DEBUG_CHECKS
        /* Current end of time-step */
        const integertime_t ti_end =
            get_integer_time_end(ti_current, bp->time_bin);

        if (ti_end != ti_current)
          error("Computing time-step of rogue particle.");
#endif
        /* Get new time-step */
        const integertime_t ti_new_step = get_bpart_timestep(bp, e);

        /* Update particle */
        bp->time_bin = get_time_bin(ti_new_step);
        bp->gpart->time_bin = get_time_bin(ti_new_step);

        /* Number of updated s-particles */
        b_updated++;
        g_updated++;

        ti_black_holes_end_min =
            min(ti_current + ti_new_step, ti_black_holes_end_min);
        ti_black_holes_end_max =
            max(ti_current + ti_new_step, ti_black_holes_end_max);
        ti_gravity_end_min = min(ti_current + ti_new_step, ti_gravity_end_min);
        ti_gravity_end_max = max(ti_current + ti_new_step, ti_gravity_end_max);

        /* What is the next starting point for this cell ? */
        ti_black_holes_beg_max = max(ti_current, ti_black_holes_beg_max);
        ti_gravity_beg_max = max(ti_current, ti_gravity_beg_max);

        /* star particle is inactive but not inhibited */
      } else {

        if (!bpart_is_inhibited(bp, e)) {

          const integertime_t ti_end =
              get_integer_time_end(ti_current, bp->time_bin);

          const integertime_t ti_beg =
              get_integer_time_begin(ti_current + 1, bp->time_bin);

          ti_black_holes_end_min = min(ti_end, ti_black_holes_end_min);
          ti_black_holes_end_max = max(ti_end, ti_black_holes_end_max);
          ti_gravity_end_min = min(ti_end, ti_gravity_end_min);
          ti_gravity_end_max = max(ti_end, ti_gravity_end_max);

          /* What is the next starting point for this cell ? */
          ti_black_holes_beg_max = max(ti_beg, ti_black_holes_beg_max);
          ti_gravity_beg_max = max(ti_beg, ti_gravity_beg_max);
        }
      }
    }

  } else {

    /* Loop over the progeny. */
    for (int k = 0; k < 8; k++) {
      if (c->progeny[k] != NULL) {
        struct cell *restrict cp = c->progeny[k];

        /* Recurse */
        runner_do_timestep(r, cp, 0);

        /* And aggregate */
        updated += cp->hydro.updated;
        g_updated += cp->grav.updated;
        s_updated += cp->stars.updated;
        b_updated += cp->black_holes.updated;

        ti_hydro_end_min = min(cp->hydro.ti_end_min, ti_hydro_end_min);
        ti_hydro_end_max = max(cp->hydro.ti_end_max, ti_hydro_end_max);
        ti_hydro_beg_max = max(cp->hydro.ti_beg_max, ti_hydro_beg_max);

        ti_gravity_end_min = min(cp->grav.ti_end_min, ti_gravity_end_min);
        ti_gravity_end_max = max(cp->grav.ti_end_max, ti_gravity_end_max);
        ti_gravity_beg_max = max(cp->grav.ti_beg_max, ti_gravity_beg_max);

        ti_stars_end_min = min(cp->stars.ti_end_min, ti_stars_end_min);
        ti_stars_end_max = max(cp->grav.ti_end_max, ti_stars_end_max);
        ti_stars_beg_max = max(cp->grav.ti_beg_max, ti_stars_beg_max);

        ti_black_holes_end_min =
            min(cp->black_holes.ti_end_min, ti_black_holes_end_min);
        ti_black_holes_end_max =
            max(cp->grav.ti_end_max, ti_black_holes_end_max);
        ti_black_holes_beg_max =
            max(cp->grav.ti_beg_max, ti_black_holes_beg_max);
      }
    }
  }

  /* Store the values. */
  c->hydro.updated = updated;
  c->grav.updated = g_updated;
  c->stars.updated = s_updated;
  c->black_holes.updated = b_updated;

  c->hydro.ti_end_min = ti_hydro_end_min;
  c->hydro.ti_end_max = ti_hydro_end_max;
  c->hydro.ti_beg_max = ti_hydro_beg_max;
  c->grav.ti_end_min = ti_gravity_end_min;
  c->grav.ti_end_max = ti_gravity_end_max;
  c->grav.ti_beg_max = ti_gravity_beg_max;
  c->stars.ti_end_min = ti_stars_end_min;
  c->stars.ti_end_max = ti_stars_end_max;
  c->stars.ti_beg_max = ti_stars_beg_max;
  c->black_holes.ti_end_min = ti_black_holes_end_min;
  c->black_holes.ti_end_max = ti_black_holes_end_max;
  c->black_holes.ti_beg_max = ti_black_holes_beg_max;

#ifdef SWIFT_DEBUG_CHECKS
  if (c->hydro.ti_end_min == e->ti_current &&
      c->hydro.ti_end_min < max_nr_timesteps)
    error("End of next hydro step is current time!");
  if (c->grav.ti_end_min == e->ti_current &&
      c->grav.ti_end_min < max_nr_timesteps)
    error("End of next gravity step is current time!");
  if (c->stars.ti_end_min == e->ti_current &&
      c->stars.ti_end_min < max_nr_timesteps)
    error("End of next stars step is current time!");
  if (c->black_holes.ti_end_min == e->ti_current &&
      c->black_holes.ti_end_min < max_nr_timesteps)
    error("End of next black holes step is current time!");
#endif

  if (timer) TIMER_TOC(timer_timestep);
}

/**
 * @brief Apply the time-step limiter to all awaken particles in a cell
 * hierarchy.
 *
 * @param r The task #runner.
 * @param c The #cell.
 * @param force Limit the particles irrespective of the #cell flags.
 * @param timer Are we timing this ?
 */
void runner_do_limiter(struct runner *r, struct cell *c, int force, int timer) {

  const struct engine *e = r->e;
  const integertime_t ti_current = e->ti_current;
  const int count = c->hydro.count;
  struct part *restrict parts = c->hydro.parts;
  struct xpart *restrict xparts = c->hydro.xparts;

  TIMER_TIC;

#ifdef SWIFT_DEBUG_CHECKS
  /* Check that we only limit local cells. */
  if (c->nodeID != engine_rank) error("Limiting dt of a foreign cell is nope.");
#endif

  integertime_t ti_hydro_end_min = max_nr_timesteps, ti_hydro_end_max = 0,
                ti_hydro_beg_max = 0;
  integertime_t ti_gravity_end_min = max_nr_timesteps, ti_gravity_end_max = 0,
                ti_gravity_beg_max = 0;

  /* Limit irrespective of cell flags? */
  force = (force || cell_get_flag(c, cell_flag_do_hydro_limiter));

  /* Early abort? */
  if (c->hydro.count == 0) {

    /* Clear the limiter flags. */
    cell_clear_flag(
        c, cell_flag_do_hydro_limiter | cell_flag_do_hydro_sub_limiter);
    return;
  }

  /* Loop over the progeny ? */
  if (c->split && (force || cell_get_flag(c, cell_flag_do_hydro_sub_limiter))) {
    for (int k = 0; k < 8; k++) {
      if (c->progeny[k] != NULL) {
        struct cell *restrict cp = c->progeny[k];

        /* Recurse */
        runner_do_limiter(r, cp, force, 0);

        /* And aggregate */
        ti_hydro_end_min = min(cp->hydro.ti_end_min, ti_hydro_end_min);
        ti_hydro_end_max = max(cp->hydro.ti_end_max, ti_hydro_end_max);
        ti_hydro_beg_max = max(cp->hydro.ti_beg_max, ti_hydro_beg_max);
        ti_gravity_end_min = min(cp->grav.ti_end_min, ti_gravity_end_min);
        ti_gravity_end_max = max(cp->grav.ti_end_max, ti_gravity_end_max);
        ti_gravity_beg_max = max(cp->grav.ti_beg_max, ti_gravity_beg_max);
      }
    }

    /* Store the updated values */
    c->hydro.ti_end_min = min(c->hydro.ti_end_min, ti_hydro_end_min);
    c->hydro.ti_end_max = max(c->hydro.ti_end_max, ti_hydro_end_max);
    c->hydro.ti_beg_max = max(c->hydro.ti_beg_max, ti_hydro_beg_max);
    c->grav.ti_end_min = min(c->grav.ti_end_min, ti_gravity_end_min);
    c->grav.ti_end_max = max(c->grav.ti_end_max, ti_gravity_end_max);
    c->grav.ti_beg_max = max(c->grav.ti_beg_max, ti_gravity_beg_max);

  } else if (!c->split && force) {

    ti_hydro_end_min = c->hydro.ti_end_min;
    ti_hydro_end_max = c->hydro.ti_end_max;
    ti_hydro_beg_max = c->hydro.ti_beg_max;
    ti_gravity_end_min = c->grav.ti_end_min;
    ti_gravity_end_max = c->grav.ti_end_max;
    ti_gravity_beg_max = c->grav.ti_beg_max;

    /* Loop over the gas particles in this cell. */
    for (int k = 0; k < count; k++) {

      /* Get a handle on the part. */
      struct part *restrict p = &parts[k];
      struct xpart *restrict xp = &xparts[k];

      /* Avoid inhibited particles */
      if (part_is_inhibited(p, e)) continue;

      /* If the particle will be active no need to wake it up */
      if (part_is_active(p, e) && p->wakeup != time_bin_not_awake)
        p->wakeup = time_bin_not_awake;

      /* Bip, bip, bip... wake-up time */
      if (p->wakeup <= time_bin_awake) {

        /* Apply the limiter and get the new time-step size */
        const integertime_t ti_new_step = timestep_limit_part(p, xp, e);

        /* What is the next sync-point ? */
        ti_hydro_end_min = min(ti_current + ti_new_step, ti_hydro_end_min);
        ti_hydro_end_max = max(ti_current + ti_new_step, ti_hydro_end_max);

        /* What is the next starting point for this cell ? */
        ti_hydro_beg_max = max(ti_current, ti_hydro_beg_max);

        /* Also limit the gpart counter-part */
        if (p->gpart != NULL) {

          /* Register the time-bin */
          p->gpart->time_bin = p->time_bin;

          /* What is the next sync-point ? */
          ti_gravity_end_min =
              min(ti_current + ti_new_step, ti_gravity_end_min);
          ti_gravity_end_max =
              max(ti_current + ti_new_step, ti_gravity_end_max);

          /* What is the next starting point for this cell ? */
          ti_gravity_beg_max = max(ti_current, ti_gravity_beg_max);
        }
      }
    }

    /* Store the updated values */
    c->hydro.ti_end_min = min(c->hydro.ti_end_min, ti_hydro_end_min);
    c->hydro.ti_end_max = max(c->hydro.ti_end_max, ti_hydro_end_max);
    c->hydro.ti_beg_max = max(c->hydro.ti_beg_max, ti_hydro_beg_max);
    c->grav.ti_end_min = min(c->grav.ti_end_min, ti_gravity_end_min);
    c->grav.ti_end_max = max(c->grav.ti_end_max, ti_gravity_end_max);
    c->grav.ti_beg_max = max(c->grav.ti_beg_max, ti_gravity_beg_max);
  }

  /* Clear the limiter flags. */
  cell_clear_flag(c,
                  cell_flag_do_hydro_limiter | cell_flag_do_hydro_sub_limiter);

  if (timer) TIMER_TOC(timer_do_limiter);
}

/**
 * @brief End the hydro force calculation of all active particles in a cell
 * by multiplying the acccelerations by the relevant constants
 *
 * @param r The #runner thread.
 * @param c The #cell.
 * @param timer Are we timing this ?
 */
void runner_do_end_hydro_force(struct runner *r, struct cell *c, int timer) {

  const struct engine *e = r->e;

  TIMER_TIC;

  /* Anything to do here? */
  if (!cell_is_active_hydro(c, e)) return;

  /* Recurse? */
  if (c->split) {
    for (int k = 0; k < 8; k++)
      if (c->progeny[k] != NULL) runner_do_end_hydro_force(r, c->progeny[k], 0);
  } else {

    const struct cosmology *cosmo = e->cosmology;
    const int count = c->hydro.count;
    struct part *restrict parts = c->hydro.parts;

    /* Loop over the gas particles in this cell. */
    for (int k = 0; k < count; k++) {

      /* Get a handle on the part. */
      struct part *restrict p = &parts[k];

      if (part_is_active(p, e)) {

        /* Finish the force loop */
        hydro_end_force(p, cosmo);
        chemistry_end_force(p, cosmo);

#ifdef SWIFT_BOUNDARY_PARTICLES

        /* Get the ID of the part */
        const long long id = p->id;

        /* Cancel hdyro forces of these particles */
        if (id < SWIFT_BOUNDARY_PARTICLES) {

          /* Don't move ! */
          hydro_reset_acceleration(p);

#if defined(GIZMO_MFV_SPH) || defined(GIZMO_MFM_SPH)

          /* Some values need to be reset in the Gizmo case. */
          hydro_prepare_force(p, &c->hydro.xparts[k], cosmo,
                              e->hydro_properties, 0);
#endif
        }
#endif
      }
    }
  }

  if (timer) TIMER_TOC(timer_end_hydro_force);
}

/**
 * @brief End the gravity force calculation of all active particles in a cell
 * by multiplying the acccelerations by the relevant constants
 *
 * @param r The #runner thread.
 * @param c The #cell.
 * @param timer Are we timing this ?
 */
void runner_do_end_grav_force(struct runner *r, struct cell *c, int timer) {

  const struct engine *e = r->e;

  TIMER_TIC;

  /* Anything to do here? */
  if (!cell_is_active_gravity(c, e)) return;

  /* Recurse? */
  if (c->split) {
    for (int k = 0; k < 8; k++)
      if (c->progeny[k] != NULL) runner_do_end_grav_force(r, c->progeny[k], 0);
  } else {

    const struct space *s = e->s;
    const int periodic = s->periodic;
    const float G_newton = e->physical_constants->const_newton_G;

    /* Potential normalisation in the case of periodic gravity */
    float potential_normalisation = 0.;
    if (periodic && (e->policy & engine_policy_self_gravity)) {
      const double volume = s->dim[0] * s->dim[1] * s->dim[2];
      const double r_s = e->mesh->r_s;
      potential_normalisation = 4. * M_PI * e->total_mass * r_s * r_s / volume;
    }

    const int gcount = c->grav.count;
    struct gpart *restrict gparts = c->grav.parts;

    /* Loop over the g-particles in this cell. */
    for (int k = 0; k < gcount; k++) {

      /* Get a handle on the gpart. */
      struct gpart *restrict gp = &gparts[k];

      if (gpart_is_active(gp, e)) {

        /* Finish the force calculation */
        gravity_end_force(gp, G_newton, potential_normalisation, periodic);

#ifdef SWIFT_MAKE_GRAVITY_GLASS

        /* Negate the gravity forces */
        gp->a_grav[0] *= -1.f;
        gp->a_grav[1] *= -1.f;
        gp->a_grav[2] *= -1.f;
#endif

#ifdef SWIFT_NO_GRAVITY_BELOW_ID

        /* Get the ID of the gpart */
        long long id = 0;
        if (gp->type == swift_type_gas)
          id = e->s->parts[-gp->id_or_neg_offset].id;
        else if (gp->type == swift_type_stars)
          id = e->s->sparts[-gp->id_or_neg_offset].id;
        else if (gp->type == swift_type_black_hole)
          error("Unexisting type");
        else
          id = gp->id_or_neg_offset;

        /* Cancel gravity forces of these particles */
        if (id < SWIFT_NO_GRAVITY_BELOW_ID) {

          /* Don't move ! */
          gp->a_grav[0] = 0.f;
          gp->a_grav[1] = 0.f;
          gp->a_grav[2] = 0.f;
        }
#endif

#ifdef SWIFT_DEBUG_CHECKS
        if ((e->policy & engine_policy_self_gravity) &&
            !(e->policy & engine_policy_black_holes)) {

          /* Let's add a self interaction to simplify the count */
          gp->num_interacted++;

          /* Check that this gpart has interacted with all the other
           * particles (via direct or multipoles) in the box */
          if (gp->num_interacted !=
              e->total_nr_gparts - e->count_inhibited_gparts) {

            /* Get the ID of the gpart */
            long long my_id = 0;
            if (gp->type == swift_type_gas)
              my_id = e->s->parts[-gp->id_or_neg_offset].id;
            else if (gp->type == swift_type_stars)
              my_id = e->s->sparts[-gp->id_or_neg_offset].id;
            else if (gp->type == swift_type_black_hole)
              error("Unexisting type");
            else
              my_id = gp->id_or_neg_offset;

            error(
                "g-particle (id=%lld, type=%s) did not interact "
                "gravitationally with all other gparts "
                "gp->num_interacted=%lld, total_gparts=%lld (local "
                "num_gparts=%zd inhibited_gparts=%lld)",
                my_id, part_type_names[gp->type], gp->num_interacted,
                e->total_nr_gparts, e->s->nr_gparts, e->count_inhibited_gparts);
          }
        }
#endif
      }
    }
  }
  if (timer) TIMER_TOC(timer_end_grav_force);
}

/**
 * @brief Process all the gas particles in a cell that have been flagged for
 * swallowing by a black hole.
 *
 * This is done by recursing down to the leaf-level and skipping the sub-cells
 * that have not been drifted as they would not have any particles with
 * swallowing flag. We then loop over the particles with a flag and look into
 * the space-wide list of black holes for the particle with the corresponding
 * ID. If found, the BH swallows the gas particle and the gas particle is
 * removed. If the cell is local, we may be looking for a foreign BH, in which
 * case, we do not update the BH (that will be done on its node) but just remove
 * the gas particle.
 *
 * @param r The thread #runner.
 * @param c The #cell.
 * @param timer Are we timing this?
 */
void runner_do_gas_swallow(struct runner *r, struct cell *c, int timer) {

  struct engine *e = r->e;
  struct space *s = e->s;
  struct bpart *bparts = s->bparts;
  const size_t nr_bpart = s->nr_bparts;
#ifdef WITH_MPI
  struct bpart *bparts_foreign = s->bparts_foreign;
  const size_t nr_bparts_foreign = s->nr_bparts_foreign;
#endif

  struct part *parts = c->hydro.parts;
  struct xpart *xparts = c->hydro.xparts;

  /* Early abort?
   * (We only want cells for which we drifted the gas as these are
   * the only ones that could have gas particles that have been flagged
   * for swallowing) */
  if (c->hydro.count == 0 || c->hydro.ti_old_part != e->ti_current) {
    return;
  }

  /* Loop over the progeny ? */
  if (c->split) {
    for (int k = 0; k < 8; k++) {
      if (c->progeny[k] != NULL) {
        struct cell *restrict cp = c->progeny[k];

        runner_do_gas_swallow(r, cp, 0);
      }
    }
  } else {

    /* Loop over all the gas particles in the cell
     * Note that the cell (and hence the parts) may be local or foreign. */
    const size_t nr_parts = c->hydro.count;
    for (size_t k = 0; k < nr_parts; k++) {

      /* Get a handle on the part. */
      struct part *const p = &parts[k];
      struct xpart *const xp = &xparts[k];

      /* Ignore inhibited particles (they have already been removed!) */
      if (part_is_inhibited(p, e)) continue;

      /* Get the ID of the black holes that will swallow this part */
      const long long swallow_id =
          black_holes_get_part_swallow_id(&p->black_holes_data);

      /* Has this particle been flagged for swallowing? */
      if (swallow_id >= 0) {

#ifdef SWIFT_DEBUG_CHECKS
        if (p->ti_drift != e->ti_current)
          error("Trying to swallow an un-drifted particle.");
#endif

        /* ID of the BH swallowing this particle */
        const long long BH_id = swallow_id;

        /* Have we found this particle's BH already? */
        int found = 0;

        /* Let's look for the hungry black hole in the local list */
        for (size_t i = 0; i < nr_bpart; ++i) {

          /* Get a handle on the bpart. */
          struct bpart *bp = &bparts[i];

          if (bp->id == BH_id) {

            /* Lock the space as we are going to work directly on the bpart list
             */
            lock_lock(&s->lock);

            /* Swallow the gas particle (i.e. update the BH properties) */
            black_holes_swallow_part(bp, p, xp, e->cosmology);

            /* Release the space as we are done updating the bpart */
            if (lock_unlock(&s->lock) != 0)
              error("Failed to unlock the space.");

            message("BH %lld swallowing gas particle %lld", bp->id, p->id);

            /* If the gas particle is local, remove it */
            if (c->nodeID == e->nodeID) {

              message("BH %lld removing gas particle %lld", bp->id, p->id);

              lock_lock(&e->s->lock);

              /* Re-check that the particle has not been removed
               * by another thread before we do the deed. */
              if (!part_is_inhibited(p, e)) {

                /* Finally, remove the gas particle from the system
                 * Recall that the gpart associated with it is also removed
                 * at the same time. */
                cell_remove_part(e, c, p, xp);
              }

              if (lock_unlock(&e->s->lock) != 0)
                error("Failed to unlock the space!");
            }

            /* In any case, prevent the particle from being re-swallowed */
            black_holes_mark_part_as_swallowed(&p->black_holes_data);

            found = 1;
            break;
          }

        } /* Loop over local BHs */

#ifdef WITH_MPI

        /* We could also be in the case of a local gas particle being
         * swallowed by a foreign BH. In this case, we won't update the
         * BH but just remove the particle from the local list. */
        if (c->nodeID == e->nodeID && !found) {

          /* Let's look for the foreign hungry black hole */
          for (size_t i = 0; i < nr_bparts_foreign; ++i) {

            /* Get a handle on the bpart. */
            struct bpart *bp = &bparts_foreign[i];

            if (bp->id == BH_id) {

              message("BH %lld removing gas particle %lld (foreign BH case)",
                      bp->id, p->id);

              lock_lock(&e->s->lock);

              /* Re-check that the particle has not been removed
               * by another thread before we do the deed. */
              if (!part_is_inhibited(p, e)) {

                /* Finally, remove the gas particle from the system */
                cell_remove_part(e, c, p, xp);
              }

              if (lock_unlock(&e->s->lock) != 0)
                error("Failed to unlock the space!");

              found = 1;
              break;
            }
          } /* Loop over foreign BHs */
        }   /* Is the cell local? */
#endif

        /* If we have a local particle, we must have found the BH in one
         * of our list of black holes. */
        if (c->nodeID == e->nodeID && !found) {
          error("Gas particle %lld could not find BH %lld to be swallowed",
                p->id, swallow_id);
        }
      } /* Part was flagged for swallowing */
    }   /* Loop over the parts */
  }     /* Cell is not split */
}

/**
 * @brief Processing of gas particles to swallow - self task case.
 *
 * @param r The thread #runner.
 * @param c The #cell.
 * @param timer Are we timing this?
 */
void runner_do_gas_swallow_self(struct runner *r, struct cell *c, int timer) {

#ifdef SWIFT_DEBUG_CHECKS
  if (c->nodeID != r->e->nodeID) error("Running self task on foreign node");
  if (!cell_is_active_black_holes(c, r->e))
    error("Running self task on inactive cell");
#endif

  runner_do_gas_swallow(r, c, timer);
}

/**
 * @brief Processing of gas particles to swallow - pair task case.
 *
 * @param r The thread #runner.
 * @param ci First #cell.
 * @param cj Second #cell.
 * @param timer Are we timing this?
 */
void runner_do_gas_swallow_pair(struct runner *r, struct cell *ci,
                                struct cell *cj, int timer) {

  const struct engine *e = r->e;

#ifdef SWIFT_DEBUG_CHECKS
  if (ci->nodeID != e->nodeID && cj->nodeID != e->nodeID)
    error("Running pair task on foreign node");
#endif

  /* Run the swallowing loop only in the cell that is the neighbour of the
   * active BH */
  if (cell_is_active_black_holes(cj, e)) runner_do_gas_swallow(r, ci, timer);
  if (cell_is_active_black_holes(ci, e)) runner_do_gas_swallow(r, cj, timer);
}

/**
 * @brief Process all the BH particles in a cell that have been flagged for
 * swallowing by a black hole.
 *
 * This is done by recursing down to the leaf-level and skipping the sub-cells
 * that have not been drifted as they would not have any particles with
 * swallowing flag. We then loop over the particles with a flag and look into
 * the space-wide list of black holes for the particle with the corresponding
 * ID. If found, the BH swallows the BH particle and the BH particle is
 * removed. If the cell is local, we may be looking for a foreign BH, in which
 * case, we do not update the BH (that will be done on its node) but just remove
 * the BH particle.
 *
 * @param r The thread #runner.
 * @param c The #cell.
 * @param timer Are we timing this?
 */
void runner_do_bh_swallow(struct runner *r, struct cell *c, int timer) {

  struct engine *e = r->e;
  struct space *s = e->s;
  struct bpart *bparts = s->bparts;
  const size_t nr_bpart = s->nr_bparts;
#ifdef WITH_MPI
  struct bpart *bparts_foreign = s->bparts_foreign;
  const size_t nr_bparts_foreign = s->nr_bparts_foreign;
#endif

  struct bpart *cell_bparts = c->black_holes.parts;

  /* Early abort?
   * (We only want cells for which we drifted the BH as these are
   * the only ones that could have BH particles that have been flagged
   * for swallowing) */
  if (c->black_holes.count == 0 ||
      c->black_holes.ti_old_part != e->ti_current) {
    return;
  }

  /* Loop over the progeny ? */
  if (c->split) {
    for (int k = 0; k < 8; k++) {
      if (c->progeny[k] != NULL) {
        struct cell *restrict cp = c->progeny[k];

        runner_do_bh_swallow(r, cp, 0);
      }
    }
  } else {

    /* Loop over all the gas particles in the cell
     * Note that the cell (and hence the bparts) may be local or foreign. */
    const size_t nr_cell_bparts = c->black_holes.count;
    for (size_t k = 0; k < nr_cell_bparts; k++) {

      /* Get a handle on the part. */
      struct bpart *const cell_bp = &cell_bparts[k];

      /* Ignore inhibited particles (they have already been removed!) */
      if (bpart_is_inhibited(cell_bp, e)) continue;

      /* Get the ID of the black holes that will swallow this part */
      const long long swallow_id =
          black_holes_get_bpart_swallow_id(&cell_bp->merger_data);

      /* message("OO id=%lld swallow_id = %lld", cell_bp->id, */
      /* 	      swallow_id); */

      /* Has this particle been flagged for swallowing? */
      if (swallow_id >= 0) {

#ifdef SWIFT_DEBUG_CHECKS
        if (cell_bp->ti_drift != e->ti_current)
          error("Trying to swallow an un-drifted particle.");
#endif

        /* ID of the BH swallowing this particle */
        const long long BH_id = swallow_id;

        /* Have we found this particle's BH already? */
        int found = 0;

        /* Let's look for the hungry black hole in the local list */
        for (size_t i = 0; i < nr_bpart; ++i) {

          /* Get a handle on the bpart. */
          struct bpart *bp = &bparts[i];

          if (bp->id == BH_id) {

            /* Lock the space as we are going to work directly on the bpart list
             */
            lock_lock(&s->lock);

            /* Swallow the gas particle (i.e. update the BH properties) */
            black_holes_swallow_bpart(bp, cell_bp, e->cosmology);

            /* Release the space as we are done updating the bpart */
            if (lock_unlock(&s->lock) != 0)
              error("Failed to unlock the space.");

            message("BH %lld swallowing BH particle %lld", bp->id, cell_bp->id);

            /* If the gas particle is local, remove it */
            if (c->nodeID == e->nodeID) {

              message("BH %lld removing BH particle %lld", bp->id, cell_bp->id);

              /* Finally, remove the gas particle from the system
               * Recall that the gpart associated with it is also removed
               * at the same time. */
              cell_remove_bpart(e, c, cell_bp);
            }

            /* In any case, prevent the particle from being re-swallowed */
            black_holes_mark_bpart_as_merged(&cell_bp->merger_data);

            found = 1;
            break;
          }

        } /* Loop over local BHs */

#ifdef WITH_MPI

        /* We could also be in the case of a local BH particle being
         * swallowed by a foreign BH. In this case, we won't update the
         * foreign BH but just remove the particle from the local list. */
        if (c->nodeID == e->nodeID && !found) {

          /* Let's look for the foreign hungry black hole */
          for (size_t i = 0; i < nr_bparts_foreign; ++i) {

            /* Get a handle on the bpart. */
            struct bpart *bp = &bparts_foreign[i];

            if (bp->id == BH_id) {

              message("BH %lld removing BH particle %lld (foreign BH case)",
                      bp->id, cell_bp->id);

              /* Finally, remove the gas particle from the system */
              cell_remove_bpart(e, c, cell_bp);

              found = 1;
              break;
            }
          } /* Loop over foreign BHs */
        }   /* Is the cell local? */
#endif

        /* If we have a local particle, we must have found the BH in one
         * of our list of black holes. */
        if (c->nodeID == e->nodeID && !found) {
          error("BH particle %lld could not find BH %lld to be swallowed",
                cell_bp->id, swallow_id);
        }
      } /* Part was flagged for swallowing */
    }   /* Loop over the parts */
  }     /* Cell is not split */
}

/**
 * @brief Processing of bh particles to swallow - self task case.
 *
 * @param r The thread #runner.
 * @param c The #cell.
 * @param timer Are we timing this?
 */
void runner_do_bh_swallow_self(struct runner *r, struct cell *c, int timer) {

#ifdef SWIFT_DEBUG_CHECKS
  if (c->nodeID != r->e->nodeID) error("Running self task on foreign node");
  if (!cell_is_active_black_holes(c, r->e))
    error("Running self task on inactive cell");
#endif

  runner_do_bh_swallow(r, c, timer);
}

/**
 * @brief Processing of bh particles to swallow - pair task case.
 *
 * @param r The thread #runner.
 * @param ci First #cell.
 * @param cj Second #cell.
 * @param timer Are we timing this?
 */
void runner_do_bh_swallow_pair(struct runner *r, struct cell *ci,
                               struct cell *cj, int timer) {

  const struct engine *e = r->e;

#ifdef SWIFT_DEBUG_CHECKS
  if (ci->nodeID != e->nodeID && cj->nodeID != e->nodeID)
    error("Running pair task on foreign node");
#endif

  /* Run the swallowing loop only in the cell that is the neighbour of the
   * active BH */
  if (cell_is_active_black_holes(cj, e)) runner_do_bh_swallow(r, ci, timer);
  if (cell_is_active_black_holes(ci, e)) runner_do_bh_swallow(r, cj, timer);
}

/**
 * @brief Write the required particles through the logger.
 *
 * @param r The runner thread.
 * @param c The cell.
 * @param timer Are we timing this ?
 */
void runner_do_logger(struct runner *r, struct cell *c, int timer) {

#ifdef WITH_LOGGER
  TIMER_TIC;

  const struct engine *e = r->e;
  struct part *restrict parts = c->hydro.parts;
  struct xpart *restrict xparts = c->hydro.xparts;
  const int count = c->hydro.count;

  /* Anything to do here? */
  if (!cell_is_active_hydro(c, e) && !cell_is_active_gravity(c, e)) return;

  /* Recurse? Avoid spending too much time in useless cells. */
  if (c->split) {
    for (int k = 0; k < 8; k++)
      if (c->progeny[k] != NULL) runner_do_logger(r, c->progeny[k], 0);
  } else {

    /* Loop over the parts in this cell. */
    for (int k = 0; k < count; k++) {

      /* Get a handle on the part. */
      struct part *restrict p = &parts[k];
      struct xpart *restrict xp = &xparts[k];

      /* If particle needs to be log */
      /* This is the same function than part_is_active, except for
       * debugging checks */
      if (part_is_active(p, e)) {

        if (logger_should_write(&xp->logger_data, e->logger)) {
          /* Write particle */
          /* Currently writing everything, should adapt it through time */
          logger_log_part(e->logger, p,
                          logger_mask_data[logger_x].mask |
                              logger_mask_data[logger_v].mask |
                              logger_mask_data[logger_a].mask |
                              logger_mask_data[logger_u].mask |
                              logger_mask_data[logger_h].mask |
                              logger_mask_data[logger_rho].mask |
                              logger_mask_data[logger_consts].mask,
                          &xp->logger_data.last_offset);

          /* Set counter back to zero */
          xp->logger_data.steps_since_last_output = 0;
        } else
          /* Update counter */
          xp->logger_data.steps_since_last_output += 1;
      }
    }
  }

  if (c->grav.count > 0) error("gparts not implemented");

  if (c->stars.count > 0) error("sparts not implemented");

  if (timer) TIMER_TOC(timer_logger);

#else
  error("Logger disabled, please enable it during configuration");
#endif
}

/**
 * @brief Recursively search for FOF groups in a single cell.
 *
 * @param r runner task
 * @param c cell
 * @param timer 1 if the time is to be recorded.
 */
void runner_do_fof_self(struct runner *r, struct cell *c, int timer) {

#ifdef WITH_FOF

  TIMER_TIC;

  const struct engine *e = r->e;
  struct space *s = e->s;
  const double dim[3] = {s->dim[0], s->dim[1], s->dim[2]};
  const int periodic = s->periodic;
  const struct gpart *const gparts = s->gparts;
  const double search_r2 = e->fof_properties->l_x2;

  rec_fof_search_self(e->fof_properties, dim, search_r2, periodic, gparts, c);

  if (timer) TIMER_TOC(timer_fof_self);

#else
  error("SWIFT was not compiled with FOF enabled!");
#endif
}

/**
 * @brief Recursively search for FOF groups between a pair of cells.
 *
 * @param r runner task
 * @param ci cell i
 * @param cj cell j
 * @param timer 1 if the time is to be recorded.
 */
void runner_do_fof_pair(struct runner *r, struct cell *ci, struct cell *cj,
                        int timer) {

#ifdef WITH_FOF

  TIMER_TIC;

  const struct engine *e = r->e;
  struct space *s = e->s;
  const double dim[3] = {s->dim[0], s->dim[1], s->dim[2]};
  const int periodic = s->periodic;
  const struct gpart *const gparts = s->gparts;
  const double search_r2 = e->fof_properties->l_x2;

  rec_fof_search_pair(e->fof_properties, dim, search_r2, periodic, gparts, ci,
                      cj);

  if (timer) TIMER_TOC(timer_fof_pair);
#else
  error("SWIFT was not compiled with FOF enabled!");
#endif
}
