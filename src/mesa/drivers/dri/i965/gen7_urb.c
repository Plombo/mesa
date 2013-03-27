/*
 * Copyright © 2011 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "main/macros.h"
#include "intel_batchbuffer.h"
#include "brw_context.h"
#include "brw_state.h"
#include "brw_defines.h"

/**
 * The following diagram shows how we partition the URB:
 *
 *      8kB         8kB              Rest of the URB space
 *   ____-____   ____-____   _________________-_________________
 *  /         \ /         \ /                                   \
 * +-------------------------------------------------------------+
 * | VS Push   | FS Push   | VS                                  |
 * | Constants | Constants | Handles                             |
 * +-------------------------------------------------------------+
 *
 * Notably, push constants must be stored at the beginning of the URB
 * space, while entries can be stored anywhere.  Ivybridge has a maximum
 * constant buffer size of 16kB.
 *
 * Currently we split the constant buffer space evenly between VS and FS.
 * This is probably not ideal, but simple.
 *
 * Ivybridge GT1 has 128kB of URB space.
 * Ivybridge GT2 has 256kB of URB space.
 *
 * See "Volume 2a: 3D Pipeline," section 1.8.
 */
void
gen7_allocate_push_constants(struct brw_context *brw)
{
   struct intel_context *intel = &brw->intel;

   /* HACK for GS: split the push constant space evenly among GS, VS,
    * and FS.
    *
    * TODO: Really what we should do is adjust the allocation based on whether
    * GS is active.
    */

   BEGIN_BATCH(2);
   OUT_BATCH(_3DSTATE_PUSH_CONSTANT_ALLOC_VS << 16 | (2 - 2));
   OUT_BATCH(5);
   ADVANCE_BATCH();

   BEGIN_BATCH(2);
   OUT_BATCH(_3DSTATE_PUSH_CONSTANT_ALLOC_GS << 16 | (2 - 2));
   OUT_BATCH(5 | 5 << GEN7_PUSH_CONSTANT_BUFFER_OFFSET_SHIFT);
   ADVANCE_BATCH();

   BEGIN_BATCH(2);
   OUT_BATCH(_3DSTATE_PUSH_CONSTANT_ALLOC_PS << 16 | (2 - 2));
   OUT_BATCH(6 | 10 << GEN7_PUSH_CONSTANT_BUFFER_OFFSET_SHIFT);
   ADVANCE_BATCH();
}

const struct brw_tracked_state gen7_push_constant_alloc = {
   .dirty = {
      .mesa = 0,
      .brw = BRW_NEW_CONTEXT,
      .cache = 0,
   },
   .emit = gen7_allocate_push_constants,
};

static void
gen7_upload_urb(struct brw_context *brw)
{
   struct intel_context *intel = &brw->intel;
   const int push_size_kB = 16;

   /* Figure out the size of URB entries for the VS and GS.  These are in
    * multiples of 64 bytes.
    */

   /* CACHE_NEW_VS_PROG */
   unsigned vs_size = MAX2(brw->vs.prog_data->base.urb_entry_size, 1);
   unsigned vs_entry_size_bytes = vs_size * 64;
   /* BRW_NEW_GEOMETRY_PROGRAM, CACHE_NEW_VEC4_GS_PROG */
   unsigned gs_size = brw->geometry_program ?
      MAX2(brw->vec4_gs.prog_data->base.urb_entry_size, 1) : 1;
   unsigned gs_entry_size_bytes = gs_size * 64;

   /* Figure out the total size of the URB, in multiples of 8192, the minimum
    * size chunk we can allocate.
    */
   const unsigned chunk_size_bytes = 8192;
   unsigned space_available_chunks =
      brw->urb.size * 1024 / chunk_size_bytes;

   /* Reserve space for push constants */
   unsigned push_constant_size_chunks = 1024 * push_size_kB / chunk_size_bytes;
   space_available_chunks -= push_constant_size_chunks;

   /* If the GS is in use, assign half the remaining URB space to it. */
   unsigned gs_size_chunks =
      brw->geometry_program ? space_available_chunks / 2 : 0;
   space_available_chunks -= gs_size_chunks;
   unsigned nr_gs_entries =
      gs_size_chunks * chunk_size_bytes / gs_entry_size_bytes;

   /* Assign the remaining URB space to the VS. */
   unsigned vs_size_chunks = space_available_chunks;
   unsigned nr_vs_entries =
      vs_size_chunks * chunk_size_bytes / vs_entry_size_bytes;

   /* Then clamp to the maximum allowed by the hardware */
   if (nr_vs_entries > brw->urb.max_vs_entries)
      nr_vs_entries = brw->urb.max_vs_entries;
   if (nr_gs_entries > brw->urb.max_gs_entries)
      nr_gs_entries = brw->urb.max_gs_entries;

   /* According to volume 2a, nr_vs_entries and nr_gs_entries must be a
    * multiple of 8.
    */
   brw->urb.nr_vs_entries = ROUND_DOWN_TO(nr_vs_entries, 8);
   brw->urb.nr_gs_entries = ROUND_DOWN_TO(nr_gs_entries, 8);

   /* Lay out the URB in the following order:
    * - push constants
    * - VS
    * - GS
    */
   brw->urb.vs_start = push_constant_size_chunks;
   brw->urb.gs_start = push_constant_size_chunks + vs_size_chunks;

   gen7_emit_vs_workaround_flush(intel);
   gen7_emit_urb_state(brw,
                       brw->urb.nr_vs_entries, vs_size, brw->urb.vs_start,
                       brw->urb.nr_gs_entries, gs_size, brw->urb.gs_start);
}

void
gen7_emit_urb_state(struct brw_context *brw,
                    GLuint nr_vs_entries, GLuint vs_size, GLuint vs_start,
                    GLuint nr_gs_entries, GLuint gs_size, GLuint gs_start)
{
   struct intel_context *intel = &brw->intel;

   BEGIN_BATCH(2);
   OUT_BATCH(_3DSTATE_URB_VS << 16 | (2 - 2));
   OUT_BATCH(nr_vs_entries |
             ((vs_size - 1) << GEN7_URB_ENTRY_SIZE_SHIFT) |
             (vs_start << GEN7_URB_STARTING_ADDRESS_SHIFT));
   ADVANCE_BATCH();

   BEGIN_BATCH(2);
   OUT_BATCH(_3DSTATE_URB_GS << 16 | (2 - 2));
   OUT_BATCH(nr_gs_entries |
             ((gs_size - 1) << GEN7_URB_ENTRY_SIZE_SHIFT) |
             (gs_start << GEN7_URB_STARTING_ADDRESS_SHIFT));
   ADVANCE_BATCH();

   /* Allocate the HS and DS zero space - we don't use them. */
   BEGIN_BATCH(2);
   OUT_BATCH(_3DSTATE_URB_HS << 16 | (2 - 2));
   OUT_BATCH((0 << GEN7_URB_ENTRY_SIZE_SHIFT) |
             (vs_start << GEN7_URB_STARTING_ADDRESS_SHIFT));
   ADVANCE_BATCH();

   BEGIN_BATCH(2);
   OUT_BATCH(_3DSTATE_URB_DS << 16 | (2 - 2));
   OUT_BATCH((0 << GEN7_URB_ENTRY_SIZE_SHIFT) |
             (vs_start << GEN7_URB_STARTING_ADDRESS_SHIFT));
   ADVANCE_BATCH();
}

const struct brw_tracked_state gen7_urb = {
   .dirty = {
      .mesa = 0,
      .brw = BRW_NEW_CONTEXT | BRW_NEW_GEOMETRY_PROGRAM,
      .cache = (CACHE_NEW_VS_PROG | CACHE_NEW_VEC4_GS_PROG),
   },
   .emit = gen7_upload_urb,
};
