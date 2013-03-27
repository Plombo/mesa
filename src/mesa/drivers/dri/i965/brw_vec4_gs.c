/*
 * Copyright © 2013 Intel Corporation
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
 * \file brw_vec4_gs.c
 *
 * State atom for client-programmable geometry shaders, and support code.
 */

#include "brw_vec4_gs.h"
#include "brw_context.h"
#include "brw_vec4_gs_visitor.h"
#include "brw_state.h"


static bool
do_vec4_gs_prog(struct brw_context *brw,
                struct gl_shader_program *prog,
                struct brw_geometry_program *gp,
                struct brw_vec4_gs_prog_key *key)
{
   struct intel_context *intel = &brw->intel;
   struct brw_vec4_gs_compile c;
   memset(&c, 0, sizeof(c));
   c.key = *key;
   c.gp = gp;

   void *mem_ctx = ralloc_context(NULL);

   /* Allocate the references to the uniforms that will end up in the
    * prog_data associated with the compiled program, and which will be freed
    * by the state cache.
    *
    * Note: param_count needs to be num_uniform_components * 4, since we add
    * padding around uniform values below vec4 size, so the worst case is that
    * every uniform is a float which gets padded to the size of a vec4.
    */
   struct gl_shader *gs = prog->_LinkedShaders[MESA_SHADER_GEOMETRY];
   int param_count = gs->num_uniform_components * 4;

   /* We also upload clip plane data as uniforms */
   param_count += MAX_CLIP_PLANES * 4;

   c.prog_data.base.param = rzalloc_array(NULL, const float *, param_count);
   c.prog_data.base.pull_param = rzalloc_array(NULL, const float *, param_count);

   brw_compute_vue_map(brw, &c.prog_data.base.vue_map,
                       gp->program.Base.OutputsWritten,
                       c.key.base.userclip_active);

   /* From the Ivy Bridge PRM, Vol2 Part1 7.2.1.1 STATE_GS - Output Vertex Size (p168):
    *
    *     Programming Restrictions: The vertex size must be programmed as a
    *     multiple of 32B units with the following exception: Rendering is
    *     disabled (as per SOL stage state) and the vertex size output by the
    *     GS thread is 16B.
    *
    *     If rendering is enabled (as per SOL state) the vertex size must be
    *     programmed as a multiple of 32B units. In other words, the only time
    *     software can program a vertex size with an odd number of 16B units
    *     is when rendering is disabled.
    *
    * Note: B=bytes in the above text.
    *
    * It doesn't seem worth the extra trouble to optimize the case where the
    * vertex size is 16B (especially since this would require special-casing
    * the GEN assembly that writes to the URB).  So we just set the vertex
    * size to a multiple of 32B (2 vec4's) in all cases.
    */
   c.prog_data.output_vertex_size_32B =
      (c.prog_data.base.vue_map.num_slots + 1) / 2;

   /* URB entry sizes are computed in multiples of 64 bytes. */
   unsigned output_size_bytes =
      c.prog_data.output_vertex_size_32B * 32 * gp->program.VerticesOut;
   c.prog_data.base.urb_entry_size = ALIGN(output_size_bytes, 64) / 64;

   c.prog_data.output_topology = prim_to_hw_prim[gp->program.OutputType];

   /* GS inputs are read from the VUE 256 bits (2 vec4's) at a time, so we
    * need to program a URB read length of ceiling(num_slots / 2).
    */
   c.prog_data.base.urb_read_length = (c.key.input_vue_map.num_slots + 1) / 2;

   GLuint program_size;
   const GLuint *program =
      brw_vec4_gs_emit(brw, prog, &c, mem_ctx, &program_size);
   if (program == NULL) {
      ralloc_free(mem_ctx);
      return false;
   }

   /* TODO: set c.prog_data.base.num_surfaces properly */

   /* Scratch space is used for register spilling */
   if (c.base.last_scratch) {
      perf_debug("Geometry shader triggered register spilling.  "
                 "Try reducing the number of live vec4 values to "
                 "improve performance.\n");

      c.prog_data.base.total_scratch
         = brw_get_scratch_size(c.base.last_scratch*REG_SIZE);

      brw_get_scratch_bo(intel, &brw->vec4_gs.scratch_bo,
			 c.prog_data.base.total_scratch * brw->max_gs_threads);
   }

   brw_upload_cache(&brw->cache, BRW_VEC4_GS_PROG,
                    &c.key, sizeof(c.key),
                    program, program_size,
                    &c.prog_data, sizeof(c.prog_data),
                    &brw->vec4_gs.prog_offset, &brw->vec4_gs.prog_data);
   ralloc_free(mem_ctx);

   return true;
}


static void
brw_upload_vec4_gs_prog(struct brw_context *brw)
{
   struct intel_context *intel = &brw->intel;
   struct gl_context *ctx = &intel->ctx;
   struct brw_vec4_gs_prog_key key;
   /* BRW_NEW_GEOMETRY_PROGRAM */
   struct brw_geometry_program *gp =
      (struct brw_geometry_program *) brw->geometry_program;

   if (gp == NULL) {
      /* No geometry shader.  Vertex data just passes straight through. */
      if (brw->state.dirty.brw & BRW_NEW_VUE_MAP_VS) {
         brw->vue_map_geom_out = brw->vue_map_vs;
         brw->state.dirty.brw |= BRW_NEW_VUE_MAP_GEOM_OUT;
      }
      return;
   }

   struct gl_program *prog = &gp->program.Base;

   memset(&key, 0, sizeof(key));

   key.base.program_string_id = gp->id;
   key.base.userclip_active = 0; /* TODO */
   key.base.uses_clip_distance = 0; /* TODO */
   key.base.nr_userclip_plane_consts = 0; /* TODO */

   /* _NEW_LIGHT | _NEW_BUFFERS */
   key.base.clamp_vertex_color = ctx->Light._ClampVertexColor;

   /* _NEW_TEXTURE */
   brw_populate_sampler_prog_key_data(ctx, prog, &key.base.tex);

   /* BRW_NEW_VUE_MAP_VS */
   key.input_vue_map = brw->vue_map_vs;

   if (!brw_search_cache(&brw->cache, BRW_VEC4_GS_PROG,
                         &key, sizeof(key),
                         &brw->vec4_gs.prog_offset, &brw->vec4_gs.prog_data)) {
      bool success = do_vec4_gs_prog(brw, ctx->Shader.CurrentGeometryProgram,
                                     gp, &key);
      assert(success);
   }
   if (memcmp(&brw->vs.prog_data->base.vue_map, &brw->vue_map_geom_out,
              sizeof(brw->vue_map_geom_out)) != 0) {
      brw->vue_map_geom_out = brw->vec4_gs.prog_data->base.vue_map;
      brw->state.dirty.brw |= BRW_NEW_VUE_MAP_GEOM_OUT;
   }
}


const struct brw_tracked_state brw_vec4_gs_prog = {
   .dirty = {
      .mesa  = (_NEW_LIGHT | _NEW_BUFFERS | _NEW_TEXTURE),
      .brw   = BRW_NEW_GEOMETRY_PROGRAM | BRW_NEW_VUE_MAP_VS,
   },
   .emit = brw_upload_vec4_gs_prog
};


bool
brw_vec4_gs_prog_data_compare(const void *in_a, const void *in_b,
                              int aux_size, const void *in_key)
{
   const struct brw_vec4_gs_prog_data *a = in_a;
   const struct brw_vec4_gs_prog_data *b = in_b;

   /* Compare the base vec4 structure. */
   if (!brw_vec4_prog_data_compare(&a->base, &b->base))
      return false;

   /* Compare the rest of the struct. */
   const unsigned offset = sizeof(struct brw_vec4_prog_data);
   if (memcmp(((char *) &a) + offset, ((char *) &b) + offset,
              sizeof(struct brw_vec4_gs_prog_data) - offset)) {
      return false;
   }

   return true;
}


void
brw_vec4_gs_prog_data_free(const void *in_prog_data)
{
   const struct brw_vec4_gs_prog_data *prog_data = in_prog_data;

   brw_vec4_prog_data_free(&prog_data->base);
}
