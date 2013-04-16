/*
 * Copyright © 2010 Intel Corporation
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
 * \file ir_set_program_inouts.cpp
 *
 * Sets the InputsRead and OutputsWritten of Mesa programs.
 *
 * Additionally, for fragment shaders, sets the InterpQualifier array, the
 * IsCentroid bitfield, and the UsesDFdy flag.
 *
 * Mesa programs (gl_program, not gl_shader_program) have a set of
 * flags indicating which varyings are read and written.  Computing
 * which are actually read from some sort of backend code can be
 * tricky when variable array indexing involved.  So this pass
 * provides support for setting InputsRead and OutputsWritten right
 * from the GLSL IR.
 */

#include "main/core.h" /* for struct gl_program */
#include "ir.h"
#include "ir_visitor.h"
#include "glsl_types.h"

class ir_set_program_inouts_visitor : public ir_hierarchical_visitor {
public:
   ir_set_program_inouts_visitor(struct gl_program *prog,
                                 bool is_fragment_shader,
                                 bool is_geometry_shader)
   {
      this->prog = prog;
      this->is_fragment_shader = is_fragment_shader;
      this->is_geometry_shader = is_geometry_shader;
   }
   ~ir_set_program_inouts_visitor()
   {
   }

   virtual ir_visitor_status visit_enter(ir_dereference_array *);
   virtual ir_visitor_status visit_enter(ir_function_signature *);
   virtual ir_visitor_status visit_enter(ir_expression *);
   virtual ir_visitor_status visit_enter(ir_discard *);
   virtual ir_visitor_status visit(ir_dereference_variable *);

   struct gl_program *prog;
   bool is_fragment_shader;
   bool is_geometry_shader;
};

static inline bool
is_shader_inout(ir_variable *var)
{
   return var->mode == ir_var_shader_in ||
          var->mode == ir_var_shader_out ||
          var->mode == ir_var_system_value;
}

static void
mark(struct gl_program *prog, ir_variable *var, int offset, int len,
     bool is_fragment_shader)
{
   /* As of GLSL 1.20, varyings can only be floats, floating-point
    * vectors or matrices, or arrays of them.  For Mesa programs using
    * InputsRead/OutputsWritten, everything but matrices uses one
    * slot, while matrices use a slot per column.  Presumably
    * something doing a more clever packing would use something other
    * than InputsRead/OutputsWritten.
    */

   for (int i = 0; i < len; i++) {
      GLbitfield64 bitfield = BITFIELD64_BIT(var->location + var->index + offset + i);
      if (var->mode == ir_var_shader_in) {
	 prog->InputsRead |= bitfield;
         if (is_fragment_shader) {
            gl_fragment_program *fprog = (gl_fragment_program *) prog;
            fprog->InterpQualifier[var->location + var->index + offset + i] =
               (glsl_interp_qualifier) var->interpolation;
            if (var->centroid)
               fprog->IsCentroid |= bitfield;
         }
      } else if (var->mode == ir_var_system_value) {
         prog->SystemValuesRead |= bitfield;
      } else {
         assert(var->mode == ir_var_shader_out);
	 prog->OutputsWritten |= bitfield;
      }
   }
}

/* Default handler: Mark all the locations in the variable as used. */
ir_visitor_status
ir_set_program_inouts_visitor::visit(ir_dereference_variable *ir)
{
   if (!is_shader_inout(ir->var))
      return visit_continue;

   if (ir->type->is_array()) {
      int matrix_columns = ir->type->fields.array->matrix_columns;
      int length = ir->type->length;
      if (this->is_geometry_shader && ir->var->mode == ir_var_shader_in)
      {
         if (ir->type->element_type()->is_array()) {
            const glsl_type *inner_array_type = ir->type->fields.array;
            matrix_columns = inner_array_type->fields.array->matrix_columns;
            length = inner_array_type->length;
         }
         else
            length = 1;
      }
      mark(this->prog, ir->var, 0, length * matrix_columns,
           this->is_fragment_shader);
   } else {
      mark(this->prog, ir->var, 0, ir->type->matrix_columns,
           this->is_fragment_shader);
   }
   return visit_continue;
}

ir_visitor_status
ir_set_program_inouts_visitor::visit_enter(ir_dereference_array *ir)
{
   ir_dereference_variable *deref_var;
   ir_constant *index = ir->array_index->as_constant();
   deref_var = ir->array->as_dereference_variable();
   ir_variable *var;
   bool is_vert_array = false, is_2D_array = false;

   /* Check whether this dereference is of a GS input array.  These are special
    * because the array index refers to the index of an input vertex instead of
    * the attribute index.  The exceptions to this exception are 2D arrays
    * such as gl_TexCoordIn.  For these, there is a nested dereference_array,
    * where the inner index specifies the vertex and the outer index specifies
    * the attribute.  To complicate things further, matrix columns are also
    * accessed with dereference_array.  So we have to correctly handle 1D
    * arrays of non-matrices, 1D arrays of matrices, 2D arrays of non-matrices,
    * and 2D arrays of matrices.
    */
   if (this->is_geometry_shader) {
      if (!deref_var) {
         /* Either an outer (attribute) dereference of a 2D array or a column
          * dereference of an array of matrices. */
         ir_dereference_array *inner_deref = ir->array->as_dereference_array();
         assert(inner_deref);
         deref_var = inner_deref->array->as_dereference_variable();
         is_2D_array = true;
      }

      if (deref_var && deref_var->var->mode == ir_var_shader_in) {
         if (ir->type->is_array())
            /* Inner (vertex) dereference of a 2D array */
            return visit_continue;
         else
            /* Dereference of a 1D (vertex) array */
            is_vert_array = true;
      }
   }

   var = deref_var ? deref_var->var : NULL;

   /* Check that we're dereferencing a shader in or out */
   if (!var || !is_shader_inout(var))
      return visit_continue;

   if (index) {
      int width = 1;
      const glsl_type *type = is_vert_array ?
                              deref_var->type->fields.array : deref_var->type;
      int offset = is_vert_array && !is_2D_array ? 0 : index->value.i[0];

      if (type->is_array() &&
	  type->fields.array->is_matrix()) {
	 width = type->fields.array->matrix_columns;
      }

      mark(this->prog, var, offset, width, this->is_fragment_shader);
      return visit_continue_with_parent;
   }

   return visit_continue;
}

ir_visitor_status
ir_set_program_inouts_visitor::visit_enter(ir_function_signature *ir)
{
   /* We don't want to descend into the function parameters and
    * consider them as shader inputs or outputs.
    */
   visit_list_elements(this, &ir->body);
   return visit_continue_with_parent;
}

ir_visitor_status
ir_set_program_inouts_visitor::visit_enter(ir_expression *ir)
{
   if (is_fragment_shader && ir->operation == ir_unop_dFdy) {
      gl_fragment_program *fprog = (gl_fragment_program *) prog;
      fprog->UsesDFdy = true;
   }
   return visit_continue;
}

ir_visitor_status
ir_set_program_inouts_visitor::visit_enter(ir_discard *)
{
   /* discards are only allowed in fragment shaders. */
   assert(is_fragment_shader);

   gl_fragment_program *fprog = (gl_fragment_program *) prog;
   fprog->UsesKill = true;

   return visit_continue;
}

void
do_set_program_inouts(exec_list *instructions, struct gl_program *prog,
                      bool is_fragment_shader, bool is_geometry_shader)
{
   ir_set_program_inouts_visitor v(prog, is_fragment_shader, is_geometry_shader);

   prog->InputsRead = 0;
   prog->OutputsWritten = 0;
   prog->SystemValuesRead = 0;
   if (is_fragment_shader) {
      gl_fragment_program *fprog = (gl_fragment_program *) prog;
      memset(fprog->InterpQualifier, 0, sizeof(fprog->InterpQualifier));
      fprog->IsCentroid = 0;
      fprog->UsesDFdy = false;
      fprog->UsesKill = false;
   }
   visit_list_elements(&v, instructions);
}
