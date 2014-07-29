#include <stdio.h>
#include <assert.h>
#include <llvm-c/Core.h>
#include "gallivm/lp_bld_const.h"
#include "gallivm/lp_bld_gather.h"
#include "gallivm/lp_bld_intr.h"
#include "gallivm/lp_bld_logic.h"
#include "gallivm/lp_bld_arit.h"
#include "gallivm/lp_bld_flow.h"
#include "tgsi/tgsi_parse.h"
#include "tgsi/tgsi_dump.h"
#include "nouveau_llvm.h"

#include "nv50/nv50_screen.h"
#define NUM_CONST_BUFFERS NV50_MAX_PIPE_CONSTBUFS

/* LLVM function parameter indices */
#define SI_PARAM_CONST		0
#define SI_PARAM_SAMPLER	1
#define SI_PARAM_RESOURCE	2
#define SI_PARAM_RW_BUFFERS	3

/* VS only parameters */
#define SI_PARAM_VERTEX_BUFFER	4
#define SI_PARAM_START_INSTANCE	5
/* the other VS parameters are assigned dynamically */

#define SI_NUM_PARAMS 22

#define CONST_ADDR_SPACE 2
#define LOCAL_ADDR_SPACE 3
#define USER_SGPR_ADDR_SPACE 8

void nouveau_llvm_shader_type(LLVMValueRef F, unsigned type);

struct nouveau_shader_output_values
{
	LLVMValueRef values[4];
	unsigned name;
	unsigned index;
	unsigned sid;
	unsigned usage;
};

struct nouveau_shader_context
{
	struct nouveau_llvm_context nouveau_bld;
	struct tgsi_parse_context parse;
	struct tgsi_token * tokens;
	//struct nv50_program *shader;
	struct nv50_program *gs_for_vs;
	unsigned type; /* TGSI_PROCESSOR_* specifies the type of shader. */
	int param_streamout_config;
	int param_streamout_write_index;
	int param_streamout_offset[4];
	int param_vertex_id;
	int param_instance_id;
	LLVMValueRef const_md;
	LLVMValueRef const_resource[NUM_CONST_BUFFERS];
#if HAVE_LLVM >= 0x0304
	LLVMValueRef ddxy_lds;
#endif
	LLVMValueRef *constants[NUM_CONST_BUFFERS];
	LLVMValueRef *resources;
	LLVMValueRef *samplers;
	LLVMValueRef so_buffers[4];
	LLVMValueRef gs_next_vertex;
};

static struct nouveau_shader_context * nouveau_shader_context(
	struct lp_build_tgsi_context * bld_base)
{
	return (struct nouveau_shader_context *)bld_base;
}

/**
 * Set the shader type we want to compile
 *
 * @param type shader type to set
 */
void nouveau_llvm_shader_type(LLVMValueRef F, unsigned type)
{
  char Str[2];
  sprintf(Str, "%1d", type);

  LLVMAddTargetDependentFunctionAttr(F, "ShaderType", Str);
}

/**
 * Build an LLVM bytecode indexed load using LLVMBuildGEP + LLVMBuildLoad
 *
 * @param offset The offset parameter specifies the number of
 * elements to offset, not the number of bytes or dwords.  An element is the
 * the type pointed to by the base_ptr parameter (e.g. int is the element of
 * an int* pointer)
 *
 * When LLVM lowers the load instruction, it will convert the element offset
 * into a dword offset automatically.
 *
 */
static LLVMValueRef build_indexed_load(
	struct nouveau_shader_context * shader_ctx,
	LLVMValueRef base_ptr,
	LLVMValueRef offset)
{
	struct lp_build_context * base = &shader_ctx->nouveau_bld.soa.bld_base.base;

	LLVMValueRef indices[2] = {
		LLVMConstInt(LLVMInt64TypeInContext(base->gallivm->context), 0, false),
		offset
	};
	LLVMValueRef computed_ptr = LLVMBuildGEP(
		base->gallivm->builder, base_ptr, indices, 2, "");

	LLVMValueRef result = LLVMBuildLoad(base->gallivm->builder, computed_ptr, "");
	LLVMSetMetadata(result, 1, shader_ctx->const_md);
	return result;
}

static LLVMValueRef load_const(LLVMBuilderRef builder, LLVMValueRef resource,
			       LLVMValueRef offset, LLVMTypeRef return_type)
{
	LLVMValueRef args[2] = {resource, offset};

	return build_intrinsic(builder, "llvm.SI.load.const", return_type, args, 2,
			       LLVMReadNoneAttribute | LLVMNoUnwindAttribute);
}

static LLVMValueRef fetch_constant(
	struct lp_build_tgsi_context * bld_base,
	const struct tgsi_full_src_register *reg,
	enum tgsi_opcode_type type,
	unsigned swizzle)
{
	struct nouveau_shader_context *shader_ctx = nouveau_shader_context(bld_base);
	struct lp_build_context * base = &bld_base->base;
	const struct tgsi_ind_register *ireg = &reg->Indirect;
	unsigned buf, idx;

	LLVMValueRef addr;
	LLVMValueRef result;

	if (swizzle == LP_CHAN_ALL) {
		unsigned chan;
		LLVMValueRef values[4];
		for (chan = 0; chan < TGSI_NUM_CHANNELS; ++chan)
			values[chan] = fetch_constant(bld_base, reg, type, chan);

		return lp_build_gather_values(bld_base->base.gallivm, values, 4);
	}

	buf = reg->Register.Dimension ? reg->Dimension.Index : 0;
	idx = reg->Register.Index * 4 + swizzle;

	if (!reg->Register.Indirect)
		return bitcast(bld_base, type, shader_ctx->constants[buf][idx]);

	addr = shader_ctx->nouveau_bld.soa.addr[ireg->Index][ireg->Swizzle];
	addr = LLVMBuildLoad(base->gallivm->builder, addr, "load addr reg");
	addr = lp_build_mul_imm(&bld_base->uint_bld, addr, 16);
	addr = lp_build_add(&bld_base->uint_bld, addr,
			    lp_build_const_int32(base->gallivm, idx * 4));

	result = load_const(base->gallivm->builder, shader_ctx->const_resource[buf],
			    addr, base->elem_type);

	return bitcast(bld_base, type, result);
}

static void llvm_load_input(
	struct nouveau_llvm_context * ctx,
	unsigned input_index,
	const struct tgsi_full_declaration *decl)
{
	//const struct r600_shader_io * input = &ctx->r600_inputs[input_index];
	unsigned chan;
#if HAVE_LLVM < 0x0304
	unsigned interp = 0;
	int ij_index;
#endif
	//int two_side = (ctx->two_side && input->name == TGSI_SEMANTIC_COLOR);
	LLVMValueRef v;

#if HAVE_LLVM >= 0x0304
	v = LLVMGetParam(ctx->main_fn, input_index);

#if 0
	if (two_side) {
		struct r600_shader_io * back_input =
			&ctx->r600_inputs[input->back_color_input];
		LLVMValueRef v2;
		LLVMValueRef face = LLVMGetParam(ctx->main_fn, ctx->face_gpr);
		face = LLVMBuildExtractElement(ctx->gallivm.builder, face,
			lp_build_const_int32(&(ctx->gallivm), 0), "");

		if (require_interp_intrinsic && back_input->spi_sid)
			v2 = llvm_load_input_vector(ctx, back_input->lds_pos,
				back_input->ij_index, (back_input->interpolate > 0));
		else
			v2 = LLVMGetParam(ctx->main_fn, back_input->gpr);
		v = llvm_face_select_helper(ctx, face, v, v2);
	}
#endif

	for (chan = 0; chan < 4; chan++) {
		unsigned soa_index = nouveau_llvm_reg_index_soa(input_index, chan);

		ctx->inputs[soa_index] = LLVMBuildExtractElement(ctx->gallivm.builder, v,
			lp_build_const_int32(&(ctx->gallivm), chan), "");
#if 0
		if (input->name == TGSI_SEMANTIC_POSITION &&
				ctx->type == TGSI_PROCESSOR_FRAGMENT && chan == 3) {
			/* RCP for fragcoord.w */
			ctx->inputs[soa_index] = LLVMBuildFDiv(ctx->gallivm.builder,
					lp_build_const_float(&(ctx->gallivm), 1.0f),
					ctx->inputs[soa_index], "");
		}
#endif
	}
#else
#error LLVM >= 3.4 required
	for (chan = 0; chan < 4; chan++) {
		unsigned soa_index = nouveau_llvm_reg_index_soa(input_index, chan);
		int loc;

		if (interp) {
			loc = 4 * input->lds_pos + chan;
		} else {
			if (input->name == TGSI_SEMANTIC_FACE)
				loc = 4 * ctx->face_gpr;
			else
				loc = 4 * input->gpr + chan;
		}

		v = llvm_load_input_helper(ctx, loc, interp, ij_index);

#if 0
		if (two_side) {
			struct r600_shader_io * back_input =
					&ctx->r600_inputs[input->back_color_input];
			int back_loc = interp ? back_input->lds_pos : back_input->gpr;
			LLVMValueRef v2;

			back_loc = 4 * back_loc + chan;
			v2 = llvm_load_input_helper(ctx, back_loc, interp, ij_index);
			v = llvm_face_select_helper(ctx, 4 * ctx->face_gpr, v, v2);
		} else if (input->name == TGSI_SEMANTIC_POSITION &&
				ctx->type == TGSI_PROCESSOR_FRAGMENT && chan == 3) {
			/* RCP for fragcoord.w */
			v = LLVMBuildFDiv(ctx->gallivm.builder,
					lp_build_const_float(&(ctx->gallivm), 1.0f),
					v, "");
		}
#endif

		ctx->inputs[soa_index] = v;
	}
#endif
}

static void declare_system_value(
	struct nouveau_llvm_context * nouveau_bld,
	unsigned index,
	const struct tgsi_full_declaration *decl)
{
	assert(!"System value loading not supported");
}

// TODO figure out what this even is
static void create_meta_data(struct nouveau_shader_context *shader_ctx)
{
	struct gallivm_state *gallivm = shader_ctx->nouveau_bld.soa.bld_base.base.gallivm;
	LLVMValueRef args[3];

	args[0] = LLVMMDStringInContext(gallivm->context, "const", 5);
	args[1] = 0;
	args[2] = lp_build_const_int32(gallivm, 1);

	shader_ctx->const_md = LLVMMDNodeInContext(gallivm->context, args, 3);
}

static void create_function(struct nouveau_shader_context *shader_ctx)
{
	struct lp_build_tgsi_context *bld_base = &shader_ctx->nouveau_bld.soa.bld_base;
	struct gallivm_state *gallivm = bld_base->base.gallivm;
	//struct si_pipe_shader *shader = shader_ctx->shader;
	LLVMTypeRef params[SI_NUM_PARAMS], f32, i8, i32, v2i32, v3i32;
	unsigned i, last_sgpr, num_params;

	i8 = LLVMInt8TypeInContext(gallivm->context);
	i32 = LLVMInt32TypeInContext(gallivm->context);
	f32 = LLVMFloatTypeInContext(gallivm->context);
	v2i32 = LLVMVectorType(i32, 2);
	v3i32 = LLVMVectorType(i32, 3);

	params[SI_PARAM_CONST] = LLVMPointerType(
		LLVMArrayType(LLVMVectorType(i8, 16), NUM_CONST_BUFFERS), CONST_ADDR_SPACE);
	params[SI_PARAM_RW_BUFFERS] = params[SI_PARAM_CONST];

#if 0
	/* We assume at most 16 textures per program at the moment.
	 * This need probably need to be changed to support bindless textures */
	params[SI_PARAM_SAMPLER] = LLVMPointerType(
		LLVMArrayType(LLVMVectorType(i8, 16), NUM_SAMPLER_STATES), CONST_ADDR_SPACE);
	params[SI_PARAM_RESOURCE] = LLVMPointerType(
		LLVMArrayType(LLVMVectorType(i8, 32), NUM_SAMPLER_VIEWS), CONST_ADDR_SPACE);
#endif

	switch (shader_ctx->type) {
	case TGSI_PROCESSOR_VERTEX:
		params[SI_PARAM_VERTEX_BUFFER] = params[SI_PARAM_CONST];
		params[SI_PARAM_START_INSTANCE] = i32;
		num_params = SI_PARAM_START_INSTANCE+1;
#if 0
		if (shader->key.vs.as_es) {
			params[SI_PARAM_ES2GS_OFFSET] = i32;
			num_params++;
		} else {
			/* The locations of the other parameters are assigned dynamically. */

			/* Streamout SGPRs. */
			if (shader->selector->so.num_outputs) {
				params[shader_ctx->param_streamout_config = num_params++] = i32;
				params[shader_ctx->param_streamout_write_index = num_params++] = i32;
			}
			/* A streamout buffer offset is loaded if the stride is non-zero. */
			for (i = 0; i < 4; i++) {
				if (!shader->selector->so.stride[i])
					continue;

				params[shader_ctx->param_streamout_offset[i] = num_params++] = i32;
			}
		}
#endif

		last_sgpr = num_params-1;

		/* VGPRs */
		params[shader_ctx->param_vertex_id = num_params++] = i32;
		params[num_params++] = i32; /* unused*/
		params[num_params++] = i32; /* unused */
		params[shader_ctx->param_instance_id = num_params++] = i32;
		break;
#if 0
	case TGSI_PROCESSOR_GEOMETRY:
		params[SI_PARAM_GS2VS_OFFSET] = i32;
		params[SI_PARAM_GS_WAVE_ID] = i32;
		last_sgpr = SI_PARAM_GS_WAVE_ID;

		/* VGPRs */
		params[SI_PARAM_VTX0_OFFSET] = i32;
		params[SI_PARAM_VTX1_OFFSET] = i32;
		params[SI_PARAM_PRIMITIVE_ID] = i32;
		params[SI_PARAM_VTX2_OFFSET] = i32;
		params[SI_PARAM_VTX3_OFFSET] = i32;
		params[SI_PARAM_VTX4_OFFSET] = i32;
		params[SI_PARAM_VTX5_OFFSET] = i32;
		params[SI_PARAM_GS_INSTANCE_ID] = i32;
		num_params = SI_PARAM_GS_INSTANCE_ID+1;
		break;

	case TGSI_PROCESSOR_FRAGMENT:
		params[SI_PARAM_ALPHA_REF] = f32;
		params[SI_PARAM_PRIM_MASK] = i32;
		last_sgpr = SI_PARAM_PRIM_MASK;
		params[SI_PARAM_PERSP_SAMPLE] = v2i32;
		params[SI_PARAM_PERSP_CENTER] = v2i32;
		params[SI_PARAM_PERSP_CENTROID] = v2i32;
		params[SI_PARAM_PERSP_PULL_MODEL] = v3i32;
		params[SI_PARAM_LINEAR_SAMPLE] = v2i32;
		params[SI_PARAM_LINEAR_CENTER] = v2i32;
		params[SI_PARAM_LINEAR_CENTROID] = v2i32;
		params[SI_PARAM_LINE_STIPPLE_TEX] = f32;
		params[SI_PARAM_POS_X_FLOAT] = f32;
		params[SI_PARAM_POS_Y_FLOAT] = f32;
		params[SI_PARAM_POS_Z_FLOAT] = f32;
		params[SI_PARAM_POS_W_FLOAT] = f32;
		params[SI_PARAM_FRONT_FACE] = f32;
		params[SI_PARAM_ANCILLARY] = i32;
		params[SI_PARAM_SAMPLE_COVERAGE] = f32;
		params[SI_PARAM_POS_FIXED_PT] = f32;
		num_params = SI_PARAM_POS_FIXED_PT+1;
		break;
#endif
	default:
		assert(0 && "unimplemented shader");
		return;
	}

	assert(num_params <= Elements(params));
	nouveau_llvm_create_func(&shader_ctx->nouveau_bld, params, num_params);
	nouveau_llvm_shader_type(shader_ctx->nouveau_bld.main_fn, shader_ctx->type);

	for (i = 0; i <= last_sgpr; ++i) {
		LLVMValueRef P = LLVMGetParam(shader_ctx->nouveau_bld.main_fn, i);
		switch (i) {
		default:
			LLVMAddAttribute(P, LLVMInRegAttribute);
			break;
#if HAVE_LLVM >= 0x0304
		/* We tell llvm that array inputs are passed by value to allow Sinking pass
		 * to move load. Inputs are constant so this is fine. */
		case SI_PARAM_CONST:
		case SI_PARAM_SAMPLER:
		case SI_PARAM_RESOURCE:
			LLVMAddAttribute(P, LLVMByValAttribute);
			break;
#endif
		}
	}

#if HAVE_LLVM >= 0x0304
	if (bld_base->info &&
	    (bld_base->info->opcode_count[TGSI_OPCODE_DDX] > 0 ||
	     bld_base->info->opcode_count[TGSI_OPCODE_DDY] > 0))
		shader_ctx->ddxy_lds =
			LLVMAddGlobalInAddressSpace(gallivm->module,
						    LLVMArrayType(i32, 64),
						    "ddxy_lds",
						    LOCAL_ADDR_SPACE);
#endif
}

static void preload_constants(struct nouveau_shader_context *shader_ctx)
{
	struct lp_build_tgsi_context * bld_base = &shader_ctx->nouveau_bld.soa.bld_base;
	struct gallivm_state * gallivm = bld_base->base.gallivm;
	const struct tgsi_shader_info * info = bld_base->info;
	unsigned buf;
	LLVMValueRef ptr = LLVMGetParam(shader_ctx->nouveau_bld.main_fn, SI_PARAM_CONST);

	for (buf = 0; buf < NUM_CONST_BUFFERS; buf++) {
		unsigned i, num_const = info->const_file_max[buf] + 1;

		if (num_const == 0)
			continue;

		/* Allocate space for the constant values */
		shader_ctx->constants[buf] = CALLOC(num_const * 4, sizeof(LLVMValueRef));

		/* Load the resource descriptor */
		shader_ctx->const_resource[buf] =
			build_indexed_load(shader_ctx, ptr, lp_build_const_int32(gallivm, buf));

		/* Load the constants, we rely on the code sinking to do the rest */
		for (i = 0; i < num_const * 4; ++i) {
			shader_ctx->constants[buf][i] =
				load_const(gallivm->builder,
					shader_ctx->const_resource[buf],
					lp_build_const_int32(gallivm, i * 4),
					bld_base->base.elem_type);
		}
	}
}

#if 0
static void nouveau_llvm_optimize(LLVMModuleRef mod)
{
	const char *data_layout = LLVMGetDataLayout(mod);
	LLVMTargetDataRef TD = LLVMCreateTargetData(data_layout);
	LLVMPassManagerBuilderRef builder = LLVMPassManagerBuilderCreate();
	LLVMPassManagerRef pass_manager = LLVMCreatePassManager();

	/* Functions calls are not supported yet, so we need to inline
	 * everything.  The most efficient way to do this is to add
	 * the always_inline attribute to all non-kernel functions
	 * and then run the Always Inline pass.  The Always Inline
	 * pass will automaically inline functions with this attribute
	 * and does not perform the expensive cost analysis that the normal
	 * inliner does.
	 */

	LLVMValueRef fn;
	for (fn = LLVMGetFirstFunction(mod); fn; fn = LLVMGetNextFunction(fn)) {
		/* All the non-kernel functions have internal linkage */
		if (LLVMGetLinkage(fn) == LLVMInternalLinkage) {
			LLVMAddFunctionAttr(fn, LLVMAlwaysInlineAttribute);
		}
	}

	LLVMAddTargetData(TD, pass_manager);
	LLVMAddAlwaysInlinerPass(pass_manager);
	LLVMPassManagerBuilderPopulateModulePassManager(builder, pass_manager);

	LLVMRunPassManager(pass_manager, mod);
	LLVMPassManagerBuilderDispose(builder);
	LLVMDisposePassManager(pass_manager);
	LLVMDisposeTargetData(TD);
}
#endif

/*int nouveau_pipe_shader_create(
	struct pipe_context *ctx,
	struct si_pipe_shader *shader)*/
LLVMModuleRef nouveau_tgsi_llvm(const struct tgsi_token * tokens)
{
	//struct si_context *sctx = (struct si_context*)ctx;
	struct nouveau_shader_context shader_ctx;
	struct tgsi_shader_info shader_info;
	struct lp_build_tgsi_context * bld_base;
	LLVMModuleRef mod;
	int r = 0;
	bool dump = TRUE;

	/* Dump TGSI code before doing TGSI->LLVM conversion in case the
	 * conversion fails. */
	if (dump) {
		tgsi_dump(tokens, 0);
		//si_dump_streamout(&sel->so);
	}

	//assert(shader->shader.noutput == 0);
	//assert(shader->shader.nparam == 0);
	//assert(shader->shader.ninput == 0);

	memset(&shader_ctx, 0, sizeof(shader_ctx));
	nouveau_llvm_context_init(&shader_ctx.nouveau_bld);
	bld_base = &shader_ctx.nouveau_bld.soa.bld_base;

	tgsi_scan_shader(tokens, &shader_info);

	//shader->shader.uses_kill = shader_info.uses_kill;
	//shader->shader.uses_instanceid = shader_info.uses_instanceid;
	bld_base->info = &shader_info;
	bld_base->emit_fetch_funcs[TGSI_FILE_CONSTANT] = fetch_constant;

#if 0
	bld_base->op_actions[TGSI_OPCODE_TEX] = tex_action;
	bld_base->op_actions[TGSI_OPCODE_TEX2] = tex_action;
	bld_base->op_actions[TGSI_OPCODE_TXB] = txb_action;
	bld_base->op_actions[TGSI_OPCODE_TXB2] = txb_action;
#if HAVE_LLVM >= 0x0304
	bld_base->op_actions[TGSI_OPCODE_TXD] = txd_action;
#endif
	bld_base->op_actions[TGSI_OPCODE_TXF] = txf_action;
	bld_base->op_actions[TGSI_OPCODE_TXL] = txl_action;
	bld_base->op_actions[TGSI_OPCODE_TXL2] = txl_action;
	bld_base->op_actions[TGSI_OPCODE_TXP] = tex_action;
	bld_base->op_actions[TGSI_OPCODE_TXQ] = txq_action;
	bld_base->op_actions[TGSI_OPCODE_TG4] = tg4_action;
	bld_base->op_actions[TGSI_OPCODE_LODQ] = lodq_action;

#if HAVE_LLVM >= 0x0304
	bld_base->op_actions[TGSI_OPCODE_DDX].emit = si_llvm_emit_ddxy;
	bld_base->op_actions[TGSI_OPCODE_DDY].emit = si_llvm_emit_ddxy;
#endif

	bld_base->op_actions[TGSI_OPCODE_EMIT].emit = si_llvm_emit_vertex;
	bld_base->op_actions[TGSI_OPCODE_ENDPRIM].emit = si_llvm_emit_primitive;
#endif

	shader_ctx.nouveau_bld.load_system_value = declare_system_value;
	shader_ctx.tokens = tokens;
	tgsi_parse_init(&shader_ctx.parse, shader_ctx.tokens);
	//shader_ctx.shader = shader;
	shader_ctx.type = shader_ctx.parse.FullHeader.Processor.Processor;

	shader_ctx.nouveau_bld.load_input = llvm_load_input;

#if 0
	switch (shader_ctx.type) {
	case TGSI_PROCESSOR_VERTEX:
		shader_ctx.nouveau_bld.load_input = declare_input_vs;	
		if (shader->key.vs.as_es) {
			shader_ctx.gs_for_vs = &sctx->gs_shader->current->shader;
			bld_base->emit_epilogue = si_llvm_emit_es_epilogue;
		} else {
			bld_base->emit_epilogue = si_llvm_emit_vs_epilogue;
		}
		break;
	case TGSI_PROCESSOR_GEOMETRY: {
		int i;

		shader_ctx.nouveau_bld.load_input = declare_input_gs;
		bld_base->emit_fetch_funcs[TGSI_FILE_INPUT] = fetch_input_gs;
		bld_base->emit_epilogue = si_llvm_emit_gs_epilogue;

		for (i = 0; i < shader_info.num_properties; i++) {
			switch (shader_info.properties[i].name) {
			case TGSI_PROPERTY_GS_INPUT_PRIM:
				shader->shader.gs_input_prim = shader_info.properties[i].data[0];
				break;
			case TGSI_PROPERTY_GS_OUTPUT_PRIM:
				shader->shader.gs_output_prim = shader_info.properties[i].data[0];
				break;
			case TGSI_PROPERTY_GS_MAX_OUTPUT_VERTICES:
				shader->shader.gs_max_out_vertices = shader_info.properties[i].data[0];
				break;
			}
		}
		break;
	}
	case TGSI_PROCESSOR_FRAGMENT: {
		int i;

		shader_ctx.nouveau_bld.load_input = declare_input_fs;
		bld_base->emit_epilogue = si_llvm_emit_fs_epilogue;
		shader->shader.ps_conservative_z = V_02880C_EXPORT_ANY_Z;

		for (i = 0; i < shader_info.num_properties; i++) {
			switch (shader_info.properties[i].name) {
			case TGSI_PROPERTY_FS_DEPTH_LAYOUT:
				switch (shader_info.properties[i].data[0]) {
				case TGSI_FS_DEPTH_LAYOUT_GREATER:
					shader->shader.ps_conservative_z = V_02880C_EXPORT_GREATER_THAN_Z;
					break;
				case TGSI_FS_DEPTH_LAYOUT_LESS:
					shader->shader.ps_conservative_z = V_02880C_EXPORT_LESS_THAN_Z;
					break;
				}
				break;
			}
		}
		break;
	}
	default:
		assert(!"Unsupported shader type");
		return -1;
	}
#endif

	create_meta_data(&shader_ctx);
	create_function(&shader_ctx);
	preload_constants(&shader_ctx);
	//preload_samplers(&shader_ctx);
	//preload_streamout_buffers(&shader_ctx);

	if (shader_ctx.type == TGSI_PROCESSOR_GEOMETRY) {
		shader_ctx.gs_next_vertex =
			lp_build_alloca(bld_base->base.gallivm,
					bld_base->uint_bld.elem_type, "");
	}

	if (!lp_build_tgsi_llvm(bld_base, tokens)) {
		fprintf(stderr, "Failed to translate shader from TGSI to LLVM\n");
		goto out;
	}

	nouveau_llvm_finalize_module(&shader_ctx.nouveau_bld);

	mod = bld_base->base.gallivm->module;
#if 0
	r = si_compile_llvm(sctx, shader, mod);
	if (r) {
		fprintf(stderr, "LLVM failed to compile shader\n");
		goto out;
	}
#endif

	nouveau_llvm_dispose(&shader_ctx.nouveau_bld);

#if 0
	if (shader_ctx.type == TGSI_PROCESSOR_GEOMETRY) {
		shader->gs_copy_shader = CALLOC_STRUCT(si_pipe_shader);
		shader->gs_copy_shader->selector = shader->selector;
		shader->gs_copy_shader->key = shader->key;
		shader_ctx.shader = shader->gs_copy_shader;
		if ((r = si_generate_gs_copy_shader(sctx, &shader_ctx, dump))) {
			free(shader->gs_copy_shader);
			shader->gs_copy_shader = NULL;
			goto out;
		}
	}
#endif

	tgsi_parse_free(&shader_ctx.parse);

out:
	for (int i = 0; i < NUM_CONST_BUFFERS; i++)
		FREE(shader_ctx.constants[i]);
	FREE(shader_ctx.resources);
	FREE(shader_ctx.samplers);

	return mod;
}


LLVMModuleRef nouveau_tgsi_llvm_bak(const struct tgsi_token * tokens);
LLVMModuleRef nouveau_tgsi_llvm_bak(const struct tgsi_token * tokens)
{
	struct tgsi_shader_info shader_info;
	struct nouveau_shader_context shader_ctx;
	struct nouveau_llvm_context * ctx;
	struct lp_build_tgsi_context * bld_base;
	
	memset(&shader_ctx, 0, sizeof(shader_ctx));
	nouveau_llvm_context_init(&shader_ctx.nouveau_bld);
	ctx = &shader_ctx.nouveau_bld;
	bld_base = &ctx->soa.bld_base;

	tgsi_scan_shader(tokens, &shader_info);
	nouveau_llvm_context_init(ctx);
#if HAVE_LLVM >= 0x0304
	LLVMTypeRef Arguments[32];
	unsigned ArgumentsCount = 0;
	for (unsigned i = 0; i < shader_info.num_inputs; i++)
		Arguments[ArgumentsCount++] = LLVMVectorType(bld_base->base.elem_type, 4);
	nouveau_llvm_create_func(ctx, Arguments, ArgumentsCount);
	for (unsigned i = 0; i < shader_info.num_inputs; i++) {
		LLVMValueRef P = LLVMGetParam(ctx->main_fn, i);
		LLVMAddAttribute(P, LLVMInRegAttribute);
	}
#else
	nouveau_llvm_create_func(ctx, NULL, 0);
#endif

	bld_base->info = &shader_info;
	bld_base->userdata = ctx;
	bld_base->emit_fetch_funcs[TGSI_FILE_CONSTANT] = fetch_constant;
	//bld_base->emit_prologue = llvm_emit_prologue;
	//bld_base->emit_epilogue = llvm_emit_epilogue;
	ctx->userdata = ctx;
	ctx->load_input = llvm_load_input;
	//ctx->load_system_value = llvm_load_system_value;

	//bld_base->op_actions[TGSI_OPCODE_DP2] = dot_action;
	//bld_base->op_actions[TGSI_OPCODE_DP3] = dot_action;
	//bld_base->op_actions[TGSI_OPCODE_DP4] = dot_action;
	//bld_base->op_actions[TGSI_OPCODE_DPH] = dot_action;
	/*bld_base->op_actions[TGSI_OPCODE_DDX].emit = llvm_emit_tex;
	bld_base->op_actions[TGSI_OPCODE_DDY].emit = llvm_emit_tex;
	bld_base->op_actions[TGSI_OPCODE_TEX].emit = llvm_emit_tex;
	bld_base->op_actions[TGSI_OPCODE_TEX2].emit = llvm_emit_tex;
	bld_base->op_actions[TGSI_OPCODE_TXB].emit = llvm_emit_tex;
	bld_base->op_actions[TGSI_OPCODE_TXB2].emit = llvm_emit_tex;
	bld_base->op_actions[TGSI_OPCODE_TXD].emit = llvm_emit_tex;
	bld_base->op_actions[TGSI_OPCODE_TXL].emit = llvm_emit_tex;
	bld_base->op_actions[TGSI_OPCODE_TXL2].emit = llvm_emit_tex;
	bld_base->op_actions[TGSI_OPCODE_TXF].emit = llvm_emit_tex;
	bld_base->op_actions[TGSI_OPCODE_TXQ].emit = llvm_emit_tex;
	bld_base->op_actions[TGSI_OPCODE_TXP].emit = llvm_emit_tex;
	bld_base->op_actions[TGSI_OPCODE_CMP].emit = emit_cndlt;*/

	lp_build_tgsi_llvm(bld_base, tokens);

	nouveau_llvm_finalize_module(ctx);

	return ctx->gallivm.module;
}

