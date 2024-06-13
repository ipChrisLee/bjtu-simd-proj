#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#define MAX_DIM 4
typedef int32_t DimArray[MAX_DIM];
typedef const int32_t ConstDimArray[MAX_DIM];

// ============ Helper Function ============
static inline int32_t get_len_from_shape(int32_t dim, const int32_t shape[MAX_DIM]) {
	int32_t len = 1;
	for (int32_t i = 0; i < dim; ++i) {
		len *= shape[i];
	}
	return len;
}

static inline bool is_legal_stride(int32_t dim, const int32_t stride[MAX_DIM]) {
	if (dim <= 0 || dim > MAX_DIM) {
		return false;
	}
	for (int32_t i = 0; i < dim; ++i) {
		if (stride[i] <= 0) {
			return false;
		}
	}
	return true;
}

static inline bool is_legal_padding(int32_t dim, const int32_t padding[MAX_DIM]) {
	if (dim <= 0 || dim > MAX_DIM) {
		return false;
	}
	for (int32_t i = 0; i < dim; ++i) {
		if (padding[i] < 0) {
			return false;
		}
	}
	return true;
}

static inline bool is_legal_kernel_size(int32_t dim, const int32_t kernelSize[MAX_DIM]) {
	if (dim <= 0 || dim > MAX_DIM) {
		return false;
	}
	for (int32_t i = 0; i < dim; ++i) {
		if (kernelSize[i] <= 0) {
			return false;
		}
	}
	return true;
}

// ============ ShapeInfo ============
typedef struct {
	int32_t dim;
	int32_t shape[MAX_DIM];
} ShapeInfo;

static inline bool shape_info_equal(const ShapeInfo * lhs, const ShapeInfo * rhs) {
	if (lhs->dim != rhs->dim) {
		return false;
	}
	for (int32_t i = 0; i < lhs->dim; ++i) {
		if (lhs->shape[i] != rhs->shape[i]) {
			return false;
		}
	}
	return true;
}

static inline bool shape_info_is_valid(const ShapeInfo * s) {
	if (s->dim <= 0 || s->dim > MAX_DIM) {
		return false;
	}
	for (int32_t i = 0; i < s->dim; ++i) {
		if (s->shape[i] <= 0) {
			return false;
		}
	}
	return true;
}

// ============ Tensor ============
typedef struct {
	int32_t dim;
	int32_t shape[MAX_DIM];
	float data[];
} Tensor;

typedef void * (*TensorMallocer)(size_t);
typedef void (*TensorFreer)(void *);
typedef float (*TensorRander)();

// ------------ ctor dtor ------------
// TODO: maybe need to be static inline.
Tensor * tensor_new(int32_t dim, const int32_t shape[MAX_DIM]);
void tensor_delete(Tensor * p);
Tensor * tensor_new_with_macllocer(TensorMallocer mallocer, int32_t dim, const int32_t shape[MAX_DIM]);
void tensor_delete_with_freer(TensorFreer freer, Tensor * p);


// ------------ binary op method ------------
bool tensor_same_shape(const Tensor * lhs, const Tensor * rhs);

// ------------ member method ------------
/**
 * @brief copy from src data to dst data. Require data length same, or program will down.
 * 
 * @param dst 
 * @param src 
 */
void tensor_memcpy(Tensor * dst, const Tensor * src);

static inline int32_t tensor_get_len(const Tensor * p) {
	int32_t len = 1;
	for (int32_t i = 0; i < p->dim; ++i) {
		len *= p->shape[i];
	}
	return len;
}

static inline void tensor_get_index_by_len(const Tensor * p, int32_t len, int32_t ind[MAX_DIM]) {
	const int32_t D = p->dim;
	for (int32_t i = D - 1; i >= 0; --i) {
		ind[i] = len % p->shape[i];
		len /= p->shape[i];
	}
}

static inline bool tensor_is_valid(const Tensor * p) {
	if (p->dim < 0 || p->dim > MAX_DIM) {
		return false;
	}
	for (int32_t i = 0; i < p->dim; ++i) {
		if (p->shape[i] <= 0) {
			return false;
		}
	}
	return true;
}

static inline const float * tensor_access_const(const Tensor * p, const int32_t ind[MAX_DIM]) {
	int32_t pos = 0, step = 1;
	for (int32_t i = p->dim - 1; i >= 0; --i) {
		assert(ind[i] < p->shape[i] && ind[i] >= 0 && "tensor_access_const out of range.");
		pos += ind[i] * step;
		step *= p->shape[i];
	}
	return (p->data) + pos;
}

static inline float * tensor_access(Tensor * p, const int32_t ind[MAX_DIM]) {
	int32_t pos = 0, step = 1;
	for (int32_t i = p->dim - 1; i >= 0; --i) {
		assert(ind[i] < p->shape[i] && ind[i] >= 0 && "tensor_access out of range.");
		pos += ind[i] * step;
		step *= p->shape[i];
	}
	return (p->data) + pos;
}

static inline float tensor_get_or_default_const(const Tensor * p, const int32_t ind[MAX_DIM], float defaultVal) {
	int32_t pos = 0, step = 1;
	for (int32_t i = p->dim - 1; i >= 0; --i) {
		if (ind[i] >= p->shape[i] || ind[i] < 0) {
			return defaultVal;
		}
		pos += ind[i] * step;
		step *= p->shape[i];
	}
	return p->data[pos];
}

static inline ShapeInfo tensor_to_shape_info(const Tensor * p) {
	ShapeInfo s;
	s.dim = p->dim;
	memcpy(s.shape, p->shape, sizeof(s.shape));
	return s;
}

static inline void tensor_dump(const char * prefix, const Tensor * p) {
	const int32_t D = p->dim;
	const int32_t L = tensor_get_len(p);
	fprintf(stderr, "%s [", prefix);
	for (int32_t i = 0; i < D; ++i) {
		fprintf(stderr, "%" PRId32, p->shape[i]);
	}
	fprintf(stderr, "] {");
	for (int32_t i = 0; i < L; ++i) {
		fprintf(stderr, "%f ", p->data[i]);
	}
	fprintf(stderr, "}\n");
}

// ------------ tensor save and load ------------
static inline Tensor * tensor_load_from_file(TensorMallocer mallocer, const char * filePath) {
	FILE * fp = fopen(filePath, "r");
	int32_t dim, len = 1;
	int32_t shape[MAX_DIM];
	fscanf(fp, "%" PRId32, &dim);
	for (int32_t i = 0; i < dim; ++i) {
		fscanf(fp, "%" PRId32, shape + i);
		len *= shape[i];
	}
	Tensor * p = tensor_new_with_macllocer(mallocer, dim, shape);
	for (int32_t i = 0; i < len; ++i) {
		fscanf(fp, "%f", p->data + i);
	}
	fclose(fp);
	return p;
}

static inline void tensor_save_to_file(const Tensor * p, const char * filePath, bool append) {
	FILE * fp = fopen(filePath, append ? "w+" : "w");
	int32_t len = 1;
	fprintf(fp, "%" PRId32 " ", p->dim);
	for (int32_t i = 0; i < p->dim; ++i) {
		fprintf(fp, "%" PRId32 " ", p->shape[i]);
		len *= p->shape[i];
	}
	fprintf(fp, "\n");
	for (int32_t i = 0; i < len; ++i) {
		fprintf(fp, "%f ", p->data[i]);
	}
	fprintf(fp, "\n");
	fclose(fp);
}

static inline void tensor_fill_rand_data(Tensor * p, TensorRander rander) {
	int32_t len = tensor_get_len(p);
	for (int32_t i = 0; i < len; ++i) {
		p->data[i] = rander();
	}
}

// ------------ tensor layers ------------
/**
 * @brief relu
 */
void tensor_relu(Tensor * dst, const Tensor * src);

static inline ShapeInfo tensor_relu_infer_shape(const ShapeInfo * srcShape) {
	assert(shape_info_is_valid(srcShape) && "tensor_relu_infer_shape srcShape is not valid.");
	return *srcShape;
}

static inline void tensor_relu_check(Tensor * dst, const Tensor * src) {
	ShapeInfo srcShape = tensor_to_shape_info(src);
	ShapeInfo expectedDstShape = tensor_relu_infer_shape(&srcShape);
	ShapeInfo dstShape = tensor_to_shape_info(dst);
	assert(shape_info_equal(&dstShape, &expectedDstShape) && "tensor_relu_check dst shape is not same as expected shape.");
}

static inline Tensor * tensor_relu_layer(TensorMallocer mallocer, const Tensor * src) {
	ShapeInfo srcShape = tensor_to_shape_info(src);
	ShapeInfo dstShape = tensor_relu_infer_shape(&srcShape);
	Tensor * dst = tensor_new_with_macllocer(mallocer, dstShape.dim, dstShape.shape);
	tensor_relu(dst, src);
	return dst;
}

/**
 * @brief relu inplace.
 */
void tensor_relu_inplace(Tensor * op);

static inline void tensor_relu_inplace_check(Tensor * op) {
	assert(tensor_is_valid(op) && "ReluInplace op is not valid.");
}

/**
 * @brief conv2d to `src` with `kernel`. Conv on the last two dims on src. padding mode is zeros.
 * @note Only supports src and dst with nchw layout. That is: 
 * @note (1, 3, 3) input is not legal since missing dimesion.
 * @note A batch tensor that consists of three channel 256 height 128 width shape image should be of (n, 3, 256, 128) shape.
 * @note According to conv2d definition, only `padding` is 2 dim.
 * @note According to conv2d definition, only `kernel` with 4 dim is supported.
 * @note Diff to torch.nn.conv2d, we don't use `in_channels`, `out_channels` and `kernel_size` parameters explicit. 
 		 Instead, `kernel` is protential to be (out_channels, in_channels, kernel_size[0], kernel_size[1]) shape.
 * @param padding 2 elements padding in hw.
 * @param stride 2 elements stride in hw.
 */
void tensor_conv2d(Tensor * dst, const Tensor * src, const Tensor * kernel, const int32_t padding[MAX_DIM], const int32_t stride[MAX_DIM]);

static inline ShapeInfo tensor_conv2d_infer_shape(const ShapeInfo * srcShape, const ShapeInfo * kernelShape, const int32_t padding[MAX_DIM], const int32_t stride[MAX_DIM]) {
	ShapeInfo dstShape;
	dstShape.dim = srcShape->dim;
	assert(shape_info_is_valid(srcShape) && "tensor_conv2d_infer_shape srcShape is not valid.");
	assert(shape_info_is_valid(kernelShape) && "tensor_conv2d_infer_shape kernelShape is not valid.");
	assert(srcShape->dim == 4 && "tensor_conv2d_infer_shape only supports 4 dim src.");
	assert(kernelShape->dim == 4 && "tensor_conv2d_infer_shape only supports 4 dim kernel.");
	assert(srcShape->dim == kernelShape->dim && "tensor_conv2d_infer_shape srcShape and kernelShape should have same dim.");
	assert(srcShape->shape[1] == kernelShape->shape[1] && "tensor_conv2d_infer_shape kernelShape[0] should be inChannels.");
	assert(is_legal_stride(2, stride) && "tensor_conv2d_infer_shape stride is not legal.");
	assert(is_legal_padding(2, padding) && "tensor_conv2d_infer_shape padding is not legal.");
	dstShape.shape[0 /*n*/] = srcShape->shape[0];
	dstShape.shape[1 /*c*/] = kernelShape->shape[0];
	dstShape.shape[2 /*h*/] = (srcShape->shape[2] + 2 * padding[0] - kernelShape->shape[2]) / stride[0] + 1;
	dstShape.shape[3 /*w*/] = (srcShape->shape[3] + 2 * padding[1] - kernelShape->shape[3]) / stride[1] + 1;
	return dstShape;
}

static inline void tensor_conv2d_check(Tensor * dst, const Tensor * src, const Tensor * kernel, const int32_t padding[MAX_DIM], const int32_t stride[MAX_DIM]) {
	ShapeInfo srcShape = tensor_to_shape_info(src);
	ShapeInfo kernelShape = tensor_to_shape_info(kernel);
	ShapeInfo expectedDstShape = tensor_conv2d_infer_shape(&srcShape, &kernelShape, padding, stride);
	ShapeInfo dstShape = tensor_to_shape_info(dst);
	assert(shape_info_equal(&expectedDstShape, &dstShape) && "tensor_conv2d_check dstShape is not equal to expected.");
}

static inline Tensor * tensor_conv2d_layer(TensorMallocer mallocer, const Tensor * src, const Tensor * kernel, const int32_t padding[MAX_DIM], const int32_t stride[MAX_DIM]) {
	ShapeInfo srcShape = tensor_to_shape_info(src);
	ShapeInfo kernelShape = tensor_to_shape_info(kernel);
	ShapeInfo dstShape = tensor_conv2d_infer_shape(&srcShape, &kernelShape, padding, stride);
	Tensor * dst = tensor_new_with_macllocer(mallocer, dstShape.dim, dstShape.shape);
	tensor_conv2d(dst, src, kernel, padding, stride);
	return dst;
}

/**
 * @brief maxpool2d to `src` with `kernel`. Conv on the last two dims on src. padding mode is zeros.
 * @note Only supports src and dst with nchw layout. That is: 
 * @note (1, 3, 3) input is not legal since missing dimesion.
 * @note A batch tensor that consists of three channel 256 height 128 width shape image should be of (n, 3, 256, 128) shape.
 * @note Accroding to maxpool2d definition, `kernelSize`, `stride` and `padding` only supports 2 dim. Only the first two values of parameter are used.
 */
void tensor_maxpool2d(Tensor * dst, const Tensor * src, const int32_t kernelSize[MAX_DIM], const int32_t stride[MAX_DIM], const int32_t padding[MAX_DIM]);

static inline ShapeInfo tensor_maxpool2d_infer_shape(const ShapeInfo * srcShape, const int32_t kernelSize[MAX_DIM], const int32_t stride[MAX_DIM], const int32_t padding[MAX_DIM]) {
	assert(shape_info_is_valid(srcShape) && "tensor_maxpool2d_infer_shape srcShape is not valid.");
	assert(srcShape->dim == 4 && "tensor_maxpool2d_infer_shape srcShape is not valid.");
	assert(is_legal_kernel_size(2, kernelSize) && "tensor_maxpool2d_infer_shape kernelSize is not valid.");
	assert(is_legal_stride(2, stride) && "tensor_maxpool2d_infer_shape stride is not valid.");
	assert(is_legal_padding(2, padding) && "tensor_maxpool2d_infer_shape padding is not valid.");
	ShapeInfo dstShape;
	dstShape.dim = 4;
	dstShape.shape[0 /*n*/] = srcShape->shape[0];
	dstShape.shape[1 /*c*/] = srcShape->shape[1];
	dstShape.shape[2 /*h*/] = (srcShape->shape[2] + 2 * padding[0] - kernelSize[0]) / stride[0] + 1;
	dstShape.shape[3 /*w*/] = (srcShape->shape[3] + 2 * padding[1] - kernelSize[1]) / stride[1] + 1;
	return dstShape;
}

static inline void tensor_maxpool2d_check(Tensor * dst, const Tensor * src, const int32_t kernelSize[MAX_DIM], const int32_t stride[MAX_DIM], const int32_t padding[MAX_DIM]) {
	ShapeInfo srcShape = tensor_to_shape_info(src);
	ShapeInfo expectedDstShape = tensor_maxpool2d_infer_shape(&srcShape, kernelSize, stride, padding);
	ShapeInfo dstShape = tensor_to_shape_info(dst);
	assert(shape_info_equal(&expectedDstShape, &dstShape) && "tensor_maxpool2d_check dstShape is not equal to expected.");
}

static inline Tensor * tensor_maxpool2d_layer(TensorMallocer mallocer, const Tensor * src, const int32_t kernelSize[MAX_DIM], const int32_t stride[MAX_DIM], const int32_t padding[MAX_DIM]) {
	ShapeInfo srcShape = tensor_to_shape_info(src);
	ShapeInfo dstShape = tensor_maxpool2d_infer_shape(&srcShape, kernelSize, stride, padding);
	Tensor * dst = tensor_new_with_macllocer(mallocer, dstShape.dim, dstShape.shape);
	tensor_maxpool2d(dst, src, kernelSize, stride, padding);
	return dst;
}

/**
 * @brief softmax
 */
void tensor_softmax(Tensor * dst, const Tensor * src, int32_t axis);

static inline ShapeInfo tensor_softmax_infer_shape(const ShapeInfo * srcShape, int32_t axis) {
	assert(shape_info_is_valid(srcShape) && "tensor_softmax_infer_shape srcShape is not valid.");
	return *srcShape;
}

static inline void tensor_softmax_check(Tensor * dst, const Tensor * src, int32_t axis) {
	assert(axis >= 0 && axis < src->dim && "tensor_softmax_check softmax axis not legal.");
	ShapeInfo srcShape = tensor_to_shape_info(src);
	ShapeInfo expectedDstShape = tensor_softmax_infer_shape(&srcShape, axis);
	ShapeInfo dstShape = tensor_to_shape_info(dst);
	assert(shape_info_equal(&expectedDstShape, &dstShape) && "tensor_softmax_check dstShape is not equal to expected.");
}

static inline Tensor * tensor_softmax_layer(TensorMallocer mallocer, const Tensor * src, int32_t axis) {
	ShapeInfo srcShape = tensor_to_shape_info(src);
	ShapeInfo dstShape = tensor_softmax_infer_shape(&srcShape, axis);
	Tensor * dst = tensor_new_with_macllocer(mallocer, dstShape.dim, dstShape.shape);
	tensor_softmax(dst, src, axis);
	return dst;
}

/**
 * @brief fc.
 * @note here `weight` is transposed weight. That is, for `y=xA`, `y` is `dst`, `x` is `src`, `A` is transposed `weight`. This make simd optimization simple.
 */
void tensor_fc(Tensor * dst, const Tensor * src, const Tensor * weight);

static inline ShapeInfo tensor_fc_infer_shape(const ShapeInfo * srcShape, const ShapeInfo * weightShape) {
	assert(shape_info_is_valid(srcShape) && "tensor_fc_infer_shape srcShape is not valid.");
	assert(shape_info_is_valid(weightShape) && "tensor_fc_infer_shape weightShape is not valid.");
	assert(weightShape->dim == 2 && "tensor_fc_infer_shape fc weight should have dim 2.");
	assert(srcShape->shape[srcShape->dim - 1] == weightShape->shape[1] && "tensor_fc_infer_shape the last dim of src should have same length as weight first dim.");
	ShapeInfo dstShape;
	dstShape.dim = srcShape->dim;
	memcpy(dstShape.shape, srcShape->shape, sizeof(dstShape.shape));
	dstShape.shape[dstShape.dim - 1] = weightShape->shape[0];
	return dstShape;
}

static inline void tensor_fc_check(Tensor * dst, const Tensor * src, const Tensor * weight) {
	ShapeInfo srcShape = tensor_to_shape_info(src);
	ShapeInfo weightShape = tensor_to_shape_info(weight);
	ShapeInfo expectedDstShape = tensor_fc_infer_shape(&srcShape, &weightShape);
	ShapeInfo dstShape = tensor_to_shape_info(dst);
	assert(shape_info_equal(&expectedDstShape, &dstShape) && "tensor_fc_check dstShape is not equal to expected.");
}

static inline Tensor * tensor_fc_layer(TensorMallocer mallocer, const Tensor * src, const Tensor * weight) {
	ShapeInfo srcShape = tensor_to_shape_info(src);
	ShapeInfo weightShape = tensor_to_shape_info(weight);
	ShapeInfo dstShape = tensor_fc_infer_shape(&srcShape, &weightShape);
	Tensor * dst = tensor_new_with_macllocer(mallocer, dstShape.dim, dstShape.shape);
	tensor_fc(dst, src, weight);
	return dst;
}

/**
 * @brief do reshape inplace.
 *
 * @param newShape new shape. There could be at most one `-1` to be auto deduced length. And after function finished, `-1` will be the deduced length.
 */
void tensor_reshape_inplace(Tensor * op, int32_t newDim, DimArray newShape);

static inline void tensor_reshape_inplace_check(Tensor * op, int32_t newDim, DimArray newShape) {
	assert(tensor_is_valid(op) && "tensor_reshape_inplace_check op is not valid.");
}