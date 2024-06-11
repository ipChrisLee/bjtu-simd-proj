#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

#define MAX_DIM 4
typedef int32_t DimArray[MAX_DIM];

inline bool is_legal_stride(int32_t dim, const int32_t stride[MAX_DIM]) {
	if (dim <= 0 || dim >= MAX_DIM) {
		return false;
	}
	for (int32_t i = 0; i < dim; ++i) {
		if (stride[i] <= 0) {
			return false;
		}
	}
	return true;
}

typedef struct {
	int32_t dim;
	int32_t shape[MAX_DIM];
	float data[];
} Tensor;

// ------------ ctor dtor ------------
Tensor * tensor_new(int32_t dim, const int32_t shape[MAX_DIM]);
void tensor_delete(Tensor * p);

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

inline int32_t tensor_get_len(const Tensor * p) {
	int32_t len = 1;
	for (int32_t i = 0; i < p->dim; ++i) {
		len *= p->shape[i];
	}
	return len;
}

inline void tensor_get_index_by_len(const Tensor * p, int32_t len, int32_t ind[MAX_DIM]) {
	const int32_t D = p->dim;
	for (int32_t i = D - 1; i >= 0; --i) {
		ind[i] = len % p->shape[i];
		len /= p->shape[i];
	}
}

inline bool tensor_is_valid(const Tensor * p) {
	if (p->dim < 0 || p->dim >= MAX_DIM) {
		return false;
	}
	for (int32_t i = 0; i < p->dim; ++i) {
		if (p->data[i] <= 0) {
			return false;
		}
	}
	return true;
}

inline const float * tensor_access_const(const Tensor * p, const int32_t ind[MAX_DIM]) {
	int32_t pos = 0, step = 1;
	for (int32_t i = p->dim - 1; i >= 0; --i) {
		pos += ind[i] * step;
		step *= p->shape[i];
	}
	return (p->data) + pos;
}

inline float * tensor_access(Tensor * p, const int32_t ind[MAX_DIM]) {
	int32_t pos = 0, step = 1;
	for (int32_t i = p->dim - 1; i >= 0; --i) {
		pos += ind[i] * step;
		step *= p->shape[i];
	}
	return (p->data) + pos;
}

inline float tensor_get_or_default_const(const Tensor * p, const int32_t ind[MAX_DIM], float defaultVal) {
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

/**
 * @brief relu
 */
void tensor_relu(Tensor * dst, const Tensor * src);

inline void tensor_relu_check(Tensor * dst, const Tensor * src) {
	assert(tensor_is_valid(dst) && "Relu dst is not valid.");
	assert(tensor_is_valid(src) && "Relu src is not valid.");
	assert(dst->dim == src->dim && "Relu, src and dst should have same shape.");
	for (int32_t i = 0; i < src->dim; ++i) {
		assert(dst->shape[i] == src->shape[i] && "Relu, src and dst should have same shape.");
	}
}

/**
 * @brief relu inplace.
 */
void tensor_relu_inplace(Tensor * op);

inline void tensor_relu_inplace_check(Tensor * op) {
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
 		 Instead, `kernel` is protential to be (in_channels, out_channels, kernel_size[0], kernel_size[1]) shape.
 * @param padding 2 elements padding in hw.
 * @param stride 2 elements stride in hw.
 */
void tensor_conv2d(Tensor * dst, const Tensor * src, const Tensor * kernel, const int32_t padding[MAX_DIM], int32_t stride[MAX_DIM]);

inline void tensor_conv2d_check(Tensor * dst, const Tensor * src, const Tensor * kernel, const int32_t padding[MAX_DIM], int32_t stride[MAX_DIM]) {
	assert(tensor_is_valid(dst) && "Conv2d dst is not valid.");
	assert(tensor_is_valid(src) && "Conv2d src is not valid.");
	assert(tensor_is_valid(kernel) && "Conv2d dst is not valid.");
	assert(dst->dim == 4 && "Conv2d only supports 4 dim dst.");
	assert(src->dim == 4 && "Conv2d only supports 4 dim src.");
	assert(kernel->dim == 4 && "Conv2d only supports 4 dim kernel.");
	int32_t hIn = src->shape[2], wIn = src->shape[3];
	int32_t hOut = dst->shape[2], wOut = dst->shape[3];
	int32_t hOutExpected = (hIn + 2 * padding[0] - kernel->shape[2]) / stride[0] + 1;
	int32_t wOutExpected = (wIn + 2 * padding[1] - kernel->shape[3]) / stride[1] + 1;
	int32_t inChannels = kernel->shape[0], outChannels = kernel->shape[1];
	assert(inChannels == src->shape[1] && "Conv2d kernel in_channels is not same as src channels.");
	assert(outChannels == dst->shape[1] && "Conv2d kernel out_channels is not same as dst channels.");
	assert(dst->shape[0] == src->shape[0] && "Conv2d dst should have same batch size as src's.");
	assert(dst->shape[1] == kernel->shape[0] && "Conv2d dst channel is not expected.");
	assert(hOut == hOutExpected && "Conv2d dst tensor h dim length is not expected.");
	assert(wOut == wOutExpected && "Conv2d dst tensor w dim length is not expected.");
	assert(is_legal_stride(2, stride) && "Conv2d stride is not legal.");
}

/**
 * @brief maxpool2d to `src` with `kernel`. Conv on the last two dims on src. padding mode is zeros.
 * @note Only supports src and dst with nchw layout. That is: 
 * @note (1, 3, 3) input is not legal since missing dimesion.
 * @note A batch tensor that consists of three channel 256 height 128 width shape image should be of (n, 3, 256, 128) shape.
 * @note Accroding to maxpool2d definition, `kernelSize`, `stride` and `padding` only supports 2 dim. Only the first two values of parameter are used.
 */
void tensor_maxpool2d(Tensor * dst, const Tensor * src, const int32_t kernelSize[MAX_DIM], const int32_t stride[MAX_DIM], const int32_t padding[MAX_DIM]);

inline void tensor_maxpool2d_check(Tensor * dst, const Tensor * src, const int32_t kernelSize[MAX_DIM], const int32_t stride[MAX_DIM], const int32_t padding[MAX_DIM]) {
	assert(tensor_is_valid(dst) && "MaxPool2d dst is not valid.");
	assert(tensor_is_valid(src) && "MaxPool2d src is not valid.");
	assert(dst->dim == 4 && "MaxPool2d only supports 4 dim dst.");
	assert(src->dim == 4 && "MaxPool2d only supports 4 dim src.");
	assert(dst->shape[0] == src->shape[0] && "MaxPool2d require src and dst have same batch size.");
	assert(dst->shape[1] == src->shape[1] && "MaxPool2d require src and dst have same channel number.");
	int32_t hIn = src->shape[2], wIn = src->shape[3];
	int32_t hOut = dst->shape[2], wOut = dst->shape[3];
	int32_t hOutExpected = (hIn + 2 * padding[0] - 1) / stride[0] + 1;
	int32_t wOutExpected = (wIn + 2 * padding[1] - 1) / stride[1] + 1;
	assert(hOut == hOutExpected && "MaxPool2d dst tensor h dim length is not expected.");
	assert(wOut == wOutExpected && "MaxPool2d dst tensor w dim length is not expected.");
}

/**
 * @brief softmax
 */
void tensor_softmax(Tensor * dst, const Tensor * src, int32_t axis);

inline void tensor_softmax_check(Tensor * dst, const Tensor * src, int32_t axis) {
	assert(tensor_is_valid(dst) && "Softmax dst is not valid.");
	assert(tensor_is_valid(src) && "Softmax src is not valid.");
	assert(dst->dim == src->dim && "Softmax, src and dst should have same shape.");
	for (int32_t i = 0; i < src->dim; ++i) {
		assert(dst->shape[i] == src->shape[i] && "Softmax, src and dst should have same shape.");
	}
	assert(axis >= 0 && axis < src->dim && "Softmax softmax axis not legal.");
}