#include "tensor.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>

static int32_t get_len_from_shape(int32_t dim, const int32_t shape[MAX_DIM]) {
	int32_t len = 1;
	for (int32_t i = 0; i < dim; ++i) {
		len *= shape[i];
	}
	return len;
}

Tensor * tensor_new(int32_t dim, const int32_t shape[MAX_DIM]) {
	size_t memToUse = sizeof(Tensor) + sizeof(float) * (size_t) get_len_from_shape(dim, shape);
	Tensor * p = malloc(memToUse);
	p->dim = dim;
	memcpy(p->shape, shape, sizeof(p->shape));
	return p;
}

void tensor_delete(Tensor * p) {
	free(p);
}

void tensor_memcpy(Tensor * dst, const Tensor * src) {
	int32_t lenDst = tensor_get_len(dst);
	int32_t lenSrc = tensor_get_len(src);
	assert(lenDst == lenSrc && "tensor_memcpy requires src and dst data length same.");
	memcpy(dst->data, src->data, sizeof(float) * (size_t) lenSrc);
}

void tensor_relu(Tensor * dst, const Tensor * src) {
	tensor_relu_check(dst, src);
	int32_t len = tensor_get_len(src);
	for (int32_t i = 0; i < len; ++i) {
		float v = src->data[i];
		v = fmaxf(v, 0);
		dst->data[i] = v;
	}
}

void tensor_relu_inplace(Tensor * op) {
	tensor_relu_inplace_check(op);
	int32_t len = tensor_get_len(op);
	for (int32_t i = 0; i < len; ++i) {
		float v = op->data[i];
		v = fmaxf(v, 0);
		op->data[i] = v;
	}
}

void tensor_conv2d(Tensor * dst, const Tensor * src, const Tensor * kernel, const int32_t padding[MAX_DIM], int32_t stride[MAX_DIM]) {
	tensor_conv2d_check(dst, src, kernel, padding, stride);
	const int32_t N = src->shape[0];
	const int32_t C_IN = src->shape[1];
	const int32_t H_IN = src->shape[2];
	const int32_t W_IN = src->shape[3];
	const int32_t C_OUT = dst->shape[1];
	const int32_t H_KER = kernel->shape[2];
	const int32_t W_KER = kernel->shape[3];
	const int32_t H_OUT = dst->shape[2];
	const int32_t W_OUT = dst->shape[3];
	const float PAD_VAL = 0;
	for (int32_t b = 0; b < N; ++b) {
		for (int32_t oc = 0; oc < C_OUT; ++oc) {
			for (int32_t h = 0; h < H_OUT; ++h) {
				for (int32_t w = 0; w < W_OUT; ++w) {
					float s = 0;
					const int32_t IND_DST[MAX_DIM] = {b, oc, h, w};
					// cross-correlation
					for (int32_t ic = 0; ic < C_IN; ++ic) {
						for (int32_t hKer = 0; hKer < H_KER; ++hKer) {
							for (int32_t wKer = 0; wKer < W_KER; ++wKer) {
								const int32_t IND_SRC[MAX_DIM] = {b, ic, h * stride[0] + hKer - padding[0], w * stride[1] + wKer - padding[1]};
								float vSrc = tensor_get_or_default_const(src, IND_SRC, PAD_VAL);
								const int32_t IND_KER[MAX_DIM] = {ic, oc, hKer, wKer};
								float vKer = *tensor_access_const(kernel, IND_KER);
								s += vSrc * vKer;
							}
						}
					}
					*tensor_access(dst, IND_DST) = s;
				}
			}
		}
	}
}

void tensor_maxpool2d(Tensor * dst, const Tensor * src, const int32_t kernelSize[MAX_DIM], const int32_t stride[MAX_DIM], const int32_t padding[MAX_DIM]) {
	tensor_maxpool2d_check(dst, src, kernelSize, stride, padding);
	const int32_t N = src->shape[0];
	const int32_t C = src->shape[1];
	const int32_t H_OUT = dst->shape[2];
	const int32_t W_OUT = dst->shape[3];
	const int32_t H_IN = src->shape[2];
	const int32_t W_IN = src->shape[3];
	const int32_t H_KER = kernelSize[0];
	const int32_t W_KER = kernelSize[1];
	const float PAD_VAL = FLT_MIN;
	for (int32_t i = 0; i < N; ++i) {
		for (int32_t j = 0; j < C; ++j) {
			for (int32_t h = 0; h < H_OUT; ++h) {
				for (int32_t w = 0; w < W_OUT; ++w) {
					float maxVal = FLT_MIN;
					for (int32_t m = 0; m < H_KER; ++m) {
						for (int32_t n = 0; n < W_KER; ++n) {
							const int32_t IND_SRC[MAX_DIM] = {i, j, stride[0] * h + m - padding[0], stride[1] * w + n - padding[0]};
							maxVal = fmaxf(maxVal, tensor_get_or_default_const(src, IND_SRC, PAD_VAL));
						}
					}
					const int32_t IND_DST[MAX_DIM] = {i, j, h, w};
					*tensor_access(dst, IND_DST) = maxVal;
				}
			}
		}
	}
}

/**
 * @brief Get the max val in softmax by dfs.
 */
static float get_max_val_in_softmax(const Tensor * p, const int32_t D, const int32_t AXIS, int32_t curIndex[MAX_DIM], int32_t curDim) {
	float maxVal = FLT_MIN;
	if (curDim == AXIS && curDim == D - 1) {
		maxVal = *tensor_access_const(p, curIndex);
	} else if (curDim == AXIS) {
		maxVal = get_max_val_in_softmax(p, D, AXIS, curIndex, curDim + 1);
	} else if (curDim == D - 1) {
		for (int32_t i = 0; i < p->shape[curDim]; ++i) {
			curIndex[curDim] = i;
			float v = *tensor_access_const(p, curIndex);
			maxVal = fmaxf(maxVal, v);
		}
	} else {
		for (int32_t i = 0; i < p->shape[curDim]; ++i) {
			curIndex[curDim] = i;
			float v = get_max_val_in_softmax(p, D, AXIS, curIndex, curDim + 1);
			maxVal = fmaxf(maxVal, v);
		}
	}
	return maxVal;
}

/**
 * @brief Get the max val in softmax by dfs.
 */
static float substract_max_val_and_exp_and_get_denominator_in_softmax(Tensor * p, const int32_t D, const int32_t AXIS, const float MAX_VAL, int32_t curIndex[MAX_DIM], int32_t curDim) {
	float denominator = 0;
	if (curDim == AXIS && curDim == D - 1) {
		float * fp = tensor_access(p, curIndex);
		*fp -= MAX_VAL;
		*fp = expf(*fp);
		denominator += *fp;
	} else if (curDim == AXIS) {
		denominator = substract_max_val_and_exp_and_get_denominator_in_softmax(p, D, AXIS, MAX_VAL, curIndex, curDim + 1);
	} else if (curDim == D - 1) {
		for (int32_t i = 0; i < p->shape[curDim]; ++i) {
			curIndex[curDim] = i;
			float * fp = tensor_access(p, curIndex);
			*fp -= MAX_VAL;
			*fp = expf(*fp);
			denominator += *fp;
		}
	} else {
		for (int32_t i = 0; i < p->shape[curDim]; ++i) {
			curIndex[curDim] = i;
			denominator += substract_max_val_and_exp_and_get_denominator_in_softmax(p, D, AXIS, MAX_VAL, curIndex, curDim + 1);
		}
	}
	return denominator;
}

/**
 * @brief Exp and divide
 */
static void devide_in_softmax(Tensor * p, const int32_t D, const int32_t AXIS, const float DENOMINATOR, int32_t curIndex[MAX_DIM], int32_t curDim) {
	if (curDim == AXIS && curDim == D - 1) {
		*tensor_access(p, curIndex) /= DENOMINATOR;
	} else if (curDim == AXIS) {
		devide_in_softmax(p, D, AXIS, DENOMINATOR, curIndex, curDim + 1);
	} else if (curDim == D - 1) {
		for (int32_t i = 0; i < p->shape[curDim]; ++i) {
			curIndex[curDim] = i;
			*tensor_access(p, curIndex) /= DENOMINATOR;
		}
	} else {
		for (int32_t i = 0; i < p->shape[curDim]; ++i) {
			curIndex[curDim] = i;
			substract_max_val_and_exp_and_get_denominator_in_softmax(p, D, AXIS, DENOMINATOR, curIndex, curDim + 1);
		}
	}
}

void tensor_softmax(Tensor * dst, const Tensor * src, int32_t axis) {
	tensor_softmax_check(dst, src, axis);
	tensor_memcpy(dst, src);
	const int32_t D = src->dim;
	const int32_t L = src->shape[axis];
	int32_t curIndex[MAX_DIM] = {};
	for (int32_t l = 0; l < L; ++l) {
		curIndex[axis] = l;
		float maxVal = get_max_val_in_softmax(src, D, axis, curIndex, 0);
		float denominator = substract_max_val_and_exp_and_get_denominator_in_softmax(dst, D, axis, maxVal, curIndex, 0);
		devide_in_softmax(dst, D, axis, denominator, curIndex, 0);
	}
}