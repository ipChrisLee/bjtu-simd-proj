#include "tensor.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>

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

Tensor * tensor_new_with_macllocer(TensorMallocer mallocer, int32_t dim, const int32_t shape[MAX_DIM]) {
	size_t memToUse = sizeof(Tensor) + sizeof(float) * (size_t) get_len_from_shape(dim, shape);
	Tensor * p = mallocer(memToUse);
	p->dim = dim;
	memcpy(p->shape, shape, sizeof(p->shape));
	return p;
}

void tensor_delete_with_freer(TensorFreer freer, Tensor * p) {
	freer(p);
}

bool tensor_same_shape(const Tensor * lhs, const Tensor * rhs) {
	if (lhs->dim != rhs->dim) {
		return false;
	}
	const int32_t D = lhs->dim;
	for (int32_t i = 0; i < D; ++i) {
		if (lhs->shape[i] != rhs->shape[i]) {
			return false;
		}
	}
	return true;
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

void tensor_conv2d(Tensor * dst, const Tensor * src, const Tensor * kernel, const int32_t padding[MAX_DIM], const int32_t stride[MAX_DIM]) {
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
								const int32_t IND_KER[MAX_DIM] = {oc, ic, hKer, wKer};
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
	const float PAD_VAL = -FLT_MAX;
	for (int32_t n = 0; n < N; ++n) {
		for (int32_t c = 0; c < C; ++c) {
			for (int32_t h = 0; h < H_OUT; ++h) {
				for (int32_t w = 0; w < W_OUT; ++w) {
					float maxVal = -FLT_MAX;
					for (int32_t hKer = 0; hKer < H_KER; ++hKer) {
						for (int32_t wKer = 0; wKer < W_KER; ++wKer) {
							const int32_t IND_SRC[MAX_DIM] = {n, c, h * stride[0] + hKer - padding[0], w * stride[1] + wKer - padding[1]};
							maxVal = fmaxf(maxVal, tensor_get_or_default_const(src, IND_SRC, PAD_VAL));
						}
					}
					const int32_t IND_DST[MAX_DIM] = {n, c, h, w};
					*tensor_access(dst, IND_DST) = maxVal;
				}
			}
		}
	}
}

static void dfs_for_softmax(Tensor * dst, int32_t curDim, int32_t axis, DimArray curIndex) {
	if (curDim == dst->dim) {
		float maxVal = -FLT_MAX;
		for (int32_t i = 0; i < dst->shape[axis]; ++i) {
			curIndex[axis] = i;
			float v = *tensor_access_const(dst, curIndex);
			maxVal = fmaxf(v, maxVal);
		}
		float denominator = 0.0f;
		for (int32_t i = 0; i < dst->shape[axis]; ++i) {
			curIndex[axis] = i;
			float v = *tensor_access_const(dst, curIndex) - maxVal;
			v = expf(v);
			*tensor_access(dst, curIndex) = v;
			denominator += v;
		}
		for (int32_t i = 0; i < dst->shape[axis]; ++i) {
			curIndex[axis] = i;
			*tensor_access(dst, curIndex) /= denominator;
		}
	} else if (curDim == axis) {
		dfs_for_softmax(dst, curDim + 1, axis, curIndex);
	} else {
		for (int32_t i = 0; i < dst->shape[curDim]; ++i) {
			curIndex[curDim] = i;
			dfs_for_softmax(dst, curDim + 1, axis, curIndex);
		}
	}
}

void tensor_softmax(Tensor * dst, const Tensor * src, int32_t axis) {
	tensor_softmax_check(dst, src, axis);
	tensor_memcpy(dst, src);
	int32_t curIndex[MAX_DIM] = {};
	dfs_for_softmax(dst, 0, axis, curIndex);
}

static void dfs_for_fc(Tensor * dst, const Tensor * src, const Tensor * weight, int32_t curDim, DimArray curIndex) {
	if (curDim == src->dim - 1) {
		//  (..., srcDimLen) dot (dstDimLen, srcDimLen)
		int32_t dstDimLen = weight->shape[0];
		int32_t srcDimLen = weight->shape[1];
		int32_t weightIndex[MAX_DIM];
		for (int32_t i = 0; i < dstDimLen; ++i) {
			weightIndex[0] = i;
			float s = 0;
			for (int32_t j = 0; j < srcDimLen; ++j) {
				curIndex[curDim] = j;
				float srcVal = *tensor_access_const(src, curIndex);
				weightIndex[1] = j;
				float weightVal = *tensor_access_const(weight, weightIndex);
				s += srcVal * weightVal;
			}
			curIndex[curDim] = i;
			*tensor_access(dst, curIndex) = s;
		}
	} else {
		const int32_t D = dst->shape[curDim];
		for (int32_t i = 0; i < D; ++i) {
			curIndex[curDim] = i;
			dfs_for_fc(dst, src, weight, curDim + 1, curIndex);
		}
	}
}

void tensor_fc(Tensor * dst, const Tensor * src, const Tensor * weight) {
	tensor_fc_check(dst, src, weight);
	int32_t curIndex[MAX_DIM];
	dfs_for_fc(dst, src, weight, 0, curIndex);
}

void tensor_reshape_inplace(Tensor * op, int32_t newDim, DimArray newShape) {
	tensor_reshape_inplace_check(op, newDim, newShape);
	int32_t oldLen = tensor_get_len(op);
	int32_t newLen = 1;
	int32_t minusOneIndex = -1;
	for (int32_t i = 0; i < newDim; ++i) {
		if (newShape[i] == -1) {
			assert(minusOneIndex == -1 && "tensor_reshape_inplace get more than one -1 in newShape.");
			minusOneIndex = i;
		} else {
			newLen *= newShape[i];
		}
	}
	if (minusOneIndex != -1) {
		assert(oldLen % newLen == 0 && "tensor_reshape_inplace shape deduce failed.");
		newShape[minusOneIndex] = oldLen / newLen;
		newLen = oldLen;
	}
	assert(newLen == oldLen && "tensor_reshape_inplace reshape requires same data length.");
	op->dim = newDim;
	memcpy(op->shape, newShape, sizeof(op->shape));
}