#pragma once

#include <stdint.h>
#include "tensor.h"

typedef enum {
	Test_Passed = 0,
	Result_Unkown,
	Test_Failed,
	Missing_Parameter,
} TestResult;

typedef struct {
	TestResult testResult;
	const char * info;// only allow static string.
} TestResultInfo;

TestResultInfo * test_result_info_new();

void test_result_info_delete(TestResultInfo * p);

typedef struct {
	Tensor * src;
	Tensor * goldenDst;
	Tensor * kernel;
	Tensor * weight;
	DimArray * kernelSize;
	DimArray * padding;
	DimArray * stride;
	int32_t * axis;
	float * rtol;
	float * atol;
} TestInfo;

typedef const char * Msg_t;
TestInfo * test_info_read_from(const char * filePath, Msg_t * msg);

typedef enum {
	Same_In_Tol = 0,
	Diff_On_Shape,
	Out_Of_Tol,
} TensorCompareResult;

const char * tensor_compare_result_str(TensorCompareResult tensorCompareResult);

TensorCompareResult tensor_compare_passed(const Tensor * dst, const Tensor * goldenDst, float * rtol, float * atol, DimArray * diffOn);