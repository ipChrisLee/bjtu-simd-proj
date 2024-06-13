/**
 * @file tensor-diff.c
Usage:
		tensor-diff "lhs-tensor" "rhs-tensor" "atol" "rtol"
 */

#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"

int main(int argc, char const ** argv) {
	const char * lhsTensorFilePath = argv[1];
	const char * rhsTensorFilePath = argv[2];
	char * end;
	float atol = strtof(argv[3], &end);
	float rtol = strtof(argv[4], &end);

	Tensor * lhsTensor = tensor_load_from_file(malloc, lhsTensorFilePath);
	Tensor * rhsTensor = tensor_load_from_file(malloc, rhsTensorFilePath);

	bool isDiffOnShape = false, isDiffOnIndex = false;
	DimArray diffIndex;

	bool compareRes = tensor_compare(lhsTensor, rhsTensor, atol, rtol, &isDiffOnIndex, &diffIndex, &isDiffOnShape);

	if (!compareRes) {
		if (isDiffOnShape) {
			fprintf(stderr, "Diff on shape.");
		} else if (isDiffOnIndex) {
			fprintf(stderr, "Diff on index [");
			ShapeInfo s = tensor_to_shape_info(lhsTensor);
			for (int32_t i = 0; i < s.dim; ++i) {
				fprintf(stderr, "%" PRId32 " ", s.shape[i]);
			}
			fprintf(stderr, "].\n");
		} else {
			assert(0 && "Unreachable.");
		}
	} else {
		printf("Compare passed.\n");
	}

	return 0;
}