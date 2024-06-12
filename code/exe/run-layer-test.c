#include <stdio.h>
#include <stdlib.h>

#include "test-helper.h"
#include "test-layer.h"

/**
 * Usage: run-test "relu" "tInfoPath"
 * 		test `relu` layer, with info path in `tInfoPath`
 */
int main(int argc, char const ** argv) {
	const char * layerName = argv[1];
	const char * tInfoPath = argv[2];

	Msg_t msg;
	TestInfo * tInfo = test_info_read_from(tInfoPath, &msg);
	if (tInfo == NULL) {
		printf("Test failed since reading tInfo failed with msg {%s}.", msg);
		exit(EXIT_FAILURE);
	}

	TestResultInfo * result = test_result_info_new();
	test_relu(result, tInfo);
	if (result->testResult != Test_Passed) {
		printf("Test failed when comparing, with info {%s}.", result->info);
		exit(EXIT_FAILURE);
	}

	return 0;
}