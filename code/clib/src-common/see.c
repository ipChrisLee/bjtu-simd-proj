#include "see.h"

#include <assert.h>
#include <stdlib.h>

See * see_new(size_t size) {
	See * p = malloc(sizeof(See));
	p->offset = 0;
	p->size = size;
	p->pool = malloc(size);
	return p;
}

void see_delete(See * p) {
	if (p->pool) {
		free(p->pool);
	}
	free(p);
}

void * see_malloc(See * see, size_t size) {
	void * p = see->pool + see->offset;
	see->offset += size;
	assert(see->offset < see->size && "see_malloc ran out of pool.");
	return p;
}