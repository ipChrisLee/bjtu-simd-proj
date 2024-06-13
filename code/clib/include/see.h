#pragma once

#include <stdint.h>
#include <stdio.h>

typedef struct {
    size_t size;
    size_t offset;
    int8_t * pool;
} See;

See * see_new(size_t size);
void see_delete(See * p);

void * see_malloc(See * see, size_t size);