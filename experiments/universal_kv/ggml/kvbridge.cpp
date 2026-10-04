// libkvbridge: the universal KV translation engine's products as ggml graphs, on any ggml backend
// (Vulkan on AMD/NVIDIA/Intel GPUs, or ggml's CPU backend), behind a small C API that
// experiments/universal_kv/backends/ggml_backend.py loads with ctypes. See README.md.
//
// What runs on the device, per call:
//
//   kvb_encode       source rows -> hub latents z: the rows of each layer's cell span are uploaded and
//                    gathered (get_rows), K de-rotated with host cos/sin tables, centred on the layer
//                    mean, projected on its stage-one basis, and the concatenated stage one through the
//                    fused encoder (plus a bias: minus the target decoder's z_mean)
//   kvb_section      z -> one cache section's rows: z[mapped] @ W_section + mean, written at their rows,
//                    the other rows the mean or given fill rows
//   kvb_cache_rows   section rows -> the cache: K rotated with host cos/sin tables, cast to the cache
//                    dtype (f16/f32), checked finite in it, copied into the caller's host arrays
//
// Every matrix product is f16 x f16 with f32 accumulation (GGML_PREC_F32): ggml's default precision
// accumulates in f16 on Vulkan (1.2e-2 relative error on the hub's products), this sits at the f16
// input-rounding floor (a few 1e-4). Weights are stored transposed (ggml_mul_mat's A is [k, m]) and in
// f16. RoPE uses cos/sin tables the caller computes (kvstate's angles, float64 cos/sin): ggml's own
// rope kernels evaluate sin/cos in f32 on the GPU, which is far from kvstate at 262K positions; the
// device only does the rotation's multiply-adds (forward x0*c - x1*s, x1*c + x0*s; the inverse with
// c/(c^2+s^2), -s/(c^2+s^2), as kvstate.Rope.invert).
//
// Graphs run with ggml_backend_graph_compute on one backend (no scheduler, so nodes run in the order
// they were built: writes into views of a scratch tensor precede the nodes reading it). Device arrays
// (z, section rows, scratch) come from a pool of buffers that kvb_release empties; the graph
// allocator's buffer is released with it. No tensor is larger than the device's largest buffer
// (ggml_backend_buft_get_max_size: 1 GiB on ggml-vulkan): the loops bound their rows per graph by
// it, and a larger array is refused as out of memory (the caller splits arrays into row blocks).
// Every entry point returns 0 (or a handle) on success and sets a thread-local message read with
// kvb_last_error otherwise.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#    define KVB_API extern "C" __declspec(dllexport)
#else
#    define KVB_API extern "C" __attribute__((visibility("default")))
#endif

#define KVB_VERSION 1

enum { KVB_F32 = 0, KVB_F16 = 1, KVB_I32 = 2 };

// return codes
enum { KVB_OK = 0, KVB_ERROR = -1, KVB_OUT_OF_MEMORY = -2, KVB_NOT_FINITE = 1 };

static thread_local std::string g_error;
static thread_local int g_code = 0;
static std::mutex g_log_mutex;
static std::string g_last_log;             // the last ggml warning/error, appended to failures

struct kvb_failure : std::runtime_error {
    int code;
    kvb_failure(const std::string & message, int code = KVB_ERROR) : std::runtime_error(message), code(code) {}
};

static std::string format(const char * fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    return buf;
}

static void log_callback(enum ggml_log_level level, const char * text, void *) {
    static const bool verbose = getenv("KVBRIDGE_VERBOSE") != nullptr;
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        g_last_log = text;
    }
    if (verbose || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

static ggml_type to_ggml(int dtype) {
    switch (dtype) {
        case KVB_F32: return GGML_TYPE_F32;
        case KVB_F16: return GGML_TYPE_F16;
        case KVB_I32: return GGML_TYPE_I32;
        default: throw kvb_failure(format("unknown dtype %d", dtype));
    }
}

// -- context, buffers and device arrays ---------------------------------------------------------

// a buffer holds a resident weight (allocated exactly, never pooled) or a device array (pooled)
enum placement { WEIGHT = 0, ARRAY = 1 };

struct kvb_ctx {
    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t dev = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    // an integrated GPU's pinned host memory: device memory the host maps is uncached (a plain read
    // of it runs at ~0.2 GB/s), so reads go through a device copy into a pinned staging buffer
    ggml_backend_buffer_type_t host_buft = nullptr;
    ggml_backend_buffer_t staging = nullptr;
    ggml_gallocr_t galloc = nullptr;
    std::recursive_mutex mu;
    int64_t used[2] = { 0, 0 };             // bytes in use per placement
    int64_t pooled = 0;                     // free array buffers kept for reuse
    int64_t peak = 0;                       // device bytes: weights + arrays + their pool + graph allocator
    std::vector<ggml_backend_buffer_t> pool;
    std::set<struct kvb_tensor *> live;     // tensors handed out: kvb_free frees what is left

    int64_t graph_bytes() const { return galloc ? (int64_t) ggml_gallocr_get_buffer_size(galloc, 0) : 0; }
    int64_t device_bytes() const { return used[WEIGHT] + used[ARRAY] + pooled + graph_bytes(); }
    int64_t staging_bytes() const { return staging ? (int64_t) ggml_backend_buffer_get_size(staging) : 0; }
    void note() { peak = std::max(peak, device_bytes()); }
};

struct kvb_tensor {
    ggml_context * meta = nullptr;      // holds t's metadata
    ggml_tensor * t = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    int place = ARRAY;
    int64_t rows = 0, cols = 0;
    int dtype = KVB_F32;
};

static const int64_t POOL_GRANULE = 1 << 20;
static const int64_t POOL_LIMIT = (int64_t) 4 << 30;   // free buffers kept beyond this are released

static void pool_clear(kvb_ctx * kc) {
    for (auto * buffer : kc->pool) {
        ggml_backend_buffer_free(buffer);
    }
    kc->pool.clear();
    kc->pooled = 0;
}

static ggml_backend_buffer_t buffer_take(kvb_ctx * kc, int64_t size, int place) {
    auto & pool = kc->pool;
    if (place != WEIGHT) {
        // the smallest free buffer that fits without wasting more than its size again
        int best = -1;
        for (size_t i = 0; i < pool.size(); ++i) {
            int64_t have = (int64_t) ggml_backend_buffer_get_size(pool[i]);
            if (have >= size && have <= 2 * size + 32 * POOL_GRANULE &&
                (best < 0 || have < (int64_t) ggml_backend_buffer_get_size(pool[best]))) {
                best = (int) i;
            }
        }
        if (best >= 0) {
            ggml_backend_buffer_t buffer = pool[best];
            pool.erase(pool.begin() + best);
            int64_t have = (int64_t) ggml_backend_buffer_get_size(buffer);
            kc->pooled -= have;
            kc->used[place] += have;
            return buffer;
        }
        size = (size + POOL_GRANULE - 1) / POOL_GRANULE * POOL_GRANULE;
    }
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(kc->buft, (size_t) std::max<int64_t>(size, 64));
    if (buffer == nullptr && !kc->pool.empty()) {
        // the pool may hold what the device needs: release it and retry once
        pool_clear(kc);
        buffer = ggml_backend_buft_alloc_buffer(kc->buft, (size_t) std::max<int64_t>(size, 64));
    }
    if (buffer == nullptr) {
        throw kvb_failure(format("out of device memory: a buffer of %lld MiB", (long long) (size >> 20)),
                          KVB_OUT_OF_MEMORY);
    }
    kc->used[place] += (int64_t) ggml_backend_buffer_get_size(buffer);
    kc->note();
    return buffer;
}

static void buffer_give(kvb_ctx * kc, ggml_backend_buffer_t buffer, int place) {
    int64_t size = (int64_t) ggml_backend_buffer_get_size(buffer);
    kc->used[place] -= size;
    if (place == WEIGHT) {
        ggml_backend_buffer_free(buffer);
        return;
    }
    auto & pool = kc->pool;
    pool.push_back(buffer);
    kc->pooled += size;
    while (kc->pooled > POOL_LIMIT && !pool.empty()) {
        kc->pooled -= (int64_t) ggml_backend_buffer_get_size(pool.front());
        ggml_backend_buffer_free(pool.front());
        pool.erase(pool.begin());
    }
}

// a device tensor: 2-D [rows, cols] (ggml ne = [cols, rows]), or 1-D of `cols` elements with rows = 0
static kvb_tensor * tensor_new(kvb_ctx * kc, int dtype, int64_t rows, int64_t cols, int place) {
    if (cols <= 0 || rows < 0) {
        throw kvb_failure(format("a device array needs positive dimensions, not [%lld, %lld]", (long long) rows,
                                 (long long) cols));
    }
    ggml_backend_buffer_type_t buft = kc->buft;
    auto * out = new kvb_tensor();
    ggml_init_params params = { ggml_tensor_overhead() * 2, nullptr, true };
    out->meta = ggml_init(params);
    if (out->meta == nullptr) {
        delete out;
        throw std::bad_alloc();
    }
    out->t = rows > 0 ? ggml_new_tensor_2d(out->meta, to_ggml(dtype), cols, rows)
                      : ggml_new_tensor_1d(out->meta, to_ggml(dtype), cols);
    out->rows = rows;
    out->cols = cols;
    out->dtype = dtype;
    out->place = place;
    try {
        int64_t size = (int64_t) ggml_backend_buft_get_alloc_size(buft, out->t);
        if ((size_t) size > ggml_backend_buft_get_max_size(buft)) {
            // out of memory, not a device fault: the engine retries on numpy and keeps the device
            throw kvb_failure(format("a device array of %lld MiB exceeds the device's largest buffer (%lld MiB)",
                                     (long long) (size >> 20), (long long) (ggml_backend_buft_get_max_size(buft) >> 20)),
                              KVB_OUT_OF_MEMORY);
        }
        out->buffer = buffer_take(kc, size, place);
        if (ggml_backend_tensor_alloc(out->buffer, out->t, ggml_backend_buffer_get_base(out->buffer)) != GGML_STATUS_SUCCESS) {
            throw kvb_failure("could not place a tensor in its buffer");
        }
        kc->live.insert(out);
    } catch (...) {
        if (out->buffer) {
            buffer_give(kc, out->buffer, place);
        }
        ggml_free(out->meta);
        delete out;
        throw;
    }
    return out;
}

// rows per graph (at most chunk_rows) whose tensors of row_bytes a row fit the device's largest buffer
static int64_t rows_within(const kvb_ctx * kc, int64_t chunk_rows, int64_t row_bytes) {
    const size_t largest = ggml_backend_buft_get_max_size(kc->buft);
    const size_t fit = largest / (size_t) std::max<int64_t>(row_bytes, 1);
    return std::max<int64_t>(1, (int64_t) std::min<size_t>((size_t) chunk_rows, fit));
}

static void tensor_free(kvb_ctx * kc, kvb_tensor * t) {
    if (t == nullptr) {
        return;
    }
    if (t->buffer) {
        buffer_give(kc, t->buffer, t->place);
    }
    kc->live.erase(t);
    ggml_free(t->meta);
    delete t;
}

struct piece {
    size_t offset;      // in the tensor
    size_t size;
    void * host;
};

// memcpy of pieces (src base + offset -> host) on up to 8 threads, split so each takes a similar share
static void copy_pieces(const char * src, size_t base, const std::vector<piece> & pieces) {
    size_t total = 0;
    for (const auto & p : pieces) {
        total += p.size;
    }
    const size_t grain = (size_t) 8 << 20;
    const int threads = (int) std::min<size_t>(8, (total + grain - 1) / grain);
    if (threads <= 1) {
        for (const auto & p : pieces) {
            memcpy(p.host, src + (p.offset - base), p.size);
        }
        return;
    }
    // thread i copies bytes [i * share, (i + 1) * share) of the concatenated pieces
    const size_t share = (total + threads - 1) / threads;
    std::vector<std::thread> pool;
    for (int i = 0; i < threads; ++i) {
        pool.emplace_back([&, i] {
            size_t begin = i * share, end = std::min(total, begin + share), at = 0;
            for (const auto & p : pieces) {
                size_t lo = std::max(begin, at), hi = std::min(end, at + p.size);
                if (lo < hi) {
                    memcpy((char *) p.host + (lo - at), src + (p.offset - base) + (lo - at), hi - lo);
                }
                at += p.size;
            }
        });
    }
    for (auto & thread : pool) {
        thread.join();
    }
}

// pieces of a tensor into host memory (pieces in increasing, non-overlapping tensor order): on an
// integrated GPU one device copy of the range they span into a pinned staging buffer per 256 MiB,
// then copies from cached memory on a few threads; else ggml's own reads
static void read_pieces(kvb_ctx * kc, kvb_tensor * t, const std::vector<piece> & pieces) {
    if (kc->host_buft == nullptr) {
        for (const auto & p : pieces) {
            ggml_backend_tensor_get(t->t, p.host, p.offset, p.size);
        }
        return;
    }
    const size_t limit = (size_t) 256 << 20;
    size_t i = 0;
    while (i < pieces.size()) {
        // the pieces [i, j) whose span fits the staging buffer (a piece larger than it is split)
        const size_t first = pieces[i].offset;
        size_t j = i, end = first;
        while (j < pieces.size() && (j == i || pieces[j].offset + pieces[j].size - first <= limit)) {
            end = pieces[j].offset + pieces[j].size;
            ++j;
        }
        size_t span = end - first;
        std::vector<piece> group(pieces.begin() + i, pieces.begin() + j);
        if (span > limit) {             // one large piece: in parts
            group.clear();
            for (size_t done = 0; done < pieces[i].size; done += limit) {
                group.push_back({ pieces[i].offset + done, std::min(limit, pieces[i].size - done),
                                  (char *) pieces[i].host + done });
            }
        }
        for (size_t k = 0; k < group.size();) {
            size_t lo = group[k].offset, hi = lo, m = k;
            while (m < group.size() && group[m].offset + group[m].size - lo <= limit) {
                hi = group[m].offset + group[m].size;
                ++m;
            }
            if (kc->staging == nullptr || ggml_backend_buffer_get_size(kc->staging) < hi - lo) {
                if (kc->staging) {
                    ggml_backend_buffer_free(kc->staging);
                    kc->staging = nullptr;
                }
                kc->staging = ggml_backend_buft_alloc_buffer(kc->host_buft, std::max<size_t>(hi - lo, (size_t) 16 << 20));
                if (kc->staging == nullptr) {
                    throw kvb_failure("out of pinned host memory for a read", KVB_OUT_OF_MEMORY);
                }
                kc->note();
            }
            char * stage = (char *) ggml_backend_buffer_get_base(kc->staging);
            ggml_backend_tensor_get_async(kc->backend, t->t, stage, lo, hi - lo);
            ggml_backend_synchronize(kc->backend);
            copy_pieces(stage, lo, std::vector<piece>(group.begin() + k, group.begin() + m));
            k = m;
        }
        i = j;
    }
}

static void read_bytes(kvb_ctx * kc, kvb_tensor * t, size_t offset, size_t size, void * host) {
    read_pieces(kc, t, { { offset, size, host } });
}

// frees a scratch tensor when leaving a scope
struct scratch {
    kvb_ctx * kc;
    kvb_tensor * t;
    scratch(kvb_ctx * kc, kvb_tensor * t) : kc(kc), t(t) {}
    ~scratch() { tensor_free(kc, t); }
};

// -- graphs -------------------------------------------------------------------------------------

// a graph allocator of its own for one call (uploads: their source blocks would otherwise stay in the
// context's allocator, up to 256 MiB, until kvb_release), freed when leaving the scope
struct transient_alloc {
    ggml_gallocr_t galloc;
    explicit transient_alloc(kvb_ctx * kc) : galloc(ggml_gallocr_new(kc->buft)) {
        if (galloc == nullptr) {
            throw std::bad_alloc();
        }
    }
    ~transient_alloc() { ggml_gallocr_free(galloc); }
};

struct graph {
    kvb_ctx * kc;
    ggml_gallocr_t galloc;                  // the context's, or a transient_alloc's
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    std::vector<std::pair<ggml_tensor *, const void *>> inputs;
    std::map<std::pair<const void *, int64_t>, ggml_tensor *> shared;      // inputs uploaded once per graph

    explicit graph(kvb_ctx * kc, size_t max_nodes = 16384, ggml_gallocr_t galloc = nullptr)
        : kc(kc), galloc(galloc ? galloc : kc->galloc) {
        ggml_init_params params = { ggml_tensor_overhead() * max_nodes * 3 + ggml_graph_overhead_custom(max_nodes, false),
                                    nullptr, true };
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::bad_alloc();
        }
        gf = ggml_new_graph_custom(ctx, max_nodes, false);
    }
    ~graph() { ggml_free(ctx); }

    // a host array uploaded before the graph runs (the same pointer and size share one tensor)
    ggml_tensor * input(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, const void * data) {
        int64_t bytes = (int64_t) ggml_row_size(type, ne0) * ne1 * ne2 * ne3;
        auto key = std::make_pair(data, bytes);
        auto found = shared.find(key);
        if (found != shared.end() && ggml_nelements(found->second) == ne0 * ne1 * ne2 * ne3 &&
            found->second->type == type) {
            return ggml_reshape_4d(ctx, found->second, ne0, ne1, ne2, ne3);
        }
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
        ggml_set_input(t);
        inputs.emplace_back(t, data);
        shared[key] = t;
        return t;
    }

    void expand(ggml_tensor * t) { ggml_build_forward_expand(gf, t); }

    void run() {
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (!ggml_backend_supports_op(kc->backend, node)) {
                throw kvb_failure(format("%s does not support %s (%s, %s -> %s)", ggml_backend_name(kc->backend),
                                         ggml_op_desc(node), node->src[0] ? ggml_type_name(node->src[0]->type) : "-",
                                         node->src[1] ? ggml_type_name(node->src[1]->type) : "-",
                                         ggml_type_name(node->type)));
            }
        }
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            throw kvb_failure("out of device memory for a compute graph", KVB_OUT_OF_MEMORY);
        }
        kc->note();
        if (galloc != kc->galloc) {
            kc->peak = std::max(kc->peak, kc->device_bytes() + (int64_t) ggml_gallocr_get_buffer_size(galloc, 0));
        }
        for (auto & item : inputs) {
            ggml_backend_tensor_set(item.first, item.second, 0, ggml_nbytes(item.first));
        }
        ggml_status status = ggml_backend_graph_compute(kc->backend, gf);
        if (status == GGML_STATUS_ALLOC_FAILED) {
            throw kvb_failure("out of device memory while computing", KVB_OUT_OF_MEMORY);
        }
        if (status != GGML_STATUS_SUCCESS) {
            throw kvb_failure(format("graph compute failed (status %d)", (int) status));
        }
    }
};

// f16 x f16 with f32 accumulation: a [k, m] (weights), b [k, n] (activations, cast to f16) -> [m, n] f32
static ggml_tensor * product(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    if (b->type != GGML_TYPE_F16) {
        b = ggml_cast(ctx, b, GGML_TYPE_F16);
    }
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    return out;
}

struct kvb_rope {
    int32_t neox;           // 1: NEOX/IMROPE pairs (i, i + n_rot/2); 0: NORM pairs (2i, 2i + 1)
    int32_t n_rot;
    int32_t offset;         // the rotated dims are [offset, offset + n_rot) of a head
    int32_t pad;
    const float * cos;      // [n, n_rot/2]: c (forward) or c/(c^2+s^2) (inverse)
    const float * sin;      // [n, n_rot/2]: s (forward) or -s/(c^2+s^2) (inverse)
};

static void check_rope(const kvb_rope & rope, int64_t head_dim) {
    if (rope.n_rot <= 0 || rope.n_rot % 2 || rope.offset < 0 || rope.offset + rope.n_rot > head_dim ||
        rope.cos == nullptr || rope.sin == nullptr) {
        throw kvb_failure(format("invalid rope (n_rot %d, offset %d, head dim %lld)", rope.n_rot, rope.offset,
                                 (long long) head_dim));
    }
}

// x f32 [head_dim, heads, n] (rows may be strided, a view) rotated pair by pair with the tables' rows
// [row0, row0 + n): a = x0 * C - x1 * S, b = x1 * C + x0 * S on the rotated dims only. Returns the
// rotated block [n_rot, heads, n] itself (contiguous) when the rope covers the whole head, else x
// with the block copied over its rotated dims (in place: nodes built after this read x after the
// copy, since graphs run in the order they were built)
static ggml_tensor * rotate(graph & g, ggml_tensor * x, const kvb_rope & rope, int64_t row0) {
    ggml_context * ctx = g.ctx;
    const int64_t dim = x->ne[0], heads = x->ne[1], n = x->ne[2], half = rope.n_rot / 2;
    const size_t es = sizeof(float);
    check_rope(rope, dim);
    ggml_tensor * x0;
    ggml_tensor * x1;
    ggml_tensor * c;
    ggml_tensor * s;
    if (rope.neox) {
        c = ggml_reshape_3d(ctx, g.input(GGML_TYPE_F32, half, 1, n, 1, rope.cos + row0 * half), half, 1, n);
        s = ggml_reshape_3d(ctx, g.input(GGML_TYPE_F32, half, 1, n, 1, rope.sin + row0 * half), half, 1, n);
        x0 = ggml_view_3d(ctx, x, half, heads, n, x->nb[1], x->nb[2], rope.offset * es);
        x1 = ggml_view_3d(ctx, x, half, heads, n, x->nb[1], x->nb[2], (rope.offset + half) * es);
    } else {
        c = g.input(GGML_TYPE_F32, 1, half, 1, n, rope.cos + row0 * half);
        s = g.input(GGML_TYPE_F32, 1, half, 1, n, rope.sin + row0 * half);
        x0 = ggml_view_4d(ctx, x, 1, half, heads, n, 2 * es, x->nb[1], x->nb[2], rope.offset * es);
        x1 = ggml_view_4d(ctx, x, 1, half, heads, n, 2 * es, x->nb[1], x->nb[2], (rope.offset + 1) * es);
    }
    ggml_tensor * a = ggml_sub(ctx, ggml_mul(ctx, x0, c), ggml_mul(ctx, x1, s));
    ggml_tensor * b = ggml_add(ctx, ggml_mul(ctx, x1, c), ggml_mul(ctx, x0, s));
    // NEOX: [a | b] per head; NORM: a and b interleaved (pairs 2i, 2i + 1)
    ggml_tensor * block = ggml_reshape_3d(ctx, ggml_concat(ctx, a, b, 0), rope.n_rot, heads, n);
    if (rope.offset == 0 && rope.n_rot == dim) {
        return block;
    }
    g.expand(ggml_cpy(ctx, block, ggml_view_3d(ctx, x, rope.n_rot, heads, n, x->nb[1], x->nb[2], rope.offset * es)));
    return x;
}

#define KVB_GUARD_BEGIN try {
#define KVB_GUARD_END(fail)                                                         \
    }                                                                               \
    catch (const kvb_failure & e) {                                                 \
        g_error = e.what();                                                         \
        g_code = e.code;                                                            \
        return e.code == KVB_OUT_OF_MEMORY ? (fail##_oom) : (fail);                 \
    }                                                                               \
    catch (const std::bad_alloc &) {                                                \
        g_error = "out of host memory";                                             \
        g_code = KVB_OUT_OF_MEMORY;                                                 \
        return fail##_oom;                                                          \
    }                                                                               \
    catch (const std::exception & e) {                                              \
        std::lock_guard<std::mutex> lock(g_log_mutex);                              \
        g_error = std::string(e.what()) + (g_last_log.empty() ? "" : " (" + g_last_log + ")"); \
        g_code = KVB_ERROR;                                                         \
        return fail;                                                                \
    }

static const int int_fail = KVB_ERROR;
static const int int_fail_oom = KVB_OUT_OF_MEMORY;
static kvb_tensor * const ptr_fail = nullptr;
static kvb_tensor * const ptr_fail_oom = nullptr;
static struct kvb_ctx * const ctx_fail = nullptr;
static struct kvb_ctx * const ctx_fail_oom = nullptr;

// -- library and devices ------------------------------------------------------------------------

KVB_API int kvb_version(void) { return KVB_VERSION; }

KVB_API const char * kvb_last_error(void) { return g_error.c_str(); }

// the code of the last failure on this thread: KVB_ERROR or KVB_OUT_OF_MEMORY
KVB_API int kvb_last_error_code(void) { return g_code; }

KVB_API const char * kvb_ggml_version(void) {
    static std::string text = std::string(ggml_version()) + " (" + ggml_commit() + ")";
    return text.c_str();
}

static void init_logging() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_log_set(log_callback, nullptr); });
}

// 0 cpu, 1 gpu, 2 igpu, 3 accel, 4 other
static int device_kind(ggml_backend_dev_t dev) {
    switch (ggml_backend_dev_type(dev)) {
        case GGML_BACKEND_DEVICE_TYPE_CPU: return 0;
        case GGML_BACKEND_DEVICE_TYPE_GPU: return 1;
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return 2;
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return 3;
        default: return 4;
    }
}

KVB_API int kvb_device_count(void) {
    init_logging();
    return (int) ggml_backend_dev_count();
}

// name / description of device `index` (ggml's order), its kind (device_kind) and memory
KVB_API int kvb_device_info(int index, char * name, size_t name_len, char * description, size_t description_len,
                            int * kind, uint64_t * free_bytes, uint64_t * total_bytes) {
    init_logging();
    if (index < 0 || (size_t) index >= ggml_backend_dev_count()) {
        g_error = format("no device %d", index);
        return KVB_ERROR;
    }
    ggml_backend_dev_t dev = ggml_backend_dev_get(index);
    snprintf(name, name_len, "%s", ggml_backend_dev_name(dev));
    snprintf(description, description_len, "%s", ggml_backend_dev_description(dev));
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(dev, &free, &total);
    *kind = device_kind(dev);
    *free_bytes = free;
    *total_bytes = total;
    return KVB_OK;
}

// device: NULL, "" or "auto" (the first GPU, else the CPU), "cpu", a ggml device name ("Vulkan0") or
// a device index (kvb_device_info's); n_threads: the CPU backend's threads (0: all hardware threads)
KVB_API kvb_ctx * kvb_init(const char * device, int n_threads) {
    init_logging();
    KVB_GUARD_BEGIN
    std::string want = device ? device : "";
    for (auto & ch : want) {
        ch = (char) tolower((unsigned char) ch);
    }
    ggml_backend_dev_t dev = nullptr;
    if (want.empty() || want == "auto") {
        for (size_t i = 0; i < ggml_backend_dev_count() && dev == nullptr; ++i) {
            int kind = device_kind(ggml_backend_dev_get(i));
            if (kind == 1 || kind == 2) {
                dev = ggml_backend_dev_get(i);
            }
        }
        if (dev == nullptr) {
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }
    } else if (want == "cpu") {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    } else if (want.size() <= 4 &&
               std::all_of(want.begin(), want.end(), [](char ch) { return isdigit((unsigned char) ch) != 0; })) {
        size_t index = (size_t) std::stoul(want);
        dev = index < ggml_backend_dev_count() ? ggml_backend_dev_get(index) : nullptr;
    } else {
        for (size_t i = 0; i < ggml_backend_dev_count() && dev == nullptr; ++i) {
            std::string name = ggml_backend_dev_name(ggml_backend_dev_get(i));
            for (auto & ch : name) {
                ch = (char) tolower((unsigned char) ch);
            }
            if (name == want) {
                dev = ggml_backend_dev_get(i);
            }
        }
    }
    if (dev == nullptr) {
        throw kvb_failure(format("no ggml device %s", want.c_str()));
    }
    auto * kc = new kvb_ctx();
    kc->dev = dev;
    kc->backend = ggml_backend_dev_init(dev, nullptr);
    if (kc->backend == nullptr) {
        delete kc;
        throw kvb_failure(format("could not initialize %s", ggml_backend_dev_name(dev)));
    }
    if (ggml_backend_is_cpu(kc->backend)) {
        int threads = n_threads > 0 ? n_threads : (int) std::max(1u, std::thread::hardware_concurrency());
        ggml_backend_cpu_set_n_threads(kc->backend, threads);
    }
    kc->buft = ggml_backend_get_default_buffer_type(kc->backend);
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
        kc->host_buft = ggml_backend_dev_host_buffer_type(dev);
    }
    kc->galloc = ggml_gallocr_new(kc->buft);
    return kc;
    KVB_GUARD_END(ctx_fail)
}

// frees the context, with every tensor of it still alive (their handles are invalid afterwards)
KVB_API void kvb_free(kvb_ctx * kc) {
    if (kc == nullptr) {
        return;
    }
    while (!kc->live.empty()) {
        tensor_free(kc, *kc->live.begin());
    }
    pool_clear(kc);
    if (kc->staging) {
        ggml_backend_buffer_free(kc->staging);
    }
    if (kc->galloc) {
        ggml_gallocr_free(kc->galloc);
    }
    ggml_backend_free(kc->backend);
    delete kc;
}

// the context's device name and description, its kind (kvb_device_info) and the largest buffer
KVB_API int kvb_info(kvb_ctx * kc, char * name, size_t name_len, char * description, size_t description_len,
                     int * kind, uint64_t * max_buffer) {
    snprintf(name, name_len, "%s", ggml_backend_dev_name(kc->dev));
    snprintf(description, description_len, "%s", ggml_backend_dev_description(kc->dev));
    *kind = device_kind(kc->dev);
    *max_buffer = ggml_backend_buft_get_max_size(kc->buft);
    return KVB_OK;
}

// bytes: out[0] weights, [1] live device arrays, [2] pooled free device buffers, [3] the graph
// allocator's buffer, [4] the peak of [0] + [1] + [2] + [3] since kvb_reset_peak, [5] device free,
// [6] device total, [7] the pinned host staging buffer of reads
KVB_API void kvb_memory(kvb_ctx * kc, int64_t * out) {
    std::lock_guard<std::recursive_mutex> lock(kc->mu);
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(kc->dev, &free, &total);
    out[0] = kc->used[WEIGHT];
    out[1] = kc->used[ARRAY];
    out[2] = kc->pooled;
    out[3] = kc->graph_bytes();
    out[4] = std::max(kc->peak, kc->device_bytes());
    out[5] = (int64_t) free;
    out[6] = (int64_t) total;
    out[7] = kc->staging_bytes();
}

KVB_API void kvb_reset_peak(kvb_ctx * kc) {
    std::lock_guard<std::recursive_mutex> lock(kc->mu);
    kc->peak = kc->device_bytes();
}

// frees the pooled buffers, the read staging and the graph allocator's buffer (arrays in use and
// weights stay)
KVB_API void kvb_release(kvb_ctx * kc) {
    std::lock_guard<std::recursive_mutex> lock(kc->mu);
    pool_clear(kc);
    if (kc->staging) {
        ggml_backend_buffer_free(kc->staging);
        kc->staging = nullptr;
    }
    if (kc->galloc) {
        ggml_gallocr_free(kc->galloc);
    }
    kc->galloc = ggml_gallocr_new(kc->buft);
}

KVB_API int kvb_synchronize(kvb_ctx * kc) {
    std::lock_guard<std::recursive_mutex> lock(kc->mu);
    KVB_GUARD_BEGIN
    ggml_backend_synchronize(kc->backend);
    return KVB_OK;
    KVB_GUARD_END(int_fail)
}

// -- device arrays ------------------------------------------------------------------------------

// an uninitialized device array [rows, cols] of dtype; weight: accounted as a resident weight (not pooled)
KVB_API kvb_tensor * kvb_new(kvb_ctx * kc, int dtype, int64_t rows, int64_t cols, int weight) {
    std::lock_guard<std::recursive_mutex> lock(kc->mu);
    KVB_GUARD_BEGIN
    if (rows <= 0) {
        throw kvb_failure("a device array needs rows");
    }
    return tensor_new(kc, dtype, rows, cols, weight ? WEIGHT : ARRAY);
    KVB_GUARD_END(ptr_fail)
}

// a host C-contiguous [rows, cols] array of src_dtype on the device as dtype: [rows, cols], or with
// transpose its transpose [cols, rows] (converted and transposed by the device)
KVB_API kvb_tensor * kvb_upload(kvb_ctx * kc, const void * data, int src_dtype, int64_t rows, int64_t cols, int dtype,
                                int transpose, int weight) {
    std::lock_guard<std::recursive_mutex> lock(kc->mu);
    KVB_GUARD_BEGIN
    if (rows <= 0 || cols <= 0 || data == nullptr) {
        throw kvb_failure("kvb_upload needs a nonempty array");
    }
    if (src_dtype == KVB_I32 || dtype == KVB_I32) {
        if (src_dtype != dtype || transpose) {
            throw kvb_failure("int32 arrays are uploaded as they are");
        }
    }
    kvb_tensor * out = transpose ? tensor_new(kc, dtype, cols, rows, weight ? WEIGHT : ARRAY)
                                 : tensor_new(kc, dtype, rows, cols, weight ? WEIGHT : ARRAY);
    try {
        if (!transpose && src_dtype == dtype) {
            ggml_backend_tensor_set(out->t, data, 0, ggml_nbytes(out->t));
            return out;
        }
        // in row blocks of at most 256 MiB of source, through an allocator freed afterwards
        int64_t row_bytes = (int64_t) ggml_row_size(to_ggml(src_dtype), cols);
        int64_t block = std::max<int64_t>(1, ((int64_t) 256 << 20) / row_bytes);
        transient_alloc alloc(kc);
        for (int64_t r0 = 0; r0 < rows; r0 += block) {
            int64_t nr = std::min(block, rows - r0);
            graph g(kc, 64, alloc.galloc);
            ggml_tensor * src = g.input(to_ggml(src_dtype), cols, nr, 1, 1, (const char *) data + r0 * row_bytes);
            ggml_tensor * dst;
            if (transpose) {
                // rows [r0, r0 + nr) of the source are columns [r0, r0 + nr) of the stored transpose
                dst = ggml_view_2d(g.ctx, out->t, nr, cols, out->t->nb[1], r0 * ggml_element_size(out->t));
                src = ggml_transpose(g.ctx, src);
            } else {
                dst = ggml_view_2d(g.ctx, out->t, cols, nr, out->t->nb[1], r0 * out->t->nb[1]);
            }
            g.expand(ggml_cpy(g.ctx, src, dst));
            g.run();
        }
    } catch (...) {
        tensor_free(kc, out);
        throw;
    }
    return out;
    KVB_GUARD_END(ptr_fail)
}

// n host matrices at once (kvb_upload's arguments per matrix, one dtype and placement): out[i] the
// device copy of data[i], converted and (with transpose) transposed by the device in one graph per
// 256 MiB of source rather than one per matrix (a decoder side holds one matrix per layer), through
// an allocator freed afterwards
KVB_API int kvb_upload_many(kvb_ctx * kc, int n, const void * const * data, const int * src_dtype, const int64_t * rows,
                            const int64_t * cols, int dtype, int transpose, int weight, kvb_tensor ** out) {
    std::lock_guard<std::recursive_mutex> lock(kc->mu);
    for (int i = 0; i < n; ++i) {
        out[i] = nullptr;
    }
    KVB_GUARD_BEGIN
    try {
        const int64_t limit = (int64_t) 256 << 20;
        for (int i = 0; i < n; ++i) {
            if (rows[i] <= 0 || cols[i] <= 0 || data[i] == nullptr || src_dtype[i] == KVB_I32 || dtype == KVB_I32) {
                throw kvb_failure(format("kvb_upload_many: matrix %d is empty or not floating point", i));
            }
            out[i] = transpose ? tensor_new(kc, dtype, cols[i], rows[i], weight ? WEIGHT : ARRAY)
                               : tensor_new(kc, dtype, rows[i], cols[i], weight ? WEIGHT : ARRAY);
        }
        transient_alloc alloc(kc);
        for (int i = 0; i < n;) {
            graph g(kc, 64 + 8 * (size_t) n, alloc.galloc);
            int64_t bytes = 0;
            int j = i;
            for (; j < n; ++j) {
                int64_t size = (int64_t) ggml_row_size(to_ggml(src_dtype[j]), cols[j]) * rows[j];
                if (j > i && bytes + size > limit) {
                    break;
                }
                bytes += size;
                if (!transpose && src_dtype[j] == dtype) {
                    ggml_backend_tensor_set(out[j]->t, data[j], 0, ggml_nbytes(out[j]->t));
                    continue;
                }
                ggml_tensor * src = g.input(to_ggml(src_dtype[j]), cols[j], rows[j], 1, 1, data[j]);
                g.expand(ggml_cpy(g.ctx, transpose ? ggml_transpose(g.ctx, src) : src, out[j]->t));
            }
            if (ggml_graph_n_nodes(g.gf) > 0) {
                g.run();
            }
            i = j;
        }
    } catch (...) {
        for (int i = 0; i < n; ++i) {
            tensor_free(kc, out[i]);
            out[i] = nullptr;
        }
        throw;
    }
    return KVB_OK;
    KVB_GUARD_END(int_fail)
}

// [sum rows, cols]: rows [row0[i], row0[i] + rows[i]) of parts[i], one after the other (same dtype and cols)
KVB_API kvb_tensor * kvb_concat_rows(kvb_ctx * kc, int n, kvb_tensor * const * parts, const int64_t * row0,
                                     const int64_t * rows, int weight) {
    std::lock_guard<std::recursive_mutex> lock(kc->mu);
    KVB_GUARD_BEGIN
    if (n <= 0) {
        throw kvb_failure("kvb_concat_rows needs parts");
    }
    int64_t total = 0;
    for (int i = 0; i < n; ++i) {
        if (parts[i]->dtype != parts[0]->dtype || parts[i]->cols != parts[0]->cols || parts[i]->rows <= 0 ||
            row0[i] < 0 || rows[i] <= 0 || row0[i] + rows[i] > parts[i]->rows) {
            throw kvb_failure(format("kvb_concat_rows: part %d does not fit", i));
        }
        total += rows[i];
    }
    kvb_tensor * out = tensor_new(kc, parts[0]->dtype, total, parts[0]->cols, weight ? WEIGHT : ARRAY);
    try {
        graph g(kc, 4 * n + 16);
        int64_t at = 0;
        for (int i = 0; i < n; ++i) {
            ggml_tensor * src = ggml_view_2d(g.ctx, parts[i]->t, parts[i]->cols, rows[i], parts[i]->t->nb[1],
                                             row0[i] * parts[i]->t->nb[1]);
            ggml_tensor * dst = ggml_view_2d(g.ctx, out->t, out->cols, rows[i], out->t->nb[1], at * out->t->nb[1]);
            g.expand(ggml_cpy(g.ctx, src, dst));
            at += rows[i];
        }
        g.run();
    } catch (...) {
        tensor_free(kc, out);
        throw;
    }
    return out;
    KVB_GUARD_END(ptr_fail)
}

// rows [row0, row0 + rows) of a device array into (read) or from (write) a C-contiguous host array
KVB_API int kvb_read(kvb_ctx * kc, kvb_tensor * t, int64_t row0, int64_t rows, void * host) {
    std::lock_guard<std::recursive_mutex> lock(kc->mu);
    KVB_GUARD_BEGIN
    if (row0 < 0 || rows < 0 || row0 + rows > std::max<int64_t>(t->rows, 1)) {
        throw kvb_failure("kvb_read: rows out of range");
    }
    size_t row_bytes = ggml_row_size(t->t->type, t->cols);
    if (rows) {
        read_bytes(kc, t, row0 * row_bytes, rows * row_bytes, host);
    }
    return KVB_OK;
    KVB_GUARD_END(int_fail)
}

KVB_API int kvb_write(kvb_ctx * kc, kvb_tensor * t, int64_t row0, int64_t rows, const void * host) {
    std::lock_guard<std::recursive_mutex> lock(kc->mu);
    KVB_GUARD_BEGIN
    if (row0 < 0 || rows < 0 || row0 + rows > std::max<int64_t>(t->rows, 1)) {
        throw kvb_failure("kvb_write: rows out of range");
    }
    size_t row_bytes = ggml_row_size(t->t->type, t->cols);
    if (rows) {
        ggml_backend_tensor_set(t->t, host, row0 * row_bytes, rows * row_bytes);
    }
    return KVB_OK;
    KVB_GUARD_END(int_fail)
}

// rows, cols, dtype and bytes held (its buffer)
KVB_API void kvb_shape(kvb_tensor * t, int64_t * rows, int64_t * cols, int * dtype, int64_t * nbytes) {
    *rows = t->rows;
    *cols = t->cols;
    *dtype = t->dtype;
    *nbytes = t->buffer ? (int64_t) ggml_backend_buffer_get_size(t->buffer) : 0;
}

KVB_API void kvb_tensor_free(kvb_ctx * kc, kvb_tensor * t) {
    if (kc == nullptr || t == nullptr) {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(kc->mu);
    tensor_free(kc, t);
}

// -- the engine's products ----------------------------------------------------------------------

struct kvb_encode_layer {
    const void * rows;          // host [span, heads * head_dim], C-contiguous, dtype
    int32_t dtype;              // KVB_F16 or KVB_F32
    int32_t rope;               // index into the ropes (K de-rotation), or -1
    int64_t span;               // rows held
    const int32_t * index;      // [n]: the row in the span of each position (shared by layers of one stream)
    int64_t heads, head_dim;
    kvb_tensor * mean;          // f32 [1, width] (or 1-D)
    kvb_tensor * basis;         // f16 [rank, width]: the stage-one basis, transposed
};

static void check_index(const int32_t * index, int64_t n, int64_t span, std::map<const int32_t *, int64_t> & seen) {
    auto found = seen.find(index);
    if (found != seen.end() && found->second == span) {
        return;
    }
    for (int64_t i = 0; i < n; ++i) {
        if (index[i] < 0 || index[i] >= span) {
            throw kvb_failure(format("a row index %d is outside a span of %lld rows", index[i], (long long) span));
        }
    }
    seen[index] = span;
}

static int64_t width_of(const kvb_tensor * t) { return t->rows > 0 ? t->rows * t->cols : t->cols; }

// z rows [out_row0, out_row0 + n) of out (f32 [rows, D]) = stage1 @ weight (+ bias), stage1 the
// concatenated (rows_l - mean_l) @ basis_l of the source layers; weight f16 [D, sum rank] (the fused
// encoder transposed), bias f32 [D] or NULL. Returns KVB_NOT_FINITE when a latent is not finite (an
// f16 overflow of the centred rows or the stage one: inf or NaN in z), as torch_backend.encode checks.
KVB_API int kvb_encode(kvb_ctx * kc, int64_t n, int n_layers, const kvb_encode_layer * layers, int n_ropes,
                       const kvb_rope * ropes, kvb_tensor * weight, kvb_tensor * bias, kvb_tensor * out,
                       int64_t out_row0) {
    std::lock_guard<std::recursive_mutex> lock(kc->mu);
    KVB_GUARD_BEGIN
    if (n <= 0 || n_layers <= 0) {
        throw kvb_failure("kvb_encode needs positions and layers");
    }
    const int64_t dim = weight->rows, ranks = weight->cols;
    if (weight->dtype != KVB_F16 || out->dtype != KVB_F32 || out->cols != dim || out_row0 < 0 ||
        out_row0 + n > out->rows || (bias && (bias->dtype != KVB_F32 || width_of(bias) != dim))) {
        throw kvb_failure("kvb_encode: the encoder weight, bias or output does not fit");
    }
    std::map<const int32_t *, int64_t> seen;
    int64_t total_rank = 0;
    for (int l = 0; l < n_layers; ++l) {
        const auto & layer = layers[l];
        const int64_t width = layer.heads * layer.head_dim;
        if (layer.rows == nullptr || layer.span <= 0 || layer.index == nullptr || width <= 0 ||
            (layer.dtype != KVB_F16 && layer.dtype != KVB_F32) || layer.basis->cols != width ||
            layer.basis->dtype != KVB_F16 || width_of(layer.mean) != width || layer.mean->dtype != KVB_F32 ||
            layer.rope >= n_ropes) {
            throw kvb_failure(format("kvb_encode: source layer %d does not fit its basis", l));
        }
        check_index(layer.index, n, layer.span, seen);
        if (layer.rope >= 0) {
            check_rope(ropes[layer.rope], layer.head_dim);
        }
        total_rank += layer.basis->rows;
    }
    if (total_rank != ranks) {
        throw kvb_failure(format("kvb_encode: the bases hold %lld ranks, the encoder %lld", (long long) total_rank,
                                 (long long) ranks));
    }
    scratch stage1(kc, tensor_new(kc, KVB_F16, n, ranks, ARRAY));
    graph g(kc);
    int64_t at = 0;
    for (int l = 0; l < n_layers; ++l) {
        const auto & layer = layers[l];
        const int64_t width = layer.heads * layer.head_dim, rank = layer.basis->rows;
        ggml_tensor * span = g.input(to_ggml(layer.dtype), width, layer.span, 1, 1, layer.rows);
        ggml_tensor * index = g.input(GGML_TYPE_I32, n, 1, 1, 1, layer.index);
        ggml_tensor * x = ggml_get_rows(g.ctx, span, index);                       // f32 [width, n]
        if (layer.rope >= 0) {
            x = rotate(g, ggml_reshape_3d(g.ctx, x, layer.head_dim, layer.heads, n), ropes[layer.rope], 0);
            x = ggml_reshape_2d(g.ctx, x, width, n);            // the gathered rows are contiguous
        }
        x = ggml_sub(g.ctx, x, ggml_reshape_1d(g.ctx, layer.mean->t, width));
        ggml_tensor * part = product(g.ctx, layer.basis->t, x);                     // f32 [rank, n]
        g.expand(ggml_cpy(g.ctx, part, ggml_view_2d(g.ctx, stage1.t->t, rank, n, stage1.t->t->nb[1],
                                                    at * ggml_element_size(stage1.t->t))));
        at += rank;
    }
    ggml_tensor * z = product(g.ctx, weight->t, stage1.t->t);                       // f32 [D, n]
    if (bias) {
        z = ggml_add(g.ctx, z, ggml_reshape_1d(g.ctx, bias->t, dim));
    }
    g.expand(ggml_cpy(g.ctx, z, ggml_view_2d(g.ctx, out->t, dim, n, out->t->nb[1], out_row0 * out->t->nb[1])));
    // finite: the sum of z, by rows first (as kvb_cache_rows' check)
    ggml_tensor * check = ggml_sum(g.ctx, ggml_sum_rows(g.ctx, z));
    ggml_set_output(check);
    g.expand(check);
    g.run();
    float total = 0.0f;
    ggml_backend_tensor_get(check, &total, 0, sizeof(float));
    return std::isfinite(total) ? KVB_OK : KVB_NOT_FINITE;
    KVB_GUARD_END(int_fail)
}

// one cache section's rows, out f32 [n_rows, sum d]: rows mapped_rows[i] = z[mapped[i]] @ weight + mean
// (z - z_mean first when z_mean is given), rows fill_rows[j] = fill[j] (fill [n_fill, sum d]) or the mean.
// weight f16 [sum d, D] (the section's decoders transposed), mean f32 [sum d]. mapped / mapped_rows
// NULL: 0 .. n_mapped - 1. Mapped rows are decoded chunk_rows at a time (fewer when a chunk's product
// would not fit the device's largest buffer).
KVB_API int kvb_section(kvb_ctx * kc, kvb_tensor * z, int64_t n_mapped, const int32_t * mapped,
                        const int32_t * mapped_rows, kvb_tensor * z_mean, kvb_tensor * weight, kvb_tensor * mean,
                        int64_t n_fill, const int32_t * fill_rows, const float * fill, kvb_tensor * out,
                        int64_t chunk_rows) {
    std::lock_guard<std::recursive_mutex> lock(kc->mu);
    KVB_GUARD_BEGIN
    const int64_t width = weight->rows, dim = weight->cols, n_rows = out->rows;
    if (weight->dtype != KVB_F16 || width_of(mean) != width || mean->dtype != KVB_F32 || out->dtype != KVB_F32 ||
        out->cols != width || n_mapped < 0 || n_fill < 0 || chunk_rows <= 0 ||
        (n_mapped && (z == nullptr || z->dtype != KVB_F32 || z->cols != dim)) ||
        (z_mean && (z_mean->dtype != KVB_F32 || width_of(z_mean) != dim)) || (n_fill && fill_rows == nullptr)) {
        throw kvb_failure("kvb_section: the decoder, latents or output do not fit");
    }
    for (int64_t i = 0; i < n_mapped; ++i) {
        int64_t source = mapped ? mapped[i] : i, row = mapped_rows ? mapped_rows[i] : i;
        if (source < 0 || source >= z->rows || row < 0 || row >= n_rows) {
            throw kvb_failure("kvb_section: a mapped row is out of range");
        }
    }
    for (int64_t i = 0; i < n_fill; ++i) {
        if (fill_rows[i] < 0 || fill_rows[i] >= n_rows) {
            throw kvb_failure("kvb_section: a fill row is out of range");
        }
    }
    chunk_rows = rows_within(kc, chunk_rows, (int64_t) sizeof(float) * std::max(width, dim));
    for (int64_t c0 = 0; c0 < n_mapped; c0 += chunk_rows) {
        const int64_t m = std::min(chunk_rows, n_mapped - c0);
        graph g(kc, 256);
        ggml_tensor * rows;
        if (mapped == nullptr) {
            rows = ggml_view_2d(g.ctx, z->t, dim, m, z->t->nb[1], c0 * z->t->nb[1]);
        } else {
            rows = ggml_get_rows(g.ctx, z->t, g.input(GGML_TYPE_I32, m, 1, 1, 1, mapped + c0));
        }
        if (z_mean) {
            rows = ggml_sub(g.ctx, rows, ggml_reshape_1d(g.ctx, z_mean->t, dim));
        }
        ggml_tensor * values = product(g.ctx, weight->t, rows);                          // f32 [width, m]
        values = ggml_add(g.ctx, values, ggml_reshape_1d(g.ctx, mean->t, width));
        if (mapped_rows == nullptr) {
            g.expand(ggml_cpy(g.ctx, values, ggml_view_2d(g.ctx, out->t, width, m, out->t->nb[1], c0 * out->t->nb[1])));
        } else {
            g.expand(ggml_set_rows(g.ctx, out->t, values, g.input(GGML_TYPE_I32, m, 1, 1, 1, mapped_rows + c0)));
        }
        g.run();
    }
    for (int64_t c0 = 0; c0 < n_fill; c0 += chunk_rows) {
        const int64_t m = std::min(chunk_rows, n_fill - c0);
        graph g(kc, 64);
        ggml_tensor * values;
        if (fill) {
            values = g.input(GGML_TYPE_F32, width, m, 1, 1, fill + c0 * width);
        } else {
            values = ggml_repeat(g.ctx, ggml_reshape_2d(g.ctx, mean->t, width, 1),
                                 ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, width, m));
        }
        g.expand(ggml_set_rows(g.ctx, out->t, values, g.input(GGML_TYPE_I32, m, 1, 1, 1, fill_rows + c0)));
        g.run();
    }
    return KVB_OK;
    KVB_GUARD_END(int_fail)
}

struct kvb_cache_layer {
    int64_t heads, head_dim;    // the layer's columns of the values, in order
    int32_t rope;               // index into the ropes (K rotation), or -1
    int32_t pad;
    void * out;                 // host [n, heads, head_dim] C-contiguous in the cache dtype
};

// rows [row0, row0 + n) of values (f32 [rows, sum width]) as the cache stores them: K rotated (tables
// of n rows; a rope over part of the head rotates in place in values), cast to dtype (KVB_F16 /
// KVB_F32), into each layer's host array, chunk_rows rows at a time (fewer when the chunk's stored
// rows would not fit the device's largest buffer). Returns KVB_NOT_FINITE (nothing more written)
// when a value is not finite in dtype (an f16 overflow: |x| >= 65520), with *bad the first such layer.
KVB_API int kvb_cache_rows(kvb_ctx * kc, kvb_tensor * values, int64_t row0, int64_t n, int n_layers,
                           const kvb_cache_layer * layers, int n_ropes, const kvb_rope * ropes, int dtype,
                           int64_t chunk_rows, int * bad) {
    std::lock_guard<std::recursive_mutex> lock(kc->mu);
    KVB_GUARD_BEGIN
    *bad = -1;
    int64_t total = 0;
    for (int l = 0; l < n_layers; ++l) {
        if (layers[l].heads <= 0 || layers[l].head_dim <= 0 || layers[l].out == nullptr || layers[l].rope >= n_ropes) {
            throw kvb_failure(format("kvb_cache_rows: layer %d is invalid", l));
        }
        if (layers[l].rope >= 0) {
            check_rope(ropes[layers[l].rope], layers[l].head_dim);
        }
        total += layers[l].heads * layers[l].head_dim;
    }
    if (values->dtype != KVB_F32 || values->cols != total || row0 < 0 || n < 0 || row0 + n > values->rows ||
        (dtype != KVB_F16 && dtype != KVB_F32) || chunk_rows <= 0) {
        throw kvb_failure("kvb_cache_rows: the values do not hold these layers' rows");
    }
    const size_t es = dtype == KVB_F16 ? sizeof(ggml_fp16_t) : sizeof(float);
    int64_t widest = 0;
    for (int l = 0; l < n_layers; ++l) {
        widest = std::max(widest, layers[l].heads * layers[l].head_dim);
    }
    // the stored scratch, and a layer's float32 rotation and check temporaries, within the largest buffer
    chunk_rows = rows_within(kc, rows_within(kc, chunk_rows, total * (int64_t) es), widest * (int64_t) sizeof(float));
    // one check per layer, each in its own 256-byte slot: a view at any other offset would be
    // misaligned on drivers whose minStorageBufferOffsetAlignment exceeds 4 bytes (up to 256)
    const int64_t slot = 256 / sizeof(float);
    for (int64_t c0 = 0; c0 < n; c0 += chunk_rows) {
        const int64_t m = std::min(chunk_rows, n - c0);
        // the stored rows, layer after layer, and the checks
        scratch stored(kc, tensor_new(kc, dtype, 0, m * total, ARRAY));
        scratch sums(kc, tensor_new(kc, KVB_F32, 0, n_layers * slot, ARRAY));
        graph g(kc);
        int64_t col = 0, at = 0;
        for (int l = 0; l < n_layers; ++l) {
            const auto & layer = layers[l];
            const int64_t width = layer.heads * layer.head_dim;
            const size_t first = (row0 + c0) * values->t->nb[1] + col * sizeof(float);
            ggml_tensor * x = ggml_view_2d(g.ctx, values->t, width, m, values->t->nb[1], first);
            if (layer.rope >= 0) {
                // rotated in place in the values (as NumpyBackend.cache_rows), or a new block
                ggml_tensor * heads = ggml_view_3d(g.ctx, values->t, layer.head_dim, layer.heads, m,
                                                   layer.head_dim * sizeof(float), values->t->nb[1], first);
                ggml_tensor * rotated = rotate(g, heads, ropes[layer.rope], c0);
                if (rotated != heads) {
                    x = ggml_reshape_2d(g.ctx, rotated, width, m);
                }
            }
            ggml_tensor * y = ggml_cpy(g.ctx, x, ggml_view_2d(g.ctx, stored.t->t, width, m, width * es, at * es));
            g.expand(y);
            // finite in the cache dtype: the sum of the stored values (an f16 overflow is inf there), by
            // rows first (ggml_sum alone is one workgroup: 1.4 ms per layer of 2048 rows on the 8060S)
            ggml_tensor * stored_f32 = dtype == KVB_F16 ? ggml_cast(g.ctx, y, GGML_TYPE_F32) : y;
            ggml_tensor * check = ggml_sum(g.ctx, ggml_sum_rows(g.ctx, stored_f32));
            g.expand(ggml_cpy(g.ctx, check, ggml_view_1d(g.ctx, sums.t->t, 1, l * slot * sizeof(float))));
            col += width;
            at += m * width;
        }
        g.run();
        std::vector<float> checks(n_layers * slot);
        read_bytes(kc, sums.t, 0, n_layers * slot * sizeof(float), checks.data());
        for (int l = 0; l < n_layers; ++l) {
            if (!std::isfinite(checks[l * slot])) {
                *bad = l;
                return KVB_NOT_FINITE;
            }
        }
        std::vector<piece> pieces;
        at = 0;
        for (int l = 0; l < n_layers; ++l) {
            const int64_t width = layers[l].heads * layers[l].head_dim;
            pieces.push_back({ (size_t) (at * es), (size_t) (m * width * es), (char *) layers[l].out + c0 * width * es });
            at += m * width;
        }
        read_pieces(kc, stored.t, pieces);
    }
    return KVB_OK;
    KVB_GUARD_END(int_fail)
}

// host x [n, heads, head_dim] f32 rotated by one rope's tables (n rows) into host y (tests, and the
// backend's check of each model's RoPE against kvstate)
KVB_API int kvb_rotate(kvb_ctx * kc, const float * x, int64_t n, int64_t heads, int64_t head_dim,
                       const kvb_rope * rope, float * y) {
    std::lock_guard<std::recursive_mutex> lock(kc->mu);
    KVB_GUARD_BEGIN
    if (n <= 0 || heads <= 0 || head_dim <= 0) {
        throw kvb_failure("kvb_rotate needs rows");
    }
    graph g(kc, 64);
    ggml_tensor * input = g.input(GGML_TYPE_F32, head_dim, heads, n, 1, x);
    ggml_tensor * out = ggml_cont(g.ctx, rotate(g, ggml_reshape_3d(g.ctx, input, head_dim, heads, n), *rope, 0));
    ggml_set_output(out);
    g.expand(out);
    g.run();
    ggml_backend_tensor_get(out, y, 0, ggml_nbytes(out));
    return KVB_OK;
    KVB_GUARD_END(int_fail)
}
