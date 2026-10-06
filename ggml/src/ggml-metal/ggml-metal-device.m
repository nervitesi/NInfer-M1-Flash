#import "ggml-metal-device.h"
#import "ggml-metal-fusion.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"
#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"

#include <Foundation/Foundation.h>

#include <Metal/Metal.h>

#include <stdatomic.h>
#include <pthread.h>
#include <mach/mach_time.h>

#ifndef TARGET_OS_VISION
#define TARGET_OS_VISION 0
#endif

// create residency sets only on macOS >= 15.0
#if !TARGET_CPU_X86_64 && TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000 || \
    TARGET_OS_IOS && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_TV && __TV_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_VISION && __VISION_OS_VERSION_MAX_ALLOWED >= 200000
#define GGML_METAL_HAS_RESIDENCY_SETS 1
#endif

// overload of MTLGPUFamilyMetalX (not available in some environments)
static const NSInteger MTLGPUFamilyMetal3_GGML = 5001;
static const NSInteger MTLGPUFamilyMetal4_GGML = 5002;

// Metal 4 submission pools: command allocators (one per command buffer in flight) and per-graph slots
// (constants buffer + argument table); a graph with host ops uses one command buffer per segment
#define GGML_METAL4_N_ALLOC     512
#define GGML_METAL4_N_SLOT      4
#define GGML_METAL4_CBUF_SIZE   (32u*1024u*1024u)

// MTLLanguageVersion4_0 is not present in older SDKs
static const NSUInteger MTLLanguageVersion4_0_GGML = 4 << 16;

#if !GGML_METAL_EMBED_LIBRARY
// Here to assist with NSBundle Path Hack
@interface GGMLMetalClass : NSObject
@end
@implementation GGMLMetalClass
@end
#endif

//
// MTLFunctionConstantValues wrapper
//

struct ggml_metal_cv {
    MTLFunctionConstantValues * obj;
};

ggml_metal_cv_t ggml_metal_cv_init(void) {
    ggml_metal_cv_t res = calloc(1, sizeof(struct ggml_metal_cv));

    res->obj = [[MTLFunctionConstantValues alloc] init];

    return res;
}

void ggml_metal_cv_free(ggml_metal_cv_t cv) {
    [cv->obj release];
    free(cv);
}

void ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeShort atIndex:idx];
}

void ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeInt atIndex:idx];
}

void ggml_metal_cv_set_bool(ggml_metal_cv_t cv, bool value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeBool atIndex:idx];
}

//
// MTLComputePipelineState wrapper
//

struct ggml_metal_pipeline {
    id<MTLComputePipelineState> obj;
};

ggml_metal_pipeline_t ggml_metal_pipeline_init(void) {
    ggml_metal_pipeline_t res = calloc(1, sizeof(struct ggml_metal_pipeline));

    *res = (struct ggml_metal_pipeline) {
        /*.obj  =*/ nil,
    };

    return res;
}

void ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline) {
    [pipeline->obj release];

    free(pipeline);
}

int ggml_metal_pipeline_max_theads_per_threadgroup(struct ggml_metal_pipeline_with_params pipeline) {
    return pipeline.pipeline->obj.maxTotalThreadsPerThreadgroup;
}

//
// MTLLibrary collection (one library per op-source, compiled separately)
//

// Single source of truth for the per-kind metal libraries. The order here
// defines the enum values and every per-kind table below, so adding a library
// is a one-line change here (plus adding its source to CMakeLists.txt).
//   X(suffix, name): name is both the kernels/<name>.metal basename and the
//   ggml_metallib_<name>_{start,end} embed-symbol stem.
#define GGML_METAL_LIBS \
    X(FA_AUX,          fa_aux)         \
    X(FA_F16,          fa_f16)         \
    X(FA_F32,          fa_f32)         \
    X(FA_Q4_0,         fa_q4_0)        \
    X(FA_Q4_1,         fa_q4_1)        \
    X(FA_Q5_0,         fa_q5_0)        \
    X(FA_Q5_1,         fa_q5_1)        \
    X(FA_Q8_0,         fa_q8_0)        \
    X(FA_VEC_F16,      fa_vec_f16)     \
    X(FA_VEC_F32,      fa_vec_f32)     \
    X(FA_VEC_Q4_0,     fa_vec_q4_0)    \
    X(FA_VEC_Q4_1,     fa_vec_q4_1)    \
    X(FA_VEC_Q5_0,     fa_vec_q5_0)    \
    X(FA_VEC_Q5_1,     fa_vec_q5_1)    \
    X(FA_VEC_Q8_0,     fa_vec_q8_0)    \
    X(MUL_MV,          mul_mv)         \
    X(MUL_MM,          mul_mm)         \
    X(QUANTIZE,        quantize)       \
    X(SOFTMAX,         softmax)        \
    X(NORM,            norm)           \
    X(UNARY,           unary)          \
    X(BINBCAST,        binbcast)       \
    X(REDUCE,          reduce)         \
    X(TRI,             tri)            \
    X(SSM,             ssm)            \
    X(WKV,             wkv)            \
    X(GATED_DELTA_NET, gated_delta_net)\
    X(SOLVE_TRI,       solve_tri)      \
    X(ROPE,            rope)           \
    X(CONV,            conv)           \
    X(UPSCALE,         upscale)        \
    X(ARGSORT,         argsort)        \
    X(POOL,            pool)           \
    X(MISC,            misc)

enum ggml_metal_lib_kind {
#define X(e, s) GGML_METAL_LIB_##e,
    GGML_METAL_LIBS
#undef X
    GGML_METAL_LIB_COUNT,
};

static const char * const k_lib_names[GGML_METAL_LIB_COUNT] = {
#define X(e, s) [GGML_METAL_LIB_##e] = #s,
    GGML_METAL_LIBS
#undef X
};

struct ggml_metal_library {
    // Per-kind compiled libraries. When single_library is true, the whole library
    // (e.g. a pre-compiled default.metallib or a from-source build) lives at
    // objs[0] and the remaining slots are nil.
    id<MTLLibrary> objs[GGML_METAL_LIB_COUNT];
    bool single_library; // true: combined library at objs[0]; false: per-kind libs in objs[*]

    // Routing table: kernel function name -> objs[] index, populated from each
    // compiled library's -[MTLLibrary functionNames]. The actual compiled
    // libraries are the single source of truth for which library owns a kernel,
    // so adding kernels later requires no manual routing maintenance.
    // nil in single_library mode (everything resolves to objs[0]).
    NSMutableDictionary<NSString *, NSNumber *> * fn_to_lib;

    // kernels from a second metallib, resolved ahead of the combined library
    NSSet<NSString *> * override_fns;

    ggml_metal_device_t dev;
    ggml_metal_pipelines_t pipelines; // cache of compiled pipelines

    NSLock * lock;
};

// Build the fn_to_lib routing table by querying each compiled library's public
// function names. Call once after all per-kind libraries have been compiled.
static void ggml_metal_library_build_index(ggml_metal_library_t lib) {
    @autoreleasepool {
        NSMutableDictionary<NSString *, NSNumber *> * index = [[NSMutableDictionary alloc] init];
        for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
            for (NSString * fname in [lib->objs[kind] functionNames]) {
                index[fname] = @(kind);
            }
        }
        lib->fn_to_lib = index;
    }
}

// note: defined below, after struct ggml_metal_device
static void ggml_metal_device_disable_tensor(ggml_metal_device_t dev);

// the tensor API headers are exposed to the shader compiler only at Metal language version 4.0
static void ggml_metal_compile_options_set_lang(MTLCompileOptions * options, bool has_tensor) {
    if (!has_tensor) {
        return;
    }

    options.languageVersion = (MTLLanguageVersion) MTLLanguageVersion4_0_GGML;
}

// Parse a `#include "name"` line. Returns the quoted name in *include_name on
// success. Whitespace-tolerant; ignores `#include <...>` (system headers).
static bool ggml_metal_library_parse_quoted_include(NSString * line, NSString ** include_name) {
    NSScanner * scanner = [NSScanner scannerWithString:line];
    scanner.charactersToBeSkipped = [NSCharacterSet whitespaceCharacterSet];

    if (![scanner scanString:@"#" intoString:NULL] ||
        ![scanner scanString:@"include" intoString:NULL] ||
        ![scanner scanString:@"\"" intoString:NULL]) {
        return false;
    }

    NSString * name = nil;
    if (![scanner scanUpToString:@"\"" intoString:&name]) {
        return false;
    }

    if (include_name) {
        *include_name = name;
    }
    return true;
}

// Recursively inline `#include "name"` directives. System includes (<...>),
// `#if/#else/#endif`, and other preprocessor lines are passed through to the
// Metal compiler unchanged. `#pragma once` is dropped since `seen` already
// guards against double-inclusion.
static bool ggml_metal_library_flatten_file(NSMutableString * dst, NSString * path,
                                            NSArray<NSString *> * search_paths,
                                            NSMutableSet<NSString *> * seen, NSError ** error) {
    NSString * key = [path stringByStandardizingPath];
    if ([seen containsObject:key]) {
        return true;
    }
    [seen addObject:key];

    NSString * src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:error];
    if (!src) {
        return false;
    }

    NSFileManager * fm = [NSFileManager defaultManager];
    for (NSString * line in [src componentsSeparatedByString:@"\n"]) {
        NSString * trimmed = [line stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
        if ([trimmed isEqualToString:@"#pragma once"]) {
            continue;
        }

        NSString * include_name = nil;
        if (ggml_metal_library_parse_quoted_include(line, &include_name)) {
            NSString * resolved = nil;
            for (NSString * dir in search_paths) {
                NSString * candidate = [dir stringByAppendingPathComponent:include_name];
                if ([fm isReadableFileAtPath:candidate]) {
                    resolved = candidate;
                    break;
                }
            }
            if (!resolved) {
                if (error) {
                    NSString * msg = [NSString stringWithFormat:@"could not resolve include \"%@\" from '%@'", include_name, path];
                    *error = [NSError errorWithDomain:@"ggml-metal-source-flatten" code:1
                                             userInfo:@{NSLocalizedDescriptionKey: msg}];
                }
                return false;
            }
            if (!ggml_metal_library_flatten_file(dst, resolved, search_paths, seen, error)) {
                return false;
            }
            continue;
        }

        [dst appendString:line];
        [dst appendString:@"\n"];
    }

    return true;
}

static NSString * ggml_metal_library_flatten_source(NSString * path_source, NSError ** error) {
    // Search paths cover both runtime layout (build/bin/kernels + build/bin)
    // and source-tree layout (ggml/src/ggml-metal/kernels + ggml/src/ggml-metal + ggml/src).
    NSString * path_kernels = [path_source stringByDeletingLastPathComponent];
    NSString * path_base    = [path_kernels stringByDeletingLastPathComponent];
    NSArray<NSString *> * search_paths = @[
        path_kernels,
        path_base,
        [path_base stringByDeletingLastPathComponent],
    ];

    NSMutableString * src = [[NSMutableString alloc] init];
    NSMutableSet<NSString *> * seen = [NSMutableSet set];

    if (!ggml_metal_library_flatten_file(src, path_source, search_paths, seen, error)) {
        [src release];
        return nil;
    }
    return src;
}

// Compile all per-kind libraries in parallel. `source_for_kind` returns the MSL
// source for a kind (the helper takes ownership and releases it), or nil with
// *err set on failure. On success the objs[] slots are populated and the routing
// index is built; on any failure every error is logged and false is returned
// (the caller is responsible for freeing `res`).
static bool ggml_metal_library_compile_all(
        ggml_metal_library_t res,
        id<MTLDevice> device,
        NSDictionary * prep,
        NSString * (^source_for_kind)(int kind, NSError ** err),
        const char * origin) {
    const int64_t t_start = ggml_time_us();

    int64_t  * t_per_lib   = calloc(GGML_METAL_LIB_COUNT, sizeof(int64_t));
    NSError ** err_per_lib = calloc(GGML_METAL_LIB_COUNT, sizeof(NSError *));
    __block atomic_bool any_failure = false;

    dispatch_group_t group = dispatch_group_create();
    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

    for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
        dispatch_group_async(group, queue, ^{

            const int64_t t0 = ggml_time_us();

            NSError * error = nil;

            NSString * src = source_for_kind(kind, &error);
            if (!src) {
                err_per_lib[kind] = [error retain];
                atomic_store(&any_failure, true);
                return;
            }

            id<MTLLibrary> lib = nil;

            @autoreleasepool {
                MTLCompileOptions * options = [MTLCompileOptions new];
                options.preprocessorMacros = prep;
                ggml_metal_compile_options_set_lang(options, ggml_metal_device_get_props(res->dev)->has_tensor);

                lib = [device newLibraryWithSource:src options:options error:&error];

                [options release];

                // retain the error before the autorelease pool drains it
                if (!lib) {
                    err_per_lib[kind] = [error retain];
                }
            }

            [src release];

            t_per_lib[kind] = ggml_time_us() - t0;

            if (!lib) {
                atomic_store(&any_failure, true);
                return;
            }

            res->objs[kind] = lib;
        });
    }
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    dispatch_release(group);

    const bool ok = !atomic_load(&any_failure);

    if (ok) {
        const int64_t t_total = ggml_time_us() - t_start;
        int64_t t_max = 0;
        for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
            GGML_LOG_DEBUG("%s: compiled '%s' library in %.3f sec\n",
                           __func__, k_lib_names[kind], t_per_lib[kind] / 1e6);
            if (t_per_lib[kind] > t_max) t_max = t_per_lib[kind];
        }
        GGML_LOG_INFO("%s: loaded %d libraries from %s in %.3f sec (max single = %.3f sec)\n",
                      __func__, GGML_METAL_LIB_COUNT, origin, t_total / 1e6, t_max / 1e6);

        ggml_metal_library_build_index(res);
    } else {
        for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
            if (err_per_lib[kind]) {
                GGML_LOG_ERROR("%s: failed to build '%s' library: %s\n", __func__,
                               k_lib_names[kind], [[err_per_lib[kind] description] UTF8String]);
                [err_per_lib[kind] release];
            }
        }
    }

    free(err_per_lib);
    free(t_per_lib);

    return ok;
}

// look for <name>.metallib as a bundle resource, then next to the running binary
static NSString * ggml_metal_find_metallib(NSBundle * bundle, NSString * name) {
    NSError * error = nil;

    NSString * path_lib = [bundle pathForResource:name ofType:@"metallib"];
    if (path_lib == nil) {
        // Try to find the resource in the directory where the current binary located.
        NSString * bin_cur = [[NSProcessInfo processInfo] arguments][0];
        NSString * bin_dir = [bin_cur stringByDeletingLastPathComponent];

        NSString * path_lib_default = [NSString pathWithComponents:@[bin_dir, [name stringByAppendingPathExtension:@"metallib"]]];
        if ([[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
            GGML_LOG_INFO("%s: found '%s'\n", __func__, [path_lib_default UTF8String]);

            NSDictionary * atts = [[NSFileManager defaultManager] attributesOfItemAtPath:path_lib_default error:&error];
            if (atts && atts[NSFileType] == NSFileTypeSymbolicLink) {
                // Optionally, if this is a symlink, try to resolve it.
                path_lib_default = [[NSFileManager defaultManager] destinationOfSymbolicLinkAtPath:path_lib_default error:&error];
                if (path_lib_default && [path_lib_default length] > 0 && ![[path_lib_default substringToIndex:1] isEqualToString:@"/"]) {
                    // It is a relative path, adding the binary directory as directory prefix.
                    path_lib_default = [NSString pathWithComponents:@[bin_dir, path_lib_default]];
                }
                if (!path_lib_default || ![[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
                    // Link to the resource could not be resolved.
                    path_lib_default = nil;
                } else {
                    GGML_LOG_INFO("%s: symlink resolved '%s'\n", __func__, [path_lib_default UTF8String]);
                }
            }
        } else {
            // The resource couldn't be found in the binary's directory.
            path_lib_default = nil;
        }

        path_lib = path_lib_default;
    }

    return path_lib;
}

ggml_metal_library_t ggml_metal_library_init(ggml_metal_device_t dev) {
    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    ggml_metal_library_t res = calloc(1, sizeof(struct ggml_metal_library));
    res->dev       = dev;
    res->pipelines = ggml_metal_pipelines_init();
    res->lock      = [NSLock new];

    // shared MTLCompileOptions preprocessor macros (matches the build-time defines)
    NSMutableDictionary * prep = [NSMutableDictionary dictionary];
    if (ggml_metal_device_get_props(dev)->has_bfloat) {
        [prep setObject:@"1" forKey:@"GGML_METAL_HAS_BF16"];
    }
    if (ggml_metal_device_get_props(dev)->has_tensor) {
        [prep setObject:@"1" forKey:@"GGML_METAL_HAS_TENSOR"];
    }
#if GGML_METAL_EMBED_LIBRARY
    [prep setObject:@"1" forKey:@"GGML_METAL_EMBED_LIBRARY"];
#endif

#if GGML_METAL_EMBED_LIBRARY
    GGML_LOG_INFO("%s: using embedded metal library\n", __func__);

    // start/end symbols emitted by CMake (see CMakeLists.txt), one pair per kind
#define X(e, s) extern const char ggml_metallib_##s##_start[]; extern const char ggml_metallib_##s##_end[];
    GGML_METAL_LIBS
#undef X

    static const char * const lib_start[GGML_METAL_LIB_COUNT] = {
#define X(e, s) [GGML_METAL_LIB_##e] = ggml_metallib_##s##_start,
    GGML_METAL_LIBS
#undef X
    };
    static const char * const lib_end[GGML_METAL_LIB_COUNT] = {
#define X(e, s) [GGML_METAL_LIB_##e] = ggml_metallib_##s##_end,
    GGML_METAL_LIBS
#undef X
    };

    const bool ok = ggml_metal_library_compile_all(res, device, prep,
        ^NSString * (int kind, NSError ** err) {
            (void) err;
            return [[NSString alloc] initWithBytes:lib_start[kind]
                                            length:(lib_end[kind] - lib_start[kind])
                                          encoding:NSUTF8StringEncoding];
        }, "embedded data");

    if (!ok) {
        ggml_metal_library_free(res);
        return NULL;
    }

    return res;
#else
#ifdef SWIFT_PACKAGE
    NSBundle * bundle = SWIFTPM_MODULE_BUNDLE;
#else
    NSBundle * bundle = [NSBundle bundleForClass:[GGMLMetalClass class]];
#endif

    const int64_t t_start = ggml_time_us();

    NSError * error = nil;
    NSString * path_lib = ggml_metal_find_metallib(bundle, @"default");

    if (path_lib != nil) {
        // pre-compiled library found: a single combined default.metallib
        NSURL * libURL = [NSURL fileURLWithPath:path_lib];
        GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_lib UTF8String]);

        res->objs[0]        = [device newLibraryWithURL:libURL error:&error];
        res->single_library = true;
        if (!res->objs[0]) {
            GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
            ggml_metal_library_free(res);
            return NULL;
        }

        // the tensor API kernels are built into a separate metallib
        if (ggml_metal_device_get_props(dev)->has_tensor) {
            NSString * path_mm = ggml_metal_find_metallib(bundle, @"ggml-tensor");

            id<MTLLibrary> lib_mm = nil;
            if (path_mm != nil) {
                lib_mm = [device newLibraryWithURL:[NSURL fileURLWithPath:path_mm] error:&error];
                if (!lib_mm && error) {
                    GGML_LOG_ERROR("%s: %s\n", __func__, [[error description] UTF8String]);
                }
            }

            if (lib_mm) {
                GGML_LOG_INFO("%s: loaded '%s'\n", __func__, [path_mm UTF8String]);

                res->objs[GGML_METAL_LIB_MUL_MM] = [lib_mm retain];
                res->override_fns                = [[NSSet setWithArray:[lib_mm functionNames]] retain];
            } else {
                GGML_LOG_INFO("%s: ggml-tensor.metallib not found - disabling the tensor API\n", __func__);

                ggml_metal_device_disable_tensor(dev);
            }
        }

        GGML_LOG_INFO("%s: loaded in %.3f sec\n", __func__, (ggml_time_us() - t_start) / 1e6);
        return res;
    }

    // no pre-compiled metallib: fall back to compiling each kernel source separately
    GGML_LOG_INFO("%s: default.metallib not found, loading kernel sources\n", __func__);

    NSString * path_resource = [[NSProcessInfo processInfo].environment objectForKey:@"GGML_METAL_PATH_RESOURCES"];
    if (path_resource) {
        GGML_LOG_INFO("%s: GGML_METAL_PATH_RESOURCES = %s\n", __func__, [path_resource UTF8String]);
    }

    // resolve each kind's source path up front (file lookup/logging stays on the calling thread)
    NSString ** path_per_kind = calloc(GGML_METAL_LIB_COUNT, sizeof(NSString *));
    for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
        NSString * rel = [NSString stringWithFormat:@"kernels/%s.metal", k_lib_names[kind]];

        NSString * path_source = nil;
        if (path_resource) {
            path_source = [path_resource stringByAppendingPathComponent:rel];
        } else {
            NSString * stem = [NSString stringWithFormat:@"kernels/%s", k_lib_names[kind]];
            path_source = [bundle pathForResource:stem ofType:@"metal"];
        }

        if (path_source == nil || ![[NSFileManager defaultManager] isReadableFileAtPath:path_source]) {
            GGML_LOG_WARN("%s: could not locate %s in bundle, falling back to cwd\n", __func__, [rel UTF8String]);
            path_source = rel;
        }

        GGML_LOG_DEBUG("%s: loading '%s'\n", __func__, [path_source UTF8String]);

        path_per_kind[kind] = [path_source retain];
    }

    const bool ok = ggml_metal_library_compile_all(res, device, prep,
        ^NSString * (int kind, NSError ** err) {
            return ggml_metal_library_flatten_source(path_per_kind[kind], err);
        }, "source");

    for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
        [path_per_kind[kind] release];
    }
    free(path_per_kind);

    if (!ok) {
        ggml_metal_library_free(res);
        return NULL;
    }

    return res;
#endif
}

ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, const char * source, bool verbose) {
    if (source == NULL) {
        GGML_LOG_ERROR("%s: source is NULL\n", __func__);
        return NULL;
    }

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    id<MTLLibrary> library = nil;
    NSError * error = nil;

    const int64_t t_start = ggml_time_us();

    NSString * src = [[NSString alloc] initWithBytes:source
                                              length:strlen(source)
                                            encoding:NSUTF8StringEncoding];
    if (!src) {
        GGML_LOG_ERROR("%s: failed to create NSString from source\n", __func__);
        return NULL;
    }

    @autoreleasepool {
        NSMutableDictionary * prep = [NSMutableDictionary dictionary];

        MTLCompileOptions * options = [MTLCompileOptions new];
        options.preprocessorMacros = prep;
        ggml_metal_compile_options_set_lang(options, ggml_metal_device_get_props(dev)->has_tensor);

        library = [device newLibraryWithSource:src options:options error:&error];
        if (error) {
            if (verbose) {
                GGML_LOG_ERROR("%s: error compiling source: %s\n", __func__, [[error description] UTF8String]);
            } else {
                GGML_LOG_ERROR("%s: error compiling source\n", __func__);
            }
            library = nil;
        }

        [options release];
    }

    [src release];

    if (!library) {
        if (verbose) {
            GGML_LOG_ERROR("%s: failed to create Metal library from source\n", __func__);
        }

        return NULL;
    }

    if (verbose) {
        GGML_LOG_INFO("%s: compiled in %.3f sec\n", __func__, (ggml_time_us() - t_start) / 1e6);
    }

    ggml_metal_library_t res = calloc(1, sizeof(struct ggml_metal_library));
    if (!res) {
        GGML_LOG_ERROR("%s: calloc failed\n", __func__);
        return NULL;
    }

    res->objs[0]        = library;
    res->single_library = true;
    res->dev            = dev;
    res->pipelines      = ggml_metal_pipelines_init();
    res->lock           = [NSLock new];

    return res;
}

void ggml_metal_library_free(ggml_metal_library_t lib) {
    if (!lib) {
        return;
    }

    for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
        if (lib->objs[kind]) {
            [lib->objs[kind] release];
        }
    }

    if (lib->fn_to_lib) {
        [lib->fn_to_lib release];
    }

    if (lib->override_fns) {
        [lib->override_fns release];
    }

    ggml_metal_pipelines_free(lib->pipelines);

    [lib->lock release];

    free(lib);
}

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib) {
    return lib->dev;
}

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline(ggml_metal_library_t lib, const char * name) {
    [lib->lock lock];

    struct ggml_metal_pipeline_with_params res = {
        /*.pipeline =*/ nil,
        /*.nsg      =*/ 0,
        /*.nr0      =*/ 0,
        /*.nr1      =*/ 0,
        /*.smem     =*/ 0,
        /*.c4       =*/ false,
        /*.cnt      =*/ false,
    };

    res.pipeline = ggml_metal_pipelines_get(lib->pipelines, name);

    [lib->lock unlock];

    return res;
}

struct ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, const char * base, const char * name, ggml_metal_cv_t cv) {
    struct ggml_metal_pipeline_with_params res = {
        /*.pipeline =*/ nil,
        /*.nsg      =*/ 0,
        /*.nr0      =*/ 0,
        /*.nr1      =*/ 0,
        /*.smem     =*/ 0,
        /*.c4       =*/ false,
        /*.cnt      =*/ false,
    };

    [lib->lock lock];

    res.pipeline = ggml_metal_pipelines_get(lib->pipelines, name);
    if (res.pipeline) {
        [lib->lock unlock];

        return res;
    }

    @autoreleasepool {
        NSError * error = nil;

        NSString * base_func = [NSString stringWithUTF8String:base];

        GGML_LOG_DEBUG("%s: compiling pipeline: base = '%s', name = '%s'\n", __func__, base, name);

        // route to the library that actually defines this kernel; fn_to_lib is
        // built from -[MTLLibrary functionNames] so it's always in sync
        int lib_idx = 0;
        if (lib->override_fns && [lib->override_fns containsObject:base_func]) {
            lib_idx = GGML_METAL_LIB_MUL_MM;
        } else if (!lib->single_library) {
            NSNumber * idx = lib->fn_to_lib[base_func];
            if (!idx) {
                [lib->lock unlock];

                GGML_LOG_ERROR("%s: kernel not found in any metal library: base = '%s', name = '%s'\n", __func__, base, name);

                return res;
            }
            lib_idx = [idx intValue];
        }

        id<MTLLibrary> mtl_lib = lib->objs[lib_idx];

        id<MTLFunction> mtl_function;
        if (!cv) {
            mtl_function = [mtl_lib newFunctionWithName:base_func];
        } else {
            mtl_function = [mtl_lib newFunctionWithName:base_func constantValues:cv->obj error:&error];
        }
        if (!mtl_function) {
            [lib->lock unlock];

            GGML_LOG_ERROR("%s: failed to compile pipeline: base = '%s', name = '%s'\n", __func__, base, name);
            if (error) {
                GGML_LOG_ERROR("%s: %s\n", __func__, [[error description] UTF8String]);
            }

            return res;
        }

        id<MTLDevice> device = ggml_metal_device_get_obj(lib->dev);
        id<MTLComputePipelineState> obj = [device newComputePipelineStateWithFunction:mtl_function error:&error];

        [mtl_function release];

        if (!obj) {
            [lib->lock unlock];

            GGML_LOG_ERROR("%s: failed to create pipeline state: base = '%s', name = '%s'\n", __func__, base, name);
            if (error) {
                GGML_LOG_ERROR("%s: %s\n", __func__, [[error description] UTF8String]);
            }

            return res;
        }

        GGML_LOG_DEBUG("%s: loaded %-40s %16p | th_max = %4d | th_width = %4d\n", __func__, name,
                (void *) obj,
                (int)    obj.maxTotalThreadsPerThreadgroup,
                (int)    obj.threadExecutionWidth);

        if (obj.maxTotalThreadsPerThreadgroup == 0 || obj.threadExecutionWidth == 0) {
            [obj release];

            [lib->lock unlock];

            GGML_LOG_ERROR("%s: incompatible pipeline %s\n", __func__, name);

            return res;
        }

        res.pipeline = ggml_metal_pipeline_init();
        res.pipeline->obj = obj;

        ggml_metal_pipelines_add(lib->pipelines, name, res.pipeline);
    }

    [lib->lock unlock];

    return res;
}

//
// MTLComputeCommandEncoder wrapper
//

// the Metal 4 code below is only reached after ggml_metal_device_supports_mtl4() checked macOS 26
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

struct ggml_metal_mtl4;

struct ggml_metal_encoder {
    id<MTLComputeCommandEncoder> obj;

    // Metal 4 (null for classic encoders): bindings go through the graph's argument table, small constants
    // through its constants buffer
    id<MTL4ComputeCommandEncoder> obj4;
    struct ggml_metal_mtl4      * m4;
};

static void ggml_metal_mtl4_set_bytes  (struct ggml_metal_mtl4 * m4, const void * data, size_t size, int idx);
static void ggml_metal_mtl4_set_address(struct ggml_metal_mtl4 * m4, uint64_t addr, int idx);
static bool ggml_metal_mtl4_serial     (struct ggml_metal_mtl4 * m4);

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent) {
    ggml_metal_encoder_t res = calloc(1, sizeof(struct ggml_metal_encoder));

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    if (concurrent) {
        res->obj = [cmd_buf computeCommandEncoderWithDispatchType: MTLDispatchTypeConcurrent];
    } else {
        res->obj = [cmd_buf computeCommandEncoder];
    }

    [res->obj retain];

    return res;
}

void ggml_metal_encoder_free(ggml_metal_encoder_t encoder) {
    if (encoder->obj4) {
        [encoder->obj4 release];
    } else {
        [encoder->obj release];
    }
    free(encoder);
}

void ggml_metal_encoder_debug_group_push(ggml_metal_encoder_t encoder, const char * name) {
    @autoreleasepool {
        NSString * str = [NSString stringWithCString:name encoding:NSUTF8StringEncoding];
        if (encoder->obj4) {
            [encoder->obj4 pushDebugGroup:str];
        } else {
            [encoder->obj pushDebugGroup:str];
        }
    }
}

void ggml_metal_encoder_debug_group_pop (ggml_metal_encoder_t encoder) {
    if (encoder->obj4) {
        [encoder->obj4 popDebugGroup];
    } else {
        [encoder->obj popDebugGroup];
    }
}

void ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, struct ggml_metal_pipeline_with_params pipeline) {
    if (!pipeline.pipeline) {
        GGML_ABORT("%s: nil Metal pipeline (missing kernel; see compile_pipeline log above)\n", __func__);
    }

    if (encoder->obj4) {
        [encoder->obj4 setComputePipelineState:pipeline.pipeline->obj];
    } else {
        [encoder->obj setComputePipelineState:pipeline.pipeline->obj];
    }
}

void ggml_metal_encoder_set_bytes(ggml_metal_encoder_t encoder, void * data, size_t size, int idx) {
    if (encoder->obj4) {
        ggml_metal_mtl4_set_bytes(encoder->m4, data, size, idx);
    } else {
        [encoder->obj setBytes:data length:size atIndex:idx];
    }
}

void ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, struct ggml_metal_buffer_id buffer, int idx) {
    if (encoder->obj4) {
        id<MTLBuffer> buf = (id<MTLBuffer>) buffer.metal;
        ggml_metal_mtl4_set_address(encoder->m4, buf ? buf.gpuAddress + buffer.offs : 0, idx);
    } else {
        [encoder->obj setBuffer:buffer.metal offset:buffer.offs atIndex:idx];
    }
}

void ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, int idx) {
    // ref: https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setthreadgroupmemorylength(_:index:)
    GGML_ASSERT(size % 16 == 0);

    if (encoder->obj4) {
        [encoder->obj4 setThreadgroupMemoryLength:size atIndex:idx];
    } else {
        [encoder->obj setThreadgroupMemoryLength:size atIndex:idx];
    }
}

// NInfer-M1 diagnostic (GGML_METAL_COUNT): number of dispatches encoded
int64_t g_ggml_metal_n_dispatch = 0;

void ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, int tg0, int tg1, int tg2, int tptg0, int tptg1, int tptg2) {
    __atomic_fetch_add(&g_ggml_metal_n_dispatch, 1, __ATOMIC_RELAXED);
    if (encoder->obj4) {
        [encoder->obj4 dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2) threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];
        // a Metal 4 encoder runs its dispatches concurrently; without the concurrency tracking of the op
        // encoder, keep the classic serial order with a barrier after each one
        if (ggml_metal_mtl4_serial(encoder->m4)) {
            ggml_metal_encoder_memory_barrier(encoder);
        }
    } else {
        [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2) threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];
    }
}

void ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder) {
    if (encoder->obj4) {
        [encoder->obj4 barrierAfterEncoderStages:MTLStageDispatch
                             beforeEncoderStages:MTLStageDispatch
                               visibilityOptions:MTL4VisibilityOptionDevice];
    } else {
        [encoder->obj memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
}

void ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder) {
    if (encoder->obj4) {
        [encoder->obj4 endEncoding];
    } else {
        [encoder->obj endEncoding];
    }
}

#pragma clang diagnostic pop

struct ggml_metal_device {
    id<MTLDevice> mtl_device;

    // a single global queue shared by all Metal backends
    // technically not needed for devices with unified memory, but enables discrete GPUs support
    // ref: https://github.com/ggml-org/llama.cpp/pull/15906
    id<MTLCommandQueue> mtl_queue;

    ggml_metal_rsets_t rsets;

    ggml_metal_library_t library;

    struct ggml_metal_device_props props;

    // shared fusion debugging context
    struct ggml_metal_fusion_info * finfo;

    // virtual address for GPU memory allocations
    atomic_uintptr_t addr_virt;

    // host ops: one shared event per device, values reserved in graph order
    id<MTLSharedEvent>       host_event;
    MTLSharedEventListener * host_listener;
    struct ggml_metal_host_spin * host_spin; // GGML_METAL_HOST_SPIN: the replies come from a spinning thread
    atomic_uint_fast64_t     host_value;

    // Metal 4 submission, created on first use (see ggml_metal_mtl4_begin); completion values on host_event
    NSLock * m4_lock;           // pools
    NSLock * m4_graph_lock;     // one graph encodes at a time: event values are reserved in queue order
    size_t   m4_cbuf_peak;      // most constants bytes used by a graph
    id       q4;                // id<MTL4CommandQueue>
    id       m4_opts;           // MTL4CommitOptions with the error feedback handler
    id       m4_alloc[GGML_METAL4_N_ALLOC];       // id<MTL4CommandAllocator>
    uint64_t m4_alloc_value[GGML_METAL4_N_ALLOC]; // free once host_event reaches it
    int      m4_alloc_next;
    id       m4_cbuf [GGML_METAL4_N_SLOT];        // id<MTLBuffer>, constants of one graph in flight
    id       m4_table[GGML_METAL4_N_SLOT];        // id<MTL4ArgumentTable>
    uint64_t m4_slot_value[GGML_METAL4_N_SLOT];
    int      m4_slot_next;
    id       m4_rset;           // every ggml buffer and the constants buffers, attached to q4 (under rsets->lock)
    atomic_bool m4_error;

    // Metal 4 command buffers not committed yet, in queue order (see ggml_metal_mtl4_pump)
    pthread_mutex_t m4_commit_mtx;
    struct ggml_metal_mtl4_item * m4_items; // ring of GGML_METAL4_N_ALLOC: every queued buffer holds an allocator
    int m4_items_head;
    int m4_items_count;
};

//
// MTLResidenceSet wrapper
//

struct ggml_metal_rsets {
    NSLock * lock;

    NSMutableArray * data;

    // number of seconds since the last graph computation
    // keep the residency sets wired for that amount of time to avoid being collected by the OS
    int keep_alive_s;
    int loops_per_s;
    int time_per_loop_ms;

    // background heartbeat thread to keep the residency sets alive
    atomic_bool d_stop;
    atomic_int  d_loop;

    dispatch_group_t d_group;
};

#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
static void ggml_metal_dummy_work(ggml_metal_device_t dev) {
    if (dev->mtl_queue == nil) {
        return;
    }

    @autoreleasepool {
        // perform a minimal dummy operation on the GPU
        id<MTLBuffer> buf = [dev->mtl_device newBufferWithLength:1 options:MTLResourceStorageModePrivate];
        id<MTLCommandBuffer> cmd_buf = [dev->mtl_queue commandBuffer];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:buf range:NSMakeRange(0, 1) value:0];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [buf release];
    }
}
#endif

ggml_metal_rsets_t ggml_metal_rsets_init(ggml_metal_device_t dev) {
    ggml_metal_rsets_t res = calloc(1, sizeof(struct ggml_metal_rsets));

    res->lock = [[NSLock alloc] init];
    res->data = [[NSMutableArray alloc] init];

    // by default keep the memory wired for 3 minutes
    res->keep_alive_s = 3*60;

    const char * GGML_METAL_RESIDENCY_KEEP_ALIVE_S = getenv("GGML_METAL_RESIDENCY_KEEP_ALIVE_S");
    if (GGML_METAL_RESIDENCY_KEEP_ALIVE_S) {
        res->keep_alive_s = atoi(GGML_METAL_RESIDENCY_KEEP_ALIVE_S);
    }

    if (res->keep_alive_s <= 0) {
        res->keep_alive_s = 3*60;
    }

    res->time_per_loop_ms = 5;
    res->loops_per_s = 1000/res->time_per_loop_ms;

    GGML_LOG_INFO("%s: creating a residency set collection (keep_alive = %d s)\n", __func__, res->keep_alive_s);

    atomic_store_explicit(&res->d_stop, false, memory_order_relaxed);
    atomic_store_explicit(&res->d_loop, res->loops_per_s*res->keep_alive_s, memory_order_relaxed);

    res->d_group = dispatch_group_create();

    // start a background thread that periodically requests residency for all the currently active sets in the collection
    // the requests stop after a certain amount of time (keep_alive_s) of inactivity
    dispatch_queue_t d_queue = dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
    dispatch_group_async(res->d_group, d_queue, ^{
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
        if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
              while (!atomic_load_explicit(&res->d_stop, memory_order_relaxed)) {
                  if (atomic_load_explicit(&res->d_loop, memory_order_relaxed) > 0) {
                      [res->lock lock];

                      for (int i = 0; i < (int) res->data.count; ++i) {
                          [res->data[i] requestResidency];
                      }

                      atomic_fetch_sub_explicit(&res->d_loop, 1, memory_order_relaxed);

                      [res->lock unlock];
                  }

                  usleep(res->time_per_loop_ms * 1000);
              }
        }
#endif
    });

#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        // workaround for residency set memory not being released if no GPU operation occurs
        // https://developer.apple.com/forums/thread/839089
        // https://github.com/ggml-org/llama.cpp/issues/25937
        ggml_metal_dummy_work(dev);
    }
#endif

    return res;
}

void ggml_metal_rsets_free(ggml_metal_rsets_t rsets) {
    if (rsets == NULL) {
        return;
    }

    // note: if you hit this assert, most likely you haven't deallocated all Metal resources before exiting
    GGML_ASSERT([rsets->data count] == 0);

    atomic_store_explicit(&rsets->d_stop, true, memory_order_relaxed);

    dispatch_group_wait(rsets->d_group, DISPATCH_TIME_FOREVER);
    dispatch_release(rsets->d_group);

    [rsets->data release];
    [rsets->lock release];

    free(rsets);
}

static const struct {
    const char *              name;
    const char *              token;
    enum ggml_metal_device_id id;
} k_metal_devices[] = {
#define DEV(name, id) { name, #id, id }
    DEV("M1",       GGML_METAL_DEVICE_M1),
    DEV("M1 Pro",   GGML_METAL_DEVICE_M1_PRO),
    DEV("M1 Max",   GGML_METAL_DEVICE_M1_MAX),
    DEV("M1 Ultra", GGML_METAL_DEVICE_M1_ULTRA),
    DEV("M2",       GGML_METAL_DEVICE_M2),
    DEV("M2 Pro",   GGML_METAL_DEVICE_M2_PRO),
    DEV("M2 Max",   GGML_METAL_DEVICE_M2_MAX),
    DEV("M2 Ultra", GGML_METAL_DEVICE_M2_ULTRA),
    DEV("M3",       GGML_METAL_DEVICE_M3),
    DEV("M3 Pro",   GGML_METAL_DEVICE_M3_PRO),
    DEV("M3 Max",   GGML_METAL_DEVICE_M3_MAX),
    DEV("M3 Ultra", GGML_METAL_DEVICE_M3_ULTRA),
    DEV("M4",       GGML_METAL_DEVICE_M4),
    DEV("M4 Pro",   GGML_METAL_DEVICE_M4_PRO),
    DEV("M4 Max",   GGML_METAL_DEVICE_M4_MAX),
    DEV("M5",       GGML_METAL_DEVICE_M5),
    DEV("M5 Pro",   GGML_METAL_DEVICE_M5_PRO),
    DEV("M5 Max",   GGML_METAL_DEVICE_M5_MAX),
    DEV("M5 Ultra", GGML_METAL_DEVICE_M5_ULTRA),
    DEV("A18 Pro",  GGML_METAL_DEVICE_A18_PRO),
#undef DEV
};

static enum ggml_metal_device_id ggml_metal_device_id_parse(const char * name) {
    if (!name) {
        return GGML_METAL_DEVICE_GENERIC;
    }

    static const char prefix[] = "Apple ";
    if (strncmp(name, prefix, sizeof(prefix) - 1) != 0) {
        return GGML_METAL_DEVICE_GENERIC;
    }
    const char * suffix = name + sizeof(prefix) - 1;

    for (size_t i = 0; i < sizeof(k_metal_devices)/sizeof(k_metal_devices[0]); ++i) {
        if (strcmp(suffix, k_metal_devices[i].name) == 0) {
            return k_metal_devices[i].id;
        }
    }
    return GGML_METAL_DEVICE_GENERIC;
}

const char * ggml_metal_device_id_token(enum ggml_metal_device_id id) {
    for (size_t i = 0; i < sizeof(k_metal_devices)/sizeof(k_metal_devices[0]); ++i) {
        if (k_metal_devices[i].id == id) {
            return k_metal_devices[i].token;
        }
    }
    return "GGML_METAL_DEVICE_GENERIC";
}

ggml_metal_device_t ggml_metal_device_init(int device, int n_devices) {
    ggml_metal_device_t dev = calloc(1, sizeof(struct ggml_metal_device));

    assert(dev != NULL);

    @autoreleasepool {
        if (dev->mtl_device == nil) {
            dev->mtl_device = MTLCreateSystemDefaultDevice();

            if (dev->mtl_device) {
                dev->mtl_queue = [dev->mtl_device newCommandQueue];
                if (dev->mtl_queue == nil) {
                    GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
                }

                dev->addr_virt = 0x000000400ULL;

                dev->host_event = [dev->mtl_device newSharedEvent];
                dev->m4_lock = [[NSLock alloc] init];
                dev->m4_graph_lock = [[NSLock alloc] init];
                pthread_mutex_init(&dev->m4_commit_mtx, NULL);
                dev->host_listener = [[MTLSharedEventListener alloc] initWithDispatchQueue:
                    dispatch_queue_create("ggml-metal-host-op", DISPATCH_QUEUE_SERIAL)];
                atomic_store(&dev->host_value, 0);

                dev->props.device = device;

                // the Metal backend uses the system default device as the single physical device;
                // additional (virtual) devices are emulated on top of it via GGML_METAL_DEVICES
                dev->props.device_phys = 0;
                dev->props.device_virt = device;

                dev->props.has_simdgroup_reduction  = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];
                dev->props.has_simdgroup_reduction |= [dev->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];

                dev->props.has_simdgroup_mm = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];
                dev->props.has_unified_memory = dev->mtl_device.hasUnifiedMemory;

                dev->props.has_bfloat  = [dev->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];
                dev->props.has_bfloat |= [dev->mtl_device supportsFamily:MTLGPUFamilyApple6];
                if (getenv("GGML_METAL_BF16_DISABLE") != NULL) {
                    dev->props.has_bfloat = false;
                }

                dev->props.has_tensor = [dev->mtl_device supportsFamily:MTLGPUFamilyMetal4_GGML];
                if (getenv("GGML_METAL_TENSOR_DISABLE") != NULL) {
                    dev->props.has_tensor = false;
                }

                // note: disable the tensor API by default for old chips because with the current implementation it is not useful
                // - M2 Ultra:   ~5% slower
                // - M4, M4 Max: no significant difference
                //
                // TODO: try to update the tensor API kernels to at least match the simdgroup performance
                if (getenv("GGML_METAL_TENSOR_ENABLE") == NULL &&
                    ![[dev->mtl_device name] containsString:@"M5"] &&
                    ![[dev->mtl_device name] containsString:@"M6"] &&
                    ![[dev->mtl_device name] containsString:@"A19"] &&
                    ![[dev->mtl_device name] containsString:@"A20"]) {
                    GGML_LOG_INFO("%s: tensor API disabled for pre-M5 and pre-A19 devices\n", __func__);
                    dev->props.has_tensor = false;
                }

                // double-check that the tensor API compiles
                if (dev->props.has_tensor) {
                    const char * src_tensor_f16 = "\n"
                        "#include <metal_stdlib> \n"
                        "#include <metal_tensor> \n"
                        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h> \n"
                        " \n"
                        "using namespace metal; \n"
                        "using namespace mpp::tensor_ops; \n"
                        " \n"
                        "kernel void dummy_kernel( \n"
                        "    tensor<device  half, dextents<int32_t, 2>> A [[buffer(0)]], \n"
                        "    tensor<device  half, dextents<int32_t, 2>> B [[buffer(1)]], \n"
                        "    device float * C [[buffer(2)]], \n"
                        "    uint2 tgid [[threadgroup_position_in_grid]]) \n"
                        "{ \n"
                        "    auto tA = A.slice(0, (int)tgid.y); \n"
                        "    auto tB = B.slice((int)tgid.x, 0); \n"
                        " \n"
                        "    matmul2d< \n"
                        "        matmul2d_descriptor(16, 16, dynamic_extent), \n"
                        "        execution_simdgroups<4>> mm; \n"
                        " \n"
                        "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>(); \n"
                        " \n"
                        "    auto sA = tA.slice(0, 0); \n"
                        "    auto sB = tB.slice(0, 0); \n"
                        "    mm.run(sB, sA, cT); \n"
                        " \n"
                        "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16)); \n"
                        " \n"
                        "    cT.store(tC); \n"
                        "}";

                    GGML_LOG_INFO("%s: testing tensor API for f16 support\n", __func__);
                    ggml_metal_library_t lib = ggml_metal_library_init_from_source(dev, src_tensor_f16, false);
                    if (lib == NULL) {
                        GGML_LOG_WARN("%s: - the tensor API is not supported in this environment - disabling\n", __func__);
                        dev->props.has_tensor = false;
                    } else {
                        struct ggml_metal_pipeline_with_params ppl = ggml_metal_library_compile_pipeline(lib, "dummy_kernel", "dummy_kernel", nil);
                        if (!ppl.pipeline) {
                            GGML_LOG_WARN("%s: - the tensor API is not supported in this environment - disabling\n", __func__);
                            dev->props.has_tensor = false;
                        }

                        ggml_metal_library_free(lib);
                    }
                }

                // try to compile a dummy kernel to determine if the tensor API is supported for bfloat
                if (dev->props.has_tensor && dev->props.has_bfloat) {
                    const char * src_tensor_bf16 = "\n"
                        "#include <metal_stdlib> \n"
                        "#include <metal_tensor> \n"
                        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h> \n"
                        " \n"
                        "using namespace metal; \n"
                        "using namespace mpp::tensor_ops; \n"
                        " \n"
                        "kernel void dummy_kernel( \n"
                        "    tensor<device bfloat, dextents<int32_t, 2>> A [[buffer(0)]], \n"
                        "    tensor<device bfloat, dextents<int32_t, 2>> B [[buffer(1)]], \n"
                        "    device float * C [[buffer(2)]], \n"
                        "    uint2 tgid [[threadgroup_position_in_grid]]) \n"
                        "{ \n"
                        "    auto tA = A.slice(0, (int)tgid.y); \n"
                        "    auto tB = B.slice((int)tgid.x, 0); \n"
                        " \n"
                        "    matmul2d< \n"
                        "        matmul2d_descriptor(16, 16, dynamic_extent), \n"
                        "        execution_simdgroups<4>> mm; \n"
                        " \n"
                        "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>(); \n"
                        " \n"
                        "    auto sA = tA.slice(0, 0); \n"
                        "    auto sB = tB.slice(0, 0); \n"
                        "    mm.run(sB, sA, cT); \n"
                        " \n"
                        "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16)); \n"
                        " \n"
                        "    cT.store(tC); \n"
                        "}";

                    GGML_LOG_INFO("%s: testing tensor API for bfloat support\n", __func__);
                    ggml_metal_library_t lib = ggml_metal_library_init_from_source(dev, src_tensor_bf16, false);
                    if (lib == NULL) {
                        GGML_LOG_WARN("%s: - the tensor API does not support bfloat - disabling bfloat support\n", __func__);
                        dev->props.has_bfloat = false;
                    } else {
                        struct ggml_metal_pipeline_with_params ppl = ggml_metal_library_compile_pipeline(lib, "dummy_kernel", "dummy_kernel", nil);
                        if (!ppl.pipeline) {
                            GGML_LOG_WARN("%s: - the tensor API does not support bfloat - disabling bfloat support\n", __func__);
                            dev->props.has_bfloat = false;
                        }

                        ggml_metal_library_free(lib);
                    }
                }

                dev->props.use_residency_sets = true;
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
                dev->props.use_residency_sets = getenv("GGML_METAL_NO_RESIDENCY") == nil;
#endif

                dev->props.use_shared_buffers = dev->props.has_unified_memory;
#if TARGET_OS_OSX && TARGET_CPU_X86_64
                // In case of eGPU, shared memory may be preferable.
                dev->props.use_shared_buffers |= [dev->mtl_device location] == MTLDeviceLocationExternal;
#endif
                if (getenv("GGML_METAL_SHARED_BUFFERS_DISABLE") != NULL) {
                    dev->props.use_shared_buffers = false;
                }
                if (getenv("GGML_METAL_SHARED_BUFFERS_ENABLE") != NULL) {
                    dev->props.use_shared_buffers = true;
                }

                dev->props.supports_gpu_family_apple7 = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];

                dev->props.device_id = ggml_metal_device_id_parse([[dev->mtl_device name] UTF8String]);

                dev->props.op_offload_min_batch_size  = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

                dev->props.max_buffer_size            = dev->mtl_device.maxBufferLength;
                dev->props.max_theadgroup_memory_size = dev->mtl_device.maxThreadgroupMemoryLength;
                if (@available(macOS 10.12, iOS 16.0, *)) {
                    dev->props.max_working_set_size   = dev->mtl_device.recommendedMaxWorkingSetSize;
                } else {
                    dev->props.max_working_set_size   = dev->mtl_device.maxBufferLength;
                }

                {
                    const char * val = getenv("GGML_METAL_FUSION_DEBUG");
                    dev->finfo = ggml_metal_fusion_info_init(
                            getenv("GGML_METAL_FUSION_DISABLE") == nil,
                            val ? atoi(val) : 0);
                }

                snprintf(dev->props.name, sizeof(dev->props.name), "%s%d", "MTL", device);
                const char * gpu_name = [[dev->mtl_device name] UTF8String];
                if (n_devices > 1) {
                    snprintf(dev->props.desc, sizeof(dev->props.desc), "%s (dev p%d/v%d)",
                             gpu_name, dev->props.device_phys, dev->props.device_virt);
                } else {
                    snprintf(dev->props.desc, sizeof(dev->props.desc), "%s", gpu_name);
                }

                dev->library = ggml_metal_library_init(dev);
                if (!dev->library) {
                    GGML_LOG_ERROR("%s: error: failed to create library\n", __func__);
                }

                if (dev->props.use_residency_sets) {
                    dev->rsets = ggml_metal_rsets_init(dev);
                } else {
                    dev->rsets = nil;
                }

                // print MTL GPU family:
                GGML_LOG_INFO("%s: GPU name:   %s (%s)\n", __func__, dev->props.name, dev->props.desc);

                // determine max supported GPU family
                // https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
                // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
                {
                    for (int i = MTLGPUFamilyApple1 + 20; i >= MTLGPUFamilyApple1; --i) {
                        if ([dev->mtl_device supportsFamily:i]) {
                            dev->props.gpu_family = i - (int) MTLGPUFamilyApple1 + 1;
                            GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyApple%d  (%d)\n", __func__, dev->props.gpu_family, i);
                            break;
                        }
                    }

#if TARGET_CPU_X86_64
                    for (int i = MTLGPUFamilyCommon1 + 5; i >= MTLGPUFamilyCommon1; --i) {
                        if ([dev->mtl_device supportsFamily:i]) {
                            GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyCommon%d (%d)\n", __func__, i - (int) MTLGPUFamilyCommon1 + 1, i);
                            break;
                        }
                    }
#endif

                    for (int i = MTLGPUFamilyMetal3_GGML + 5; i >= MTLGPUFamilyMetal3_GGML; --i) {
                        if ([dev->mtl_device supportsFamily:i]) {
                            GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyMetal%d  (%d)\n", __func__, i - (int) MTLGPUFamilyMetal3_GGML + 3, i);
                            break;
                        }
                    }
                }

                GGML_LOG_INFO("%s: simdgroup reduction   = %s\n", __func__, dev->props.has_simdgroup_reduction ? "true" : "false");
                GGML_LOG_INFO("%s: simdgroup matrix mul. = %s\n", __func__, dev->props.has_simdgroup_mm        ? "true" : "false");
                GGML_LOG_INFO("%s: has unified memory    = %s\n", __func__, dev->props.has_unified_memory      ? "true" : "false");
                GGML_LOG_INFO("%s: has bfloat            = %s\n", __func__, dev->props.has_bfloat              ? "true" : "false");
                GGML_LOG_INFO("%s: has tensor            = %s\n", __func__, dev->props.has_tensor              ? "true" : "false");
                GGML_LOG_INFO("%s: use residency sets    = %s\n", __func__, dev->props.use_residency_sets      ? "true" : "false");
                GGML_LOG_INFO("%s: use shared buffers    = %s\n", __func__, dev->props.use_shared_buffers      ? "true" : "false");

#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
                if (@available(macOS 10.12, iOS 16.0, *)) {
                    GGML_LOG_INFO("%s: recommendedMaxWorkingSetSize  = %8.2f MB\n", __func__, dev->props.max_working_set_size / 1e6);
                }
#endif
            }
        }
    }

    return dev;
}

//
// host-op replies from a spinning thread
//
// MTLSharedEventListener runs the reply block ~30 us after the GPU signals (54 us per round trip against 22 us when
// a CPU thread spins on signaledValue, tools/metal-sync-bench/gatebench). By default one thread
// takes the host ops in the order they were encoded, spins until the GPU reaches each one, runs it and signals back.
// It keeps one core busy while ops are pending and sleeps when the queue is empty; after 2 ms of spinning it falls
// back to a blocking wait, so a long GPU segment or a stalled queue does not burn the core.

#define GGML_METAL_HOST_SPIN_CAP 4096

struct ggml_metal_host_spin {
    ggml_metal_device_t dev;

    pthread_t       thread;
    pthread_mutex_t mtx;
    pthread_cond_t  cond;
    bool            stop;

    // pending ops in encode order, which is the order the GPU reaches them
    uint64_t             val [GGML_METAL_HOST_SPIN_CAP];
    struct ggml_tensor * snap[GGML_METAL_HOST_SPIN_CAP];
    int                  head;
    int                  count;
};

static void ggml_metal_host_op_run(struct ggml_tensor * node);

// Nothing the GPU signals can go past value v before the reply v + 1, unless it skipped the wait for it (a Metal 4 queue
// wait behind too much GPU work, see m4_max_ms in ggml-metal-context.m). The graph then ran on data the host op had not
// written yet: fail it loudly instead of returning wrong results. Returns false when the op should not run: from then on
// the ops only reply, so the failing graph drains without tripping over the inputs it corrupted.
static bool ggml_metal_host_op_check_order(ggml_metal_device_t dev, id<MTLSharedEvent> event, uint64_t v, const char * name);

// Metal 4 command buffers queued behind a host op are committed once the GPU reaches it (jit mode)
static void ggml_metal_mtl4_pump(ggml_metal_device_t dev);

static void * ggml_metal_host_spin_main(void * arg) {
    struct ggml_metal_host_spin * hs = (struct ggml_metal_host_spin *) arg;
    id<MTLSharedEvent> event = hs->dev->host_event;

    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    const uint64_t spin_ticks = (uint64_t) (2e6 * tb.denom / tb.numer); // 2 ms

    for (;;) {
        pthread_mutex_lock(&hs->mtx);
        while (hs->count == 0 && !hs->stop) {
            pthread_cond_wait(&hs->cond, &hs->mtx);
        }
        if (hs->count == 0) { // stop
            pthread_mutex_unlock(&hs->mtx);
            break;
        }
        const uint64_t       v    = hs->val [hs->head];
        struct ggml_tensor * snap = hs->snap[hs->head];
        pthread_mutex_unlock(&hs->mtx);

        bool ready = true;
        const uint64_t t0 = mach_absolute_time();
        while (event.signaledValue < v) {
            if (mach_absolute_time() - t0 < spin_ticks) {
                __asm__ __volatile__("yield");
                continue;
            }
            [event waitUntilSignaledValue:v timeoutMS:100];
            if (hs->stop && event.signaledValue < v) {
                ready = false; // shutting down and the GPU never got here
                break;
            }
        }

        if (ready) {
            ggml_metal_mtl4_pump(hs->dev); // the buffers after this op, in jit mode: committed while it runs
            if (ggml_metal_host_op_check_order(hs->dev, event, v, snap[0].name)) {
                ggml_metal_host_op_run(&snap[0]);
            }
            event.signaledValue = v + 1;
        }
        free(snap);

        pthread_mutex_lock(&hs->mtx);
        hs->head = (hs->head + 1) % GGML_METAL_HOST_SPIN_CAP;
        hs->count--;
        pthread_mutex_unlock(&hs->mtx);
    }

    return NULL;
}

// on by default; GGML_METAL_HOST_SPIN=0 goes back to the listener
static bool ggml_metal_host_spin_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * s = getenv("GGML_METAL_HOST_SPIN");
        enabled = !s || atoi(s) != 0;
    }
    return enabled != 0;
}

// queue the reply to host op value v; returns false when the spinning thread is not used
static bool ggml_metal_host_spin_submit(ggml_metal_device_t dev, uint64_t v, struct ggml_tensor * snap) {
    if (!ggml_metal_host_spin_enabled()) {
        return false;
    }

    if (dev->host_spin == NULL) {
        struct ggml_metal_host_spin * hs = calloc(1, sizeof(struct ggml_metal_host_spin));
        hs->dev = dev;
        pthread_mutex_init(&hs->mtx, NULL);
        pthread_cond_init(&hs->cond, NULL);
        if (pthread_create(&hs->thread, NULL, ggml_metal_host_spin_main, hs) != 0) {
            GGML_ABORT("%s: failed to start the host-op thread\n", __func__);
        }
        dev->host_spin = hs;
    }

    struct ggml_metal_host_spin * hs = dev->host_spin;

    pthread_mutex_lock(&hs->mtx);
    GGML_ASSERT(hs->count < GGML_METAL_HOST_SPIN_CAP && "too many host ops in flight");
    if (hs->count > 0) {
        const int last = (hs->head + hs->count - 1) % GGML_METAL_HOST_SPIN_CAP;
        GGML_ASSERT(hs->val[last] < v && "host ops must be queued in value order");
    }
    const int tail = (hs->head + hs->count) % GGML_METAL_HOST_SPIN_CAP;
    hs->val [tail] = v;
    hs->snap[tail] = snap;
    hs->count++;
    pthread_cond_signal(&hs->cond);
    pthread_mutex_unlock(&hs->mtx);

    return true;
}

static void ggml_metal_host_spin_free(ggml_metal_device_t dev) {
    struct ggml_metal_host_spin * hs = dev->host_spin;
    if (hs == NULL) {
        return;
    }
    pthread_mutex_lock(&hs->mtx);
    hs->stop = true;
    pthread_cond_signal(&hs->cond);
    pthread_mutex_unlock(&hs->mtx);
    pthread_join(hs->thread, NULL);
    pthread_mutex_destroy(&hs->mtx);
    pthread_cond_destroy(&hs->cond);
    free(hs);
    dev->host_spin = NULL;
}

void ggml_metal_device_free(ggml_metal_device_t dev) {
    assert(dev != NULL);

    @autoreleasepool {
        ggml_metal_fusion_info_free(dev->finfo);

        ggml_metal_rsets_free(dev->rsets);

        ggml_metal_library_free(dev->library);
        dev->library = NULL;

        if (dev->mtl_queue) {
            [dev->mtl_queue release];
            dev->mtl_queue = nil;
        }

        if (dev->q4) {
            for (int i = 0; i < GGML_METAL4_N_ALLOC; ++i) {
                [dev->m4_alloc[i] release];
            }
            for (int i = 0; i < GGML_METAL4_N_SLOT; ++i) {
                [dev->m4_cbuf[i] release];
                [dev->m4_table[i] release];
            }
            [dev->m4_rset release];
            [dev->m4_opts release];
            [dev->q4 release];
            dev->q4 = nil;
        }
        [dev->m4_lock release];
        [dev->m4_graph_lock release];
        pthread_mutex_destroy(&dev->m4_commit_mtx);
        free(dev->m4_items);

        ggml_metal_host_spin_free(dev);

        if (dev->host_listener) {
            [dev->host_listener release];
            dev->host_listener = nil;
        }

        if (dev->host_event) {
            [dev->host_event release];
            dev->host_event = nil;
        }

        if (dev->mtl_device) {
            [dev->mtl_device release];
            dev->mtl_device = nil;
        }
    }

    free(dev);
}

void * ggml_metal_device_get_obj(ggml_metal_device_t dev) {
    return dev->mtl_device;
}

void * ggml_metal_device_get_queue(ggml_metal_device_t dev) {
    return dev->mtl_queue;
}

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev) {
    return dev->library;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

// Metal 4 command buffers see only the allocations of residency sets attached to them or to their queue, at most 32
// sets per command buffer: one set on the queue holds the allocations of every buffer (call with rsets->lock held)
static void ggml_metal_mtl4_rset_update_locked(ggml_metal_device_t dev, id<MTLResidencySet> rset, bool add) {
    if (dev->m4_rset == nil) {
        return;
    }

    id<MTLResidencySet> all = dev->m4_rset;
    for (id<MTLAllocation> a in rset.allAllocations) {
        if (add) {
            [all addAllocation:a];
        } else {
            [all removeAllocation:a];
        }
    }
    [all commit];
}

#pragma clang diagnostic pop

void ggml_metal_device_rsets_add(ggml_metal_device_t dev, ggml_metal_rset_t rset) {
    if (rset == nil) {
        return;
    }

    GGML_ASSERT(dev->rsets);

    [dev->rsets->lock lock];

    [dev->rsets->data addObject:rset];
    ggml_metal_mtl4_rset_update_locked(dev, rset, true);

    [dev->rsets->lock unlock];
}

void ggml_metal_device_rsets_rm(ggml_metal_device_t dev, ggml_metal_rset_t rset) {
    if (rset == nil) {
        return;
    }

    GGML_ASSERT(dev->rsets);

    [dev->rsets->lock lock];

    ggml_metal_mtl4_rset_update_locked(dev, rset, false);
    [dev->rsets->data removeObject:rset];

    [dev->rsets->lock unlock];
}

void ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev) {
    if (dev->rsets == NULL) {
        return;
    }

    atomic_store_explicit(&dev->rsets->d_loop, dev->rsets->loops_per_s*dev->rsets->keep_alive_s, memory_order_relaxed);
}

struct ggml_metal_event {
    void * obj; // id<MTLSharedEvent>

    atomic_int value;
};

void ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf_raw) {
    id<MTLSharedEvent> event = (id<MTLSharedEvent>)ev->obj;

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    [cmd_buf encodeSignalEvent:event value:atomic_fetch_add_explicit(&ev->value, 1, memory_order_relaxed) + 1];
}

void ggml_metal_event_encode_wait(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf_raw) {
    id<MTLSharedEvent> event = (id<MTLSharedEvent>)ev->obj;

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    [cmd_buf encodeWaitForEvent:event value:atomic_load_explicit(&ev->value, memory_order_relaxed)];
}

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev) {
    id<MTLSharedEvent> event = [dev->mtl_device newSharedEvent];

    ggml_metal_event_t ev = calloc(1, sizeof(struct ggml_metal_event));

    ev->obj = (__bridge void *)event;
    ev->value = 0;

    return ev;
}

void ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLSharedEvent> event = ev->obj;
        [event release];

        free(ev);

        GGML_UNUSED(dev);
    }
}

void ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev) {
    id<MTLSharedEvent> event = ev->obj;
    const bool res = [event waitUntilSignaledValue:atomic_load_explicit(&ev->value, memory_order_relaxed) timeoutMS:60000];
    if (!res) {
        GGML_ABORT("%s: failed to wait for event\n", __func__);
    }

    GGML_UNUSED(dev);
}

//
// host ops
//

#define GGML_METAL_MAX_HOST_OPS 16

static void *      g_ggml_metal_host_ops[GGML_METAL_MAX_HOST_OPS];
static atomic_int  g_ggml_metal_n_host_ops = 0;
static NSLock    * g_ggml_metal_host_ops_lock = nil;

void ggml_metal_host_op_register(void * fun) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        g_ggml_metal_host_ops_lock = [[NSLock alloc] init];
    });

    [g_ggml_metal_host_ops_lock lock];
    const int n = atomic_load(&g_ggml_metal_n_host_ops);
    bool found = false;
    for (int i = 0; i < n; ++i) {
        found = found || g_ggml_metal_host_ops[i] == fun;
    }
    if (!found) {
        GGML_ASSERT(n < GGML_METAL_MAX_HOST_OPS);
        g_ggml_metal_host_ops[n] = fun;
        atomic_store(&g_ggml_metal_n_host_ops, n + 1);
    }
    [g_ggml_metal_host_ops_lock unlock];
}

bool ggml_metal_host_op_is_registered(const struct ggml_tensor * op) {
    if (op->op != GGML_OP_MAP_CUSTOM1 && op->op != GGML_OP_MAP_CUSTOM2 && op->op != GGML_OP_CUSTOM) {
        return false;
    }

    // the function pointer is the first field of all three op param structs
    void * fun;
    memcpy(&fun, op->op_params, sizeof(fun));

    const int n = atomic_load(&g_ggml_metal_n_host_ops);
    for (int i = 0; i < n; ++i) {
        if (g_ggml_metal_host_ops[i] == fun) {
            return true;
        }
    }
    return false;
}

uint64_t ggml_metal_device_host_op_reserve(ggml_metal_device_t dev, int n_ops) {
    return atomic_fetch_add(&dev->host_value, 2*(uint64_t) n_ops) + 1;
}

static void ggml_metal_host_op_run(struct ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_MAP_CUSTOM1:
            {
                struct ggml_map_custom1_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                p.fun(node, node->src[0], 0, 1, p.userdata);
            } break;
        case GGML_OP_MAP_CUSTOM2:
            {
                struct ggml_map_custom2_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                p.fun(node, node->src[0], node->src[1], 0, 1, p.userdata);
            } break;
        case GGML_OP_CUSTOM:
            {
                struct ggml_custom_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                p.fun(node, 0, 1, p.userdata);
            } break;
        default:
            GGML_ABORT("%s: not a host op\n", __func__);
    }
}

ggml_metal_cmd_buf_t ggml_metal_device_encode_host_op(ggml_metal_device_t dev, ggml_metal_cmd_buf_t cmd_buf_raw, struct ggml_tensor * node, bool release_cmd_buf) {
    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;
    id<MTLSharedEvent>   event   = dev->host_event;

    uint64_t v;
    memcpy(&v, (const char *) node->op_params + GGML_METAL_HOST_OP_VALUE_OFFS, sizeof(v));
    GGML_ASSERT(v > 0 && "host op value not assigned");

    // The op runs when the GPU gets here, possibly after llama has already reset the graph and built the
    // next one in the same memory (ubatches are pipelined). Snapshot the node and its sources now: the
    // buffers they point into stay valid until the backend is synchronized.
    struct ggml_tensor * snap = (struct ggml_tensor *) malloc(sizeof(struct ggml_tensor)*(1 + GGML_MAX_SRC));
    GGML_ASSERT(snap != NULL);
    memcpy(&snap[0], node, sizeof(struct ggml_tensor));
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        if (node->src[j]) {
            memcpy(&snap[1 + j], node->src[j], sizeof(struct ggml_tensor));
            snap[0].src[j] = &snap[1 + j];
        }
    }

    // registered before the command buffer is committed, so the signal cannot be missed; the reply
    // must always be sent, the GPU is blocked on it
    if (!ggml_metal_host_spin_submit(dev, v, snap)) {
        [event notifyListener:dev->host_listener atValue:v block:^(id<MTLSharedEvent> e, uint64_t value) {
            GGML_UNUSED(value);
            ggml_metal_mtl4_pump(dev);
            if (ggml_metal_host_op_check_order(dev, e, v, snap[0].name)) {
                ggml_metal_host_op_run(&snap[0]);
            }
            free(snap);
            e.signaledValue = v + 1;
        }];
    }

    [cmd_buf encodeSignalEvent:event value:v];
    if (release_cmd_buf) {
        // an intermediate command buffer of this graph: nothing else holds it
        [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            [cb release];
        }];
    }
    [cmd_buf commit];

    id<MTLCommandBuffer> next = [dev->mtl_queue commandBufferWithUnretainedReferences];
    [next retain];
    [next enqueue];
    [next encodeWaitForEvent:event value:v + 1];

    return (ggml_metal_cmd_buf_t) next;
}

//
// Metal 4 submission
//

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

struct ggml_metal_mtl4 {
    ggml_metal_device_t dev;

    bool serial; // barrier after every dispatch

    // per-graph slot: constants buffer and argument table, free again once host_event reaches the end value
    int        slot;
    uint8_t  * cbuf_base;
    uint64_t   cbuf_gpu;
    size_t     cbuf_off;

    id<MTL4ArgumentTable> table;

    // current command buffer and its encoder
    id<MTL4CommandBuffer> cb;
    int                   cb_alloc;

    struct ggml_metal_encoder * enc;

    // allocators of command buffers committed without a signal (ggml_metal_mtl4_flush): free once the next signaled
    // value is reached, which the queue order guarantees comes after them
    int pending_alloc[GGML_METAL4_N_ALLOC];
    int n_pending;

    // A queue-level waitForEvent stops holding back the work committed after it once it has sat behind a few seconds of
    // GPU work (M1 Max, macOS 26, tools/metal-sync-bench/mtl4waitcheck.m): with a whole prompt graph committed upfront
    // the GPU ran past host ops still loading experts. In jit mode the command buffers after a host op are committed
    // only once the GPU has reached it, so no wait is ever queued behind running work; short graphs (decode) commit
    // everything upfront.
    bool     jit;
    uint64_t wait_next; // queue wait for the next command buffer: the reply of the host op before it, 0 = none
    uint64_t gate;      // value the GPU signals at the last host op: in jit mode, the next buffers wait for it
};

// a command buffer waiting to be committed, with the queue wait before it and the signal after it
struct ggml_metal_mtl4_item {
    id<MTL4CommandBuffer> cb;
    uint64_t wait;   // 0 = none
    uint64_t signal; // 0 = none
    uint64_t after;  // committed once host_event reaches it, 0 = right away
};

bool ggml_metal_device_supports_mtl4(ggml_metal_device_t dev) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
        return dev->rsets != nil && dev->props.use_residency_sets && [dev->mtl_device supportsFamily:MTLGPUFamilyMetal4_GGML];
    }
    return false;
}

// commit the queued command buffers whose gate the GPU has reached, in order; called by the encoding thread after
// queueing, by the host-op thread when the GPU reaches a host op (before the op runs, so the commit overlaps it) and by
// the CPU waits. After an error everything goes, so the waits drain.
static void ggml_metal_mtl4_pump(ggml_metal_device_t dev) {
    if (dev->q4 == nil) {
        return;
    }
    pthread_mutex_lock(&dev->m4_commit_mtx);
    while (dev->m4_items_count > 0) {
        struct ggml_metal_mtl4_item * it = &dev->m4_items[dev->m4_items_head];
        if (it->after > 0 && dev->host_event.signaledValue < it->after && !atomic_load(&dev->m4_error)) {
            break;
        }
        id<MTL4CommandQueue> q = (id<MTL4CommandQueue>) dev->q4;
        if (it->wait > 0) {
            [q waitForEvent:dev->host_event value:it->wait];
        }
        id<MTL4CommandBuffer> cb = it->cb;
        [q commit:&cb count:1 options:dev->m4_opts];
        if (it->signal > 0) {
            [q signalEvent:dev->host_event value:it->signal];
        }
        [it->cb release];
        it->cb = nil;
        dev->m4_items_head = (dev->m4_items_head + 1) % GGML_METAL4_N_ALLOC;
        dev->m4_items_count--;
    }
    pthread_mutex_unlock(&dev->m4_commit_mtx);
}

static void ggml_metal_mtl4_enqueue(ggml_metal_device_t dev, id<MTL4CommandBuffer> cb, uint64_t wait, uint64_t signal, uint64_t after) {
    pthread_mutex_lock(&dev->m4_commit_mtx);
    // a queued buffer holds its allocator, and an allocator is reused only after its buffer completed
    GGML_ASSERT(dev->m4_items_count < GGML_METAL4_N_ALLOC);
    struct ggml_metal_mtl4_item * it = &dev->m4_items[(dev->m4_items_head + dev->m4_items_count) % GGML_METAL4_N_ALLOC];
    it->cb     = [cb retain];
    it->wait   = wait;
    it->signal = signal;
    it->after  = after;
    dev->m4_items_count++;
    pthread_mutex_unlock(&dev->m4_commit_mtx);

    ggml_metal_mtl4_pump(dev);
}

static void ggml_metal_mtl4_wait_value(ggml_metal_device_t dev, uint64_t value) {
    ggml_metal_mtl4_pump(dev);
    while (dev->host_event.signaledValue < value) {
        if ([dev->host_event waitUntilSignaledValue:value timeoutMS:1000]) {
            break;
        }
        // after a failed command buffer the signal may never come
        if (atomic_load(&dev->m4_error)) {
            break;
        }
    }
}

void ggml_metal_device_mtl4_wait(ggml_metal_device_t dev, uint64_t value) {
    ggml_metal_mtl4_wait_value(dev, value);
}

bool ggml_metal_device_mtl4_error(ggml_metal_device_t dev) {
    return atomic_load(&dev->m4_error);
}

static bool ggml_metal_host_op_check_order(ggml_metal_device_t dev, id<MTLSharedEvent> event, uint64_t v, const char * name) {
    const uint64_t seen = event.signaledValue;
    if (seen > v && !atomic_exchange(&dev->m4_error, true)) {
        GGML_LOG_ERROR("%s: the GPU ran past host op %s (value %llu, event already at %llu): the graph used data the op had "
                "not written, failing it\n", __func__, name, (unsigned long long) v, (unsigned long long) seen);
    }
    return !atomic_load(&dev->m4_error);
}

// first use: the queue, the pools and the residency set the queue sees (under m4_lock)
static void ggml_metal_mtl4_init_locked(ggml_metal_device_t dev) {
    if (dev->q4 != nil) {
        return;
    }

    dev->q4 = [dev->mtl_device newMTL4CommandQueue];
    GGML_ASSERT(dev->q4 != nil);

    MTL4CommitOptions * opts = [[MTL4CommitOptions alloc] init];
    [opts addFeedbackHandler:^(id<MTL4CommitFeedback> fb) {
        if (fb.error) {
            GGML_LOG_ERROR("ggml_metal_mtl4: command buffer failed: %s\n", [[fb.error description] UTF8String]);
            atomic_store(&dev->m4_error, true);
        }
    }];
    dev->m4_opts = opts;

    MTLResidencySetDescriptor * desc = [[MTLResidencySetDescriptor alloc] init];
    desc.label = @"ggml_metal_mtl4";
    NSError * error = nil;
    id<MTLResidencySet> rset = [dev->mtl_device newResidencySetWithDescriptor:desc error:&error];
    [desc release];
    GGML_ASSERT(rset != nil);

    MTL4ArgumentTableDescriptor * atd = [[MTL4ArgumentTableDescriptor alloc] init];
    atd.maxBufferBindCount = 31;

    for (int i = 0; i < GGML_METAL4_N_SLOT; ++i) {
        id<MTLBuffer> buf = [dev->mtl_device newBufferWithLength:GGML_METAL4_CBUF_SIZE options:MTLResourceStorageModeShared];
        GGML_ASSERT(buf != nil);
        [rset addAllocation:buf];
        dev->m4_cbuf[i] = buf;

        id<MTL4ArgumentTable> table = [dev->mtl_device newArgumentTableWithDescriptor:atd error:&error];
        GGML_ASSERT(table != nil);
        dev->m4_table[i] = table;

        dev->m4_slot_value[i] = 0;
    }
    [atd release];

    // from now on ggml_metal_device_rsets_add/rm keep the set in sync with the buffers
    [dev->rsets->lock lock];
    for (id<MTLResidencySet> buf_rset in dev->rsets->data) {
        for (id<MTLAllocation> a in buf_rset.allAllocations) {
            [rset addAllocation:a];
        }
    }
    [rset commit];
    dev->m4_rset = rset;
    const int n_alloc = (int) rset.allocationCount;
    [dev->rsets->lock unlock];

    [rset requestResidency];
    [(id<MTL4CommandQueue>) dev->q4 addResidencySet:rset];

    dev->m4_items = calloc(GGML_METAL4_N_ALLOC, sizeof(struct ggml_metal_mtl4_item));
    GGML_ASSERT(dev->m4_items != NULL);

    for (int i = 0; i < GGML_METAL4_N_ALLOC; ++i) {
        dev->m4_alloc[i] = [dev->mtl_device newCommandAllocator];
        GGML_ASSERT(dev->m4_alloc[i] != nil);
        dev->m4_alloc_value[i] = 0;
    }

    GGML_LOG_INFO("%s: Metal 4 submission ready (%d command allocators, %d graph slots of %u MiB constants, %d allocations resident)\n",
            __func__, GGML_METAL4_N_ALLOC, GGML_METAL4_N_SLOT, GGML_METAL4_CBUF_SIZE >> 20, n_alloc);
}

static void ggml_metal_mtl4_begin_cb(struct ggml_metal_mtl4 * m4) {
    ggml_metal_device_t dev = m4->dev;

    // an allocator holds the command memory until the GPU completes its command buffer
    [dev->m4_lock lock];
    const int a = dev->m4_alloc_next;
    dev->m4_alloc_next = (dev->m4_alloc_next + 1) % GGML_METAL4_N_ALLOC;
    const uint64_t busy_until = dev->m4_alloc_value[a];
    [dev->m4_lock unlock];

    ggml_metal_mtl4_wait_value(dev, busy_until);

    id<MTL4CommandAllocator> alloc = dev->m4_alloc[a];
    [alloc reset];

    m4->cb_alloc = a;
    m4->cb = [dev->mtl_device newCommandBuffer];
    [m4->cb beginCommandBufferWithAllocator:alloc];

    id<MTL4ComputeCommandEncoder> enc = [m4->cb computeCommandEncoder];
    [enc retain];
    [enc setArgumentTable:m4->table];

    // the previous command buffers in the queue (earlier segments and graphs) finish before this one runs
    [enc barrierAfterQueueStages:MTLStageDispatch | MTLStageBlit
                    beforeStages:MTLStageDispatch
               visibilityOptions:MTL4VisibilityOptionDevice];

    m4->enc = calloc(1, sizeof(struct ggml_metal_encoder));
    m4->enc->obj4 = enc;
    m4->enc->m4   = m4;
}

// end the current command buffer and queue it for the commit, with the pending wait before it and `signal` after it
static void ggml_metal_mtl4_submit_cb(struct ggml_metal_mtl4 * m4, uint64_t signal) {
    ggml_metal_device_t dev = m4->dev;

    [m4->enc->obj4 endEncoding];
    ggml_metal_encoder_free(m4->enc);
    m4->enc = NULL;

    [m4->cb endCommandBuffer];

    ggml_metal_mtl4_enqueue(dev, m4->cb, m4->wait_next, signal, m4->jit ? m4->gate : 0);
    m4->wait_next = 0;

    [m4->cb release];
    m4->cb = nil;
}

// end the current command buffer, commit it and signal `value` on the queue once it is done
static void ggml_metal_mtl4_commit_cb(struct ggml_metal_mtl4 * m4, uint64_t value) {
    ggml_metal_device_t dev = m4->dev;

    ggml_metal_mtl4_submit_cb(m4, value);

    [dev->m4_lock lock];
    dev->m4_alloc_value[m4->cb_alloc] = value;
    for (int i = 0; i < m4->n_pending; ++i) {
        dev->m4_alloc_value[m4->pending_alloc[i]] = value;
    }
    m4->n_pending = 0;
    [dev->m4_lock unlock];
}

ggml_metal_mtl4_t ggml_metal_mtl4_begin(ggml_metal_device_t dev, bool serial, bool jit) {
    struct ggml_metal_mtl4 * m4 = calloc(1, sizeof(struct ggml_metal_mtl4));
    m4->dev    = dev;
    m4->serial = serial;
    m4->jit    = jit;

    // released by ggml_metal_mtl4_end: the values of a graph are reserved and signaled in queue order
    [dev->m4_graph_lock lock];

    [dev->m4_lock lock];
    ggml_metal_mtl4_init_locked(dev);
    const int slot = dev->m4_slot_next;
    dev->m4_slot_next = (dev->m4_slot_next + 1) % GGML_METAL4_N_SLOT;
    const uint64_t busy_until = dev->m4_slot_value[slot];
    [dev->m4_lock unlock];

    // the slot's constants buffer is still read by a graph in flight
    ggml_metal_mtl4_wait_value(dev, busy_until);

    m4->slot      = slot;
    m4->cbuf_base = (uint8_t *) [(id<MTLBuffer>) dev->m4_cbuf[slot] contents];
    m4->cbuf_gpu  = [(id<MTLBuffer>) dev->m4_cbuf[slot] gpuAddress];
    m4->cbuf_off  = 0;
    m4->table     = dev->m4_table[slot];

    ggml_metal_mtl4_begin_cb(m4);

    return m4;
}

ggml_metal_encoder_t ggml_metal_mtl4_encoder(ggml_metal_mtl4_t m4) {
    return m4->enc;
}

ggml_metal_encoder_t ggml_metal_mtl4_host_op(ggml_metal_mtl4_t m4, struct ggml_tensor * node) {
    ggml_metal_device_t dev = m4->dev;
    id<MTLSharedEvent> event = dev->host_event;

    // encoding is single-threaded and in node order, so the values can be reserved right here
    const uint64_t v = atomic_fetch_add(&dev->host_value, 2) + 1;

    // snapshot the node and its sources: the op runs when the GPU gets here, possibly after llama has
    // reset the graph and built the next one in the same memory
    struct ggml_tensor * snap = (struct ggml_tensor *) malloc(sizeof(struct ggml_tensor)*(1 + GGML_MAX_SRC));
    GGML_ASSERT(snap != NULL);
    memcpy(&snap[0], node, sizeof(struct ggml_tensor));
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        if (node->src[j]) {
            memcpy(&snap[1 + j], node->src[j], sizeof(struct ggml_tensor));
            snap[0].src[j] = &snap[1 + j];
        }
    }

    // registered before the commit, so the signal cannot be missed; the reply must always be sent
    if (!ggml_metal_host_spin_submit(dev, v, snap)) {
        [event notifyListener:dev->host_listener atValue:v block:^(id<MTLSharedEvent> e, uint64_t value) {
            GGML_UNUSED(value);
            ggml_metal_mtl4_pump(dev);
            if (ggml_metal_host_op_check_order(dev, e, v, snap[0].name)) {
                ggml_metal_host_op_run(&snap[0]);
            }
            free(snap);
            e.signaledValue = v + 1;
        }];
    }

    ggml_metal_mtl4_commit_cb(m4, v);
    m4->wait_next = v + 1;
    m4->gate      = v;
    ggml_metal_mtl4_begin_cb(m4);

    return m4->enc;
}

ggml_metal_encoder_t ggml_metal_mtl4_flush(ggml_metal_mtl4_t m4) {
    ggml_metal_device_t dev = m4->dev;

    const int a = m4->cb_alloc;
    GGML_ASSERT(m4->n_pending < GGML_METAL4_N_ALLOC/2);

    // busy until the next signaled value (set by ggml_metal_mtl4_commit_cb)
    [dev->m4_lock lock];
    dev->m4_alloc_value[a] = UINT64_MAX;
    [dev->m4_lock unlock];
    m4->pending_alloc[m4->n_pending++] = a;

    ggml_metal_mtl4_submit_cb(m4, 0);
    ggml_metal_mtl4_begin_cb(m4);

    return m4->enc;
}

uint64_t ggml_metal_mtl4_end(ggml_metal_mtl4_t m4) {
    ggml_metal_device_t dev = m4->dev;

    const uint64_t v = atomic_fetch_add(&dev->host_value, 2) + 1;
    ggml_metal_mtl4_commit_cb(m4, v);

    [dev->m4_lock lock];
    dev->m4_slot_value[m4->slot] = v;
    if (m4->cbuf_off > dev->m4_cbuf_peak) {
        dev->m4_cbuf_peak = m4->cbuf_off;
        if (getenv("GGML_METAL4_DEBUG")) {
            GGML_LOG_INFO("%s: constants peak %.2f MiB per graph\n", __func__, m4->cbuf_off/1024.0/1024.0);
        }
    }
    [dev->m4_lock unlock];

    [dev->m4_graph_lock unlock];

    free(m4);

    return v;
}

static void ggml_metal_mtl4_set_bytes(struct ggml_metal_mtl4 * m4, const void * data, size_t size, int idx) {
    const size_t off = (m4->cbuf_off + 31) & ~(size_t) 31;
    if (off + size > GGML_METAL4_CBUF_SIZE) {
        GGML_ABORT("%s: constants buffer full (%u MiB per graph)\n", __func__, GGML_METAL4_CBUF_SIZE >> 20);
    }
    memcpy(m4->cbuf_base + off, data, size);
    m4->cbuf_off = off + size;

    [m4->table setAddress:m4->cbuf_gpu + off atIndex:idx];
}

static void ggml_metal_mtl4_set_address(struct ggml_metal_mtl4 * m4, uint64_t addr, int idx) {
    [m4->table setAddress:addr atIndex:idx];
}

static bool ggml_metal_mtl4_serial(struct ggml_metal_mtl4 * m4) {
    return m4->serial;
}

#pragma clang diagnostic pop

void ggml_metal_device_get_memory(ggml_metal_device_t dev, size_t * free, size_t * total) {
    if (@available(macOS 10.12, iOS 16.0, *)) {
        *total     = dev->mtl_device.recommendedMaxWorkingSetSize;
        size_t cur = dev->mtl_device.currentAllocatedSize;
        // it's possible to allocate more than `recommendedMaxWorkingSetSize`
        *free      = *total > cur ? *total - cur : 0;
    } else {
        *free = 0;
        *total = 0;
    }
}

static bool ggml_metal_supports_mul_mat_op(
        bool has_simdgroup_reduction,
        const struct ggml_tensor * op,
        bool src0_f16_has_mv,
        bool mm_path) {
    if (!has_simdgroup_reduction ||
        op->src[0]->type == GGML_TYPE_NVFP4 ||
        op->src[0]->type == GGML_TYPE_TQ1_0) {
        return false;
    }

    if (op->src[1]->type != GGML_TYPE_F16) {
        return true;
    }

    if (op->src[0]->type == GGML_TYPE_BF16) {
        return false;
    }

    if (src0_f16_has_mv && op->src[0]->type == GGML_TYPE_F16) {
        return true;
    }

    return mm_path;
}

bool ggml_metal_device_supports_op(ggml_metal_device_t dev, const struct ggml_tensor * op) {
    const bool has_simdgroup_mm        = dev->props.has_simdgroup_mm;
    const bool has_simdgroup_reduction = dev->props.has_simdgroup_reduction;
    const bool has_bfloat              = dev->props.has_bfloat;

    if (!has_bfloat) {
        if (op->type == GGML_TYPE_BF16) {
            return false;
        }

        for (size_t i = 0, n = 3; i < n; ++i) {
            if (op->src[i] != NULL && op->src[i]->type == GGML_TYPE_BF16) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_MAP_CUSTOM1:
        case GGML_OP_MAP_CUSTOM2:
        case GGML_OP_CUSTOM:
            return dev->props.use_shared_buffers && ggml_metal_host_op_is_registered(op);
        case GGML_OP_SCALE:
        case GGML_OP_FILL:
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            return ggml_is_contiguous_rows(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                case GGML_UNARY_OP_XIELU:
                    return ggml_is_contiguous_rows(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
                default:
                    return false;
            }
        case GGML_OP_SILU_BACK:
            return (op->src[0]->type == GGML_TYPE_F32) &&
                (op->src[1]->type == GGML_TYPE_F32) &&
                (op->type == GGML_TYPE_F32) &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_is_contiguous(op) &&
                ggml_are_same_shape(op->src[0], op->src[1]);
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    return ggml_is_contiguous_1(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
               default:
                    return false;
            }
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
            return true;
        case GGML_OP_CONCAT:
            {
                const enum ggml_type src0_type = op->src[0]->type;
                const enum ggml_type src1_type = op->src[1]->type;
                if (src0_type != src1_type || src0_type != op->type) {
                    return false;
                }
                switch (src0_type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_I8:
                    case GGML_TYPE_I16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_I64:
                        return true;
                    case GGML_TYPE_BF16:
                        return has_bfloat;
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_ADD_ID:
            return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) && (op->src[0]->type == op->src[1]->type);
        case GGML_OP_ACC:
            return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_REPEAT:
        case GGML_OP_CONV_TRANSPOSE_1D:
            return true;
        case GGML_OP_CONV_TRANSPOSE_2D:
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) &&
                (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_COL2IM_1D:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) &&
                op->type == op->src[0]->type &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op);
        case GGML_OP_CONV_3D:
            return ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_SUM:
            return has_simdgroup_reduction && ggml_is_contiguous(op->src[0]);
        case GGML_OP_TRI:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_SUM_ROWS:
        case GGML_OP_CUMSUM:
        case GGML_OP_MEAN:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_L2_NORM:
            return has_simdgroup_reduction && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_COUNT_EQUAL:
            return has_simdgroup_reduction &&
                op->src[0]->type == GGML_TYPE_I32 &&
                op->src[1]->type == GGML_TYPE_I32 &&
                op->type == GGML_TYPE_I64;
        case GGML_OP_ARGMAX:
            return has_simdgroup_reduction;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            return has_simdgroup_reduction && (ggml_is_contiguous_rows(op->src[0]));
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return true;
        case GGML_OP_IM2COL:
            return ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32 && (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);
        case GGML_OP_CONV_2D:
            return ggml_is_contiguous(op->src[0]) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
        case GGML_OP_CONV_2D_DW:
            return op->src[1]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
        case GGML_OP_UPSCALE:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_POOL_1D:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_POOL_2D:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_PAD:
            // TODO: add circular padding support for metal, see https://github.com/ggml-org/llama.cpp/pull/16985
            if (ggml_get_op_params_i32(op, 8) != 0) {
                return false;
            }

            return (ggml_get_op_params_i32(op, 0) == 0) && (ggml_get_op_params_i32(op, 2) == 0) &&
                   (ggml_get_op_params_i32(op, 4) == 0) && (ggml_get_op_params_i32(op, 6) == 0);
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_TIMESTEP_EMBEDDING:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_LEAKY_RELU:
            return op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16;
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_ARANGE:
            return true;
        case GGML_OP_ROLL:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_FLASH_ATTN_EXT:
            // for new head sizes, add checks here
            if (op->src[0]->ne[0] != 32 &&
                op->src[0]->ne[0] != 40 &&
                op->src[0]->ne[0] != 48 &&
                op->src[0]->ne[0] != 64 &&
                op->src[0]->ne[0] != 72 &&
                op->src[0]->ne[0] != 80 &&
                op->src[0]->ne[0] != 96 &&
                op->src[0]->ne[0] != 112 &&
                op->src[0]->ne[0] != 128 &&
                op->src[0]->ne[0] != 192 &&
                op->src[0]->ne[0] != 256 &&
                op->src[0]->ne[0] != 320 &&
                op->src[0]->ne[0] != 512 &&
                op->src[0]->ne[0] != 576) {
                return false;
            }
            if (op->src[1]->ne[0] == 72 && op->src[1]->ne[0] != op->src[2]->ne[0]) {
                return false;
            }
            if (op->src[1]->ne[0] < op->src[2]->ne[0]) {
                return false;
            }
            if (op->src[1]->type != op->src[2]->type) {
                return false;
            }
            switch (op->src[1]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                    break;
                case GGML_TYPE_BF16:
                    if (!has_bfloat) {
                        return false;
                    }
                    break;
                default:
                    return false;
            }
            return has_simdgroup_mm; // TODO: over-restricted for vec-kernels
        case GGML_OP_LIGHTNING_INDEXER:
            if (op->src[0]->ne[0] != OP_LIGHTNING_INDEXER_DK ||
                op->src[0]->ne[1] != OP_LIGHTNING_INDEXER_NH) {
                return false;
            }
            if (!has_simdgroup_mm ||
                op->src[0]->type != GGML_TYPE_F32 ||
                op->src[2]->type != GGML_TYPE_F32 ||
                op->src[3]->type != GGML_TYPE_F16 ||
                op->type         != GGML_TYPE_F32 ||
                !ggml_is_contiguous_rows(op->src[0]) ||
                !ggml_is_contiguous_rows(op->src[1]) ||
                !ggml_is_contiguous_rows(op->src[2]) ||
                !ggml_is_contiguous_rows(op->src[3])) {
                return false;
            }
            switch (op->src[1]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return true;
                case GGML_TYPE_BF16:
                    return has_bfloat;
                default:
                    return false;
            }
        case GGML_OP_DSV4_HC_COMB:
            return has_simdgroup_reduction &&
                op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->type         == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == 24 &&
                op->src[1]->ne[0] >= 3 &&
                op->src[2]->ne[0] == 24 &&
                ggml_is_contiguous_rows(op->src[0]) &&
                ggml_is_contiguous_rows(op->src[1]) &&
                ggml_is_contiguous_rows(op->src[2]);
        case GGML_OP_DSV4_HC_PRE:
            return has_simdgroup_reduction &&
                op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type         == GGML_TYPE_F32 &&
                ggml_is_contiguous_rows(op->src[0]) &&
                ggml_is_contiguous_rows(op->src[1]);
        case GGML_OP_DSV4_HC_POST:
            return has_simdgroup_reduction &&
                op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                (op->src[3] == NULL || op->src[3]->type == GGML_TYPE_F32) &&
                op->type         == GGML_TYPE_F32 &&
                op->src[1]->ne[1] == 4 &&
                op->src[2]->ne[0] == 4 &&
                (op->src[3] == NULL || (op->src[3]->ne[0] == 4 && op->src[3]->ne[1] == 4)) &&
                ggml_is_contiguous_rows(op->src[0]) &&
                ggml_is_contiguous_rows(op->src[1]) &&
                ggml_is_contiguous_rows(op->src[2]) &&
                (op->src[3] == NULL || ggml_is_contiguous_rows(op->src[3]));
        case GGML_OP_SSM_SCAN:
            return has_simdgroup_reduction;
        case GGML_OP_SSM_CONV:
            return has_simdgroup_reduction;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            return has_simdgroup_reduction && op->src[2]->ne[0] % 32 == 0;
        case GGML_OP_SOLVE_TRI:
            return has_simdgroup_reduction && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_MUL_MAT:
            // the FWHT kernels read an F16 source directly; every other F16 src1 path
            // still goes through ggml_metal_supports_mul_mat_op
            if (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F16 &&
                ggml_metal_op_mul_mat_use_fwht(op, dev->props.max_theadgroup_memory_size)) {
                return has_simdgroup_reduction;
            }
            return ggml_metal_supports_mul_mat_op(
                    has_simdgroup_reduction, op, true,
                    ggml_metal_op_mul_mat_use_mm(op, has_simdgroup_mm));
        case GGML_OP_MUL_MAT_ID:
            return ggml_metal_supports_mul_mat_op(
                    has_simdgroup_reduction, op, false,
                    ggml_metal_op_mul_mat_id_use_mm(op, has_simdgroup_mm));
        case GGML_OP_SET:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        switch (op->type) {
                           case GGML_TYPE_F32:
                           case GGML_TYPE_F16:
                           case GGML_TYPE_BF16:
                           case GGML_TYPE_Q8_0:
                           case GGML_TYPE_Q1_0:
                           case GGML_TYPE_Q2_0:
                           case GGML_TYPE_Q4_0:
                           case GGML_TYPE_Q4_1:
                           case GGML_TYPE_Q5_0:
                           case GGML_TYPE_Q5_1:
                           case GGML_TYPE_IQ4_NL:
                           case GGML_TYPE_TQ2_0:
                           case GGML_TYPE_I32:
                                return true;
                           default:
                                return false;
                        }
                    case GGML_TYPE_F16:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_BF16:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_BF16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_TQ2_0:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_I32:
                        return op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32;
                    default:
                        return false;
                };
            }
        case GGML_OP_GET_ROWS:
            return op->src[0]->type != GGML_TYPE_NVFP4 &&
                   op->src[0]->type != GGML_TYPE_TQ1_0;
        case GGML_OP_SET_ROWS:
            {
                if (op->src[0]->type == GGML_TYPE_F16) {
                    return op->type == GGML_TYPE_F16;
                }

                if (op->src[0]->type != GGML_TYPE_F32) {
                    return false;
                }

                switch (op->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_TQ2_0:
                        return true;
                    default:
                        return false;
                };
            }
        case GGML_OP_DIAG:
            return true;
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            return has_simdgroup_reduction;
        default:
            return false;
    }
}

const struct ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev) {
    return &dev->props;
}

static void ggml_metal_device_disable_tensor(ggml_metal_device_t dev) {
    dev->props.has_tensor = false;
}

struct ggml_metal_fusion_info * ggml_metal_device_get_fusion_info(ggml_metal_device_t dev) {
    return dev->finfo;
}

//
// device buffers
//

// max memory buffers that can be mapped to the device
#define GGML_METAL_MAX_BUFFERS 64

struct ggml_metal_buffer_wrapper {
    void   * data;
    size_t   size;

    id<MTLBuffer> metal;
};

struct ggml_metal_buffer {
    void * all_data;
    size_t all_size;

    // if false, the Metal buffer data is allocated in private GPU memory and is not shared with the host
    bool is_shared;
    bool owned;

    // multiple buffers are used only to avoid the maximum buffer size limitation when using mmap
    int n_buffers;
    struct ggml_metal_buffer_wrapper buffers[GGML_METAL_MAX_BUFFERS];

    bool use_residency_sets;

    // optional MTLResidencySet
    // note: cannot use explicitly "id<MTLResidencySet>" here because it is not available on certain OSes
    id rset;

    // pointers to global device
    ggml_metal_device_t dev;
};

static void ggml_metal_log_allocated_size(id<MTLDevice> device, size_t size_aligned) {
#ifndef GGML_METAL_NDEBUG
#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
    if (@available(macOS 10.12, iOS 16.0, *)) {
        GGML_LOG_DEBUG("%s: allocated buffer, size = %8.2f MiB, (%8.2f / %8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0,
                device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);

        if (device.currentAllocatedSize > device.recommendedMaxWorkingSetSize) {
            GGML_LOG_WARN("%s: warning: current allocated size is greater than the recommended max working set size\n", __func__);
        }
    } else {
        GGML_LOG_INFO("%s: allocated buffer, size = %8.2f MiB, (%8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0);
    }
#endif
#endif
    GGML_UNUSED(device);
    GGML_UNUSED(size_aligned);
}

// rset init
static bool ggml_metal_buffer_rset_init(ggml_metal_buffer_t buf) {
    buf->rset = nil;

    if (!buf->use_residency_sets) {
        return true;
    }

#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        MTLResidencySetDescriptor * desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"ggml_metal";
        desc.initialCapacity = buf->n_buffers;

        NSError * error;
        buf->rset = [buf->dev->mtl_device newResidencySetWithDescriptor:desc error:&error];
        if (error) {
            GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
            [desc release];
            return false;
        }

        [desc release];

        for (int i = 0; i < buf->n_buffers; i++) {
            [buf->rset addAllocation:buf->buffers[i].metal];
        }

        [buf->rset commit];
        [buf->rset requestResidency];

        return true;
    }
#endif

    return true;
}

// rset free
static void ggml_metal_buffer_rset_free(ggml_metal_buffer_t buf) {
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        if (buf->rset) {
            [buf->rset endResidency];
            [buf->rset removeAllAllocations];
            [buf->rset commit];
            [buf->rset release];
        }
    }
#else
    GGML_UNUSED(buf);
#endif
}

static void * ggml_metal_host_malloc(size_t n) {
    void * data = NULL;

#if TARGET_OS_OSX
    kern_return_t err = vm_allocate((vm_map_t) mach_task_self(), (void *) &data, n, VM_FLAGS_ANYWHERE);
    if (err != KERN_SUCCESS) {
        GGML_LOG_ERROR("%s: error: vm_allocate failed\n", __func__);
        return NULL;
    }
#else
    const int result = posix_memalign((void **) &data, sysconf(_SC_PAGESIZE), n);
    if (result != 0) {
        GGML_LOG_ERROR("%s: error: posix_memalign failed\n", __func__);
        return NULL;
    }
#endif

    return data;
}

// NInfer-M1: upper limit for the Metal buffers of the process. On 2026-10-06 an M1 Max 64 GB panicked with 51 GiB of
// Metal buffers (54.3 GiB wired, 123 MiB free): the GPU driver got kIOReturnNoMemory and crashed the kernel. Allocations
// beyond the limit fail cleanly instead. GGML_METAL_MAX_ALLOC_GB sets it (0 disables); default physical memory - 14 GiB.
static size_t ggml_metal_alloc_limit(void) {
    static size_t limit = 0;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        const char * s = getenv("GGML_METAL_MAX_ALLOC_GB");
        if (s) {
            limit = (size_t) (atof(s) * 1024.0 * 1024.0 * 1024.0);
        } else {
            const uint64_t ram = [[NSProcessInfo processInfo] physicalMemory];
            limit = ram > (28ull << 30) ? (size_t) (ram - (14ull << 30)) : (size_t) (ram / 2);
        }
    });
    return limit;
}

static bool ggml_metal_alloc_allowed(id<MTLDevice> device, size_t size, const char * what) {
    const size_t limit = ggml_metal_alloc_limit();
    const size_t cur   = device.currentAllocatedSize;

    if (limit > 0 && cur + size > limit) {
        GGML_LOG_ERROR("%s: refusing to allocate %.2f MiB (%s): %.2f GiB of Metal buffers already allocated, limit %.2f GiB "
                "(GGML_METAL_MAX_ALLOC_GB). Above it the GPU driver can run out of wired memory and panic the kernel; "
                "reduce the context, the ubatch or the expert cache.\n",
                __func__, size/1024.0/1024.0, what, cur/1024.0/1024.0/1024.0, limit/1024.0/1024.0/1024.0);
        return false;
    }
    if (size >= (512ull << 20)) {
        GGML_LOG_INFO("%s: %s of %.2f MiB, Metal buffers %.2f GiB of %.2f GiB allowed\n",
                __func__, what, size/1024.0/1024.0, (cur + size)/1024.0/1024.0/1024.0, limit/1024.0/1024.0/1024.0);
    }
    return true;
}

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t dev, size_t size, bool shared) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    res->dev = dev;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    if (!ggml_metal_alloc_allowed(dev->mtl_device, size_aligned, "buffer")) {
        free(res);
        return NULL;
    }

    shared = shared && props_dev->use_shared_buffers;

    // allocate shared buffer if the device supports it and it is required by the buffer type
    if (shared) {
        res->all_data = ggml_metal_host_malloc(size_aligned);
        res->is_shared = true;
    } else {
        // use virtual address
        res->all_data = (void *) atomic_fetch_add_explicit(&dev->addr_virt, size_aligned, memory_order_relaxed);
        res->is_shared = false;
    }
    res->all_size = size_aligned;

    res->owned = true;

    res->n_buffers = 1;

    if (res->all_data != NULL) {
        res->buffers[0].size  = size;
        res->buffers[0].metal = nil;

        if (size_aligned > 0) {
            if (props_dev->use_shared_buffers && shared) {
                res->buffers[0].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:res->all_data
                                                                  length:size_aligned
                                                                 options:MTLResourceStorageModeShared
                                                             deallocator:nil];
            } else {
                res->buffers[0].metal = [res->dev->mtl_device newBufferWithLength:size_aligned options:MTLResourceStorageModePrivate];
            }
        }

        res->buffers[0].data = res->all_data;
    }

    if (size_aligned > 0 && (res->all_data == NULL || res->buffers[0].metal == nil)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
        free(res);
        return NULL;
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(res);
        return NULL;
    }

    ggml_metal_device_rsets_add(dev, res->rset);

    //ggml_metal_log_allocated_size(device, size_aligned);

    return res;
}

ggml_metal_buffer_t ggml_metal_buffer_map(ggml_metal_device_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    res->dev = dev;

    res->all_data = ptr;
    res->all_size = size;

    res->is_shared = true;
    res->owned = false;

    res->n_buffers = 0;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    // page-align the data ptr
    {
        const uintptr_t offs = (uintptr_t) ptr % size_page;
        ptr  = (void *) ((char *) ptr - offs);
        size += offs;
    }

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    // mapped file pages count too: residency wires them
    if (!ggml_metal_alloc_allowed(dev->mtl_device, size_aligned, "mapped buffer")) {
        free(res);
        return NULL;
    }

    // the buffer fits into the max buffer size allowed by the device
    if (size_aligned <= props_dev->max_buffer_size) {
        res->buffers[res->n_buffers].data  = ptr;
        res->buffers[res->n_buffers].size  = size;
        res->buffers[res->n_buffers].metal = nil;

        if (size_aligned > 0) {
            res->buffers[res->n_buffers].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:ptr length:size_aligned options:MTLResourceStorageModeShared deallocator:nil];

            if (res->buffers[res->n_buffers].metal == nil) {
                GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
                free(res);
                return NULL;
            }
        }

        ggml_metal_log_allocated_size(res->dev->mtl_device, size_aligned);

        ++res->n_buffers;
    } else {
        // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
        // one of the views
        const size_t size_ovlp = ((max_tensor_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
        const size_t size_step = props_dev->max_buffer_size - size_ovlp;
        const size_t size_view = props_dev->max_buffer_size;

        for (size_t i = 0; i < size; i += size_step) {
            const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

            res->buffers[res->n_buffers].data  = (void *) ((uint8_t *) ptr + i);
            res->buffers[res->n_buffers].size  = size_step_aligned;
            res->buffers[res->n_buffers].metal = nil;

            if (size_step_aligned > 0) {
                res->buffers[res->n_buffers].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:(void *) ((uint8_t *) ptr + i) length:size_step_aligned options:MTLResourceStorageModeShared deallocator:nil];

                if (res->buffers[res->n_buffers].metal == nil) {
                    GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_step_aligned / 1024.0 / 1024.0);
                    free(res);
                    return NULL;
                }
            }

            ggml_metal_log_allocated_size(res->dev->mtl_device, size_step_aligned);

            if (i + size_step < size) {
                GGML_LOG_INFO("\n");
            }

            ++res->n_buffers;
        }
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(res);
        return NULL;
    }

    ggml_metal_device_rsets_add(dev, res->rset);

    return res;
}

void ggml_metal_buffer_free(ggml_metal_buffer_t buf) {
    @autoreleasepool {
        ggml_metal_device_rsets_rm(buf->dev, buf->rset);

        for (int i = 0; i < buf->n_buffers; i++) {
            [buf->buffers[i].metal release];
        }

        ggml_metal_buffer_rset_free(buf);
    }

    if (buf->is_shared && buf->owned) {
#if TARGET_OS_OSX
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)buf->all_data, buf->all_size);
#else
        free(buf->all_data);
#endif
    }

    free(buf);
}

void * ggml_metal_buffer_get_base(ggml_metal_buffer_t buf) {
    return buf->all_data;
}

bool ggml_metal_buffer_is_shared(ggml_metal_buffer_t buf) {
    return buf->is_shared;
}

void ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    if (buf->is_shared) {
        memset((char *) tensor->data + offset, value, size);
        return;
    }

    @autoreleasepool {
        // dst
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
        bid_dst.offs += offset;

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:bid_dst.metal
                          range:NSMakeRange(bid_dst.offs, bid_dst.offs + size)
                          value:value];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_buffer_set_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (buf->is_shared) {
        memcpy((char *) tensor->data + offset, data, size);
        return;
    }

    @autoreleasepool {
        // src
        void * data_ptr = (void *)(uintptr_t) data; // "const cast" the src data
        id<MTLBuffer> buf_src = [buf->dev->mtl_device newBufferWithBytesNoCopy:data_ptr
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        GGML_ASSERT(buf_src);

        // dst
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
        bid_dst.offs += offset;

        // note: for experimentation purposes, here we use a semaphore to wait for the copy to complete
        //       this is alternative to waitUntilCompleted, which should be faster, but don't seem to make much difference
        dispatch_semaphore_t completion_semaphore = dispatch_semaphore_create(0);

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:buf_src
                       sourceOffset:0
                           toBuffer:bid_dst.metal
                  destinationOffset:bid_dst.offs
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                             // TODO: can check for errors here
            GGML_UNUSED(cb);

            dispatch_semaphore_signal(completion_semaphore);
        }];

        [cmd_buf commit];

        dispatch_semaphore_wait(completion_semaphore, DISPATCH_TIME_FOREVER);
        dispatch_release(completion_semaphore);

        //[cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_buffer_get_tensor(ggml_metal_buffer_t buf, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (buf->is_shared) {
        memcpy(data, (const char *) tensor->data + offset, size);
        return;
    }

    @autoreleasepool {
        // src
        struct ggml_metal_buffer_id bid_src = ggml_metal_buffer_get_id(buf, tensor);
        bid_src.offs += offset;

        // dst
        id<MTLBuffer> buf_dst = [buf->dev->mtl_device newBufferWithBytesNoCopy:data
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        GGML_ASSERT(buf_dst);

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:bid_src.metal
                       sourceOffset:bid_src.offs
                           toBuffer:buf_dst
                  destinationOffset:0
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

bool ggml_metal_buffer_cpy_tensor(ggml_metal_buffer_t buf_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_metal_buffer_t buf_src = (ggml_metal_buffer_t)src->buffer->context;

    const size_t size = ggml_nbytes(src);

    // if both buffers are shared, we can use memcpy directly
    if (buf_dst->is_shared && buf_src->is_shared) {
        memcpy(dst->data, src->data, size);
        return true;
    }

    // for private buffers, we need to use Metal blit commands
    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_buffer_get_id(buf_src, src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf_dst, dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        id<MTLCommandBuffer> cmd_buf = [buf_dst->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:bid_src.metal
                       sourceOffset:bid_src.offs
                           toBuffer:bid_dst.metal
                  destinationOffset:bid_dst.offs
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }

    return true;
}

void ggml_metal_buffer_clear(ggml_metal_buffer_t buf, uint8_t value) {
    if (buf->is_shared) {
        memset(buf->all_data, value, buf->all_size);
        return;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:buf->buffers[0].metal
                          range:NSMakeRange(0, buf->buffers[0].size)
                          value:value];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

struct ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t buf, const struct ggml_tensor * t) {
    struct ggml_metal_buffer_id res = { nil, 0 };

    const int64_t tsize = ggml_nbytes(t);

    // find the view that contains the tensor fully
    for (int i = 0; i < buf->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) buf->buffers[i].data;

        //GGML_LOG_INFO("ioffs = %10ld, tsize = %10ld, sum = %10ld, buf->buffers[%d].size = %10ld\n", ioffs, tsize, ioffs + tsize, i, buf->buffers[i].size);
        if (ioffs >= 0 && ioffs + tsize <= (int64_t) buf->buffers[i].size) {
            res.metal = buf->buffers[i].metal;
            res.offs  = (size_t) ioffs;

            //GGML_LOG_INFO("%s: tensor '%16s', offs = %8ld\n", __func__, t->name, *offs);

            return res;
        }
    }

    GGML_LOG_ERROR("%s: error: tensor '%s' buffer is nil\n", __func__, t->name);

    return res;
}
