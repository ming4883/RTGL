// NrcCommon.glsl - Neural Radiance Cache shared definitions
// NrcCommon.glsl - Neural Radiance Cache shared definitions
// (adapted from VkNRC by AdamYuan, MIT license)
// RTGL integration: NV-only path via VK_NV_cooperative_matrix.

#define NRC_SET 12

// ---- minimal RTGL scene access (getTriangle + global uniforms) ----
// ---- prerequisites expected by RTGL shader headers ----

#ifndef UINT8_MAX
#define UINT8_MAX 255
#endif
#ifndef UINT16_MAX
#define UINT16_MAX 65535
#endif


// ---- enable only the RTGL accessors the NRC passes need ----
#define DESC_SET_GLOBAL_UNIFORM 2
#define DESC_SET_VERTEX_DATA    3
#define DESC_SET_TEXTURES       4
#define DESC_SET_FRAMEBUFFERS   1

#include "ShaderCommonGLSLFunc.h"
#include "Surface.inl"
#include "BRDF.h"

// bindings inside NRC_SET
#define NRC_BINDING_EVAL_COUNT 0
#define NRC_BINDING_EVAL_RECORDS 1
#define NRC_BINDING_TRAIN_COUNTS 2
#define NRC_BINDING_TRAIN_RECORDS 3
#define NRC_BINDING_WEIGHTS 4
#define NRC_BINDING_GRADIENTS 5
#define NRC_BINDING_OPTIMIZER_ENTRIES 6
#define NRC_BINDING_OPTIMIZER_STATE 7
#define NRC_BINDING_DISPATCH_CMD 8
#define NRC_BINDING_BIAS_FACTOR_R 9
#define NRC_BINDING_FACTOR_GB 10
#define NRC_BINDING_PARAMS 11

// NN architecture (must match VkNRC defaults)
#define NRC_HIDDEN_LAYERS 5
#define NRC_TRAIN_BATCH_COUNT 4
#define NRC_TRAIN_BATCH_SIZE 16384
#define NRC_WEIGHT_COUNT ( 64 * 64 * NRC_HIDDEN_LAYERS + 64 * 3 ) // 20608
#define NRC_ADAM_BETA1 0.9
#define NRC_ADAM_BETA2 0.999
#define NRC_EMA_ALPHA 0.99
#define NRC_LEARNING_RATE 0.002

// probability of a path being used as a training sample
#define NRC_TRAIN_PROBABILITY 0.25

// record types
#define NRC_RECORD_TYPE_SCREEN 0
#define NRC_RECORD_TYPE_TRAIN 1

// ShPayload-compatible packed NRC input
struct PackedNRCInputRTGL
{
    uint instIdAndIndex;
    uint geomAndPrimIndex;
    uint barycentric_2x16U;
    uint scattered_dir_2x16U;
};

struct NRCEvalRecordRTGL
{
    uint               dst;
    PackedNRCInputRTGL packed_input;
};

struct NRCTrainRecordRTGL
{
    float              bias_r, bias_g, bias_b;
    float              factor_r, factor_g, factor_b;
    PackedNRCInputRTGL packed_input;
};

// frame params uploaded per-frame by NRCCache (uniform buffer, NRC_BINDING_PARAMS)
struct NRCFrameParams
{
    vec4  uPosNormalizeScale;   // xyz = 1/scale, w = unused
    vec4  uPosNormalizeCenter;  // xyz = center
    uint  uFrameId;
    uint  uWidth;
    uint  uHeight;
    float uTrainProbability;
};

struct NRCDispatchCmd
{
    uint x;
    uint y;
    uint z;
};