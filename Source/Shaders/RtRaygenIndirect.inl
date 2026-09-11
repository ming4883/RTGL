// Copyright (c) 2021-2022 Sultim Tsyrendashiev
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// For indirect sample
//#extension GL_EXT_shader_16bit_storage : require
//#extension GL_EXT_shader_explicit_arithmetic_types : require

layout (constant_id = 0) const uint maxAlbedoLayerCount = 0;
layout (constant_id = 1) const uint lightmapLayerIndex = 3;
#define MATERIAL_MAX_ALBEDO_LAYERS maxAlbedoLayerCount
#define MATERIAL_LIGHTMAP_LAYER_INDEX lightmapLayerIndex

#define FORCE_EVALBRDF_GGX_LOOSE


#ifndef RT_FORCE_COMPUTE

#define DESC_SET_TLAS 0
#define DESC_SET_FRAMEBUFFERS 1
#define DESC_SET_GLOBAL_UNIFORM 2
#define DESC_SET_VERTEX_DATA 3
#define DESC_SET_TEXTURES 4
#define DESC_SET_RANDOM 5
#define DESC_SET_LIGHT_SOURCES 6
#define DESC_SET_CUBEMAPS 7
#define DESC_SET_RENDER_CUBEMAP 8
#define DESC_SET_RESTIR_INDIRECT 10
#define LIGHT_SAMPLE_METHOD (LIGHT_SAMPLE_METHOD_INDIR)
#include "RaygenCommon.h"
#include "ReservoirIndirect.h"

#ifdef NRC_ENABLED
#extension GL_EXT_scalar_block_layout : require
#define NRC_SET 12
#define NRC_BINDING_EVAL_COUNT 0
#define NRC_BINDING_EVAL_RECORDS 1
#define NRC_BINDING_TRAIN_COUNTS 2
#define NRC_BINDING_TRAIN_RECORDS 3
#define NRC_HIDDEN_LAYERS 5
#define NRC_TRAIN_BATCH_COUNT 4
#define NRC_TRAIN_BATCH_SIZE 16384
#define NRC_TRAIN_PROBABILITY 0.25
#define NRC_RECORD_TYPE_SCREEN 0
#define NRC_RECORD_TYPE_TRAIN 1

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

// max training records the train records buffer can hold (see NRCCache.cpp)
#define NRC_TRAIN_RECORD_CAPACITY ( NRC_TRAIN_BATCH_COUNT * NRC_TRAIN_BATCH_SIZE )

// must match NRCFrameParamsGpu in NRCCache.h; 60 bytes
struct NRCFrameParams
{
    vec4  uPosNormalizeScale;   // xyz = 1/scale, w = unused
    vec4  uPosNormalizeCenter;  // xyz = center
    uint  uFrameId;
    uint  uWidth;
    uint  uHeight;
    float uTrainProbability;
    uint  uTrainBatchSize;
    float uLearningRate;
    float uEmaAlpha;
};

layout( set = NRC_SET, binding = NRC_BINDING_EVAL_COUNT, scalar ) buffer NRCEvalCountBuffer
{
    uint uEvalCount;
};

layout( set = NRC_SET, binding = NRC_BINDING_EVAL_RECORDS, scalar ) buffer NRCEvalRecordsBuffer
{
    NRCEvalRecordRTGL uEvalRecords[];
};

layout( set = NRC_SET, binding = NRC_BINDING_TRAIN_COUNTS, scalar ) buffer NRCTrainCountsBuffer
{
    uint uTotalTrainCount;
};

layout( set = NRC_SET, binding = NRC_BINDING_TRAIN_RECORDS, scalar ) buffer NRCTrainRecordsBuffer
{
    NRCTrainRecordRTGL uTrainRecords[];
};

#define NRC_BINDING_PARAMS 11
layout( set = NRC_SET, binding = NRC_BINDING_PARAMS, scalar ) uniform NRCParamsUBO
{
    NRCFrameParams uNRCParams;
};
#endif // NRC_ENABLED

#else // RT_FORCE_COMPUTE

    #define DESC_SET_FRAMEBUFFERS 1
    #define DESC_SET_GLOBAL_UNIFORM 2
    #define DESC_SET_RESTIR_INDIRECT 10
    #define LIGHT_SAMPLE_METHOD (LIGHT_SAMPLE_METHOD_INDIR)
    #include "ShaderCommonGLSLFunc.h"
    #include "Surface.inl"
    #include "Light.h"
    #include "ReservoirIndirect.h"
    layout( local_size_x = COMPUTE_INDIRECT_FINAL_GROUP_SIZE_X,
            local_size_y = COMPUTE_INDIRECT_FINAL_GROUP_SIZE_Y,
            local_size_z = 1 ) in;

#endif // !RT_FORCE_COMPUTE


#define OPTIMIZE_MEM 1

bool useDiffuse( const float roughness )
{
    return roughness >= FAKE_ROUGH_SPECULAR_THRESHOLD;
}

float getDiffuseWeight( float roughness )
{
    return smoothstep( MIN_GGX_ROUGHNESS,
                       FAKE_ROUGH_SPECULAR_THRESHOLD + FAKE_ROUGH_SPECULAR_LENGTH,
                       roughness );
}

// v -- direction to viewer
// n -- surface normal
vec3 getSpecularBounce(const uint seed, uint bounceIndex,
                       const vec3 n, const float roughness, const vec3 surfSpecularColor,
                       const vec3 v, 
                       out float oneOverSourcePdf)
{
#if OPTIMIZE_MEM
    const vec2 u = rnd8_4( seed, RANDOM_SALT_SPEC_BOUNCE( bounceIndex ) ).xy;
#else
    const vec2 u = rndBlueNoise16_2( seed, RANDOM_SALT_SPEC_BOUNCE( bounceIndex ) );
#endif
    return sampleSmithGGX( n, v, roughness, u[ 0 ], u[ 1 ], oneOverSourcePdf );
}

// n -- surface normal
vec3 getDiffuseBounce(const uint seed, uint bounceIndex, const vec3 n, out float oneOverSourcePdf)
{
#if OPTIMIZE_MEM
    const vec2 u = rnd8_4( seed, RANDOM_SALT_DIFF_BOUNCE( bounceIndex ) ).xy;
#else
    const vec2 u = rndBlueNoise16_2( seed, RANDOM_SALT_DIFF_BOUNCE( bounceIndex ) );
#endif
    return sampleLambertian( n, u[ 0 ], u[ 1 ], oneOverSourcePdf );
}

#ifndef RT_FORCE_COMPUTE

#define FIRST_BOUNCE_MIP_BIAS 0
#define SECOND_BOUNCE_MIP_BIAS 32

// Max number of diffuse bounces traced by the indirect ray generation:
//   1 - first bounce only, no diffuse tail after the first hit
//   2 - the historical behavior: exactly one extra diffuse bounce
//   3..8 - multi-bounce diffuse path tracing
// The upper bound of 8 comes from the random salt layout: diffuse bounces
// 1..8 use RANDOM_SALT_DIFF_BOUNCE salts 9..16 (see Random.h).
// Compile-time constant ON PURPOSE: it must never move into the global
// uniform table, or the spv/DLL uniform layout mismatch that once
// black-screened the renderer comes back. Override before including.
#ifndef INDIRECT_DIFFUSE_MAX_BOUNCES
#define INDIRECT_DIFFUSE_MAX_BOUNCES 2
#endif

// payload of the last traced non-sky bounce; used to rebuild NRC inputs for
// the cache records, and read by the tail bounce tracing to snapshot the
// entry state (unconditional - classic builds use it too)
ShPayload g_lastBouncePayload;

Surface traceBounce(const vec3 originPosition, float originRoughness, uint originInstCustomIndex,
                    const vec3 bounceDir, float bounceMipBias, out vec3 out_emission)
{
    const ShPayload p = traceIndirectRay(originInstCustomIndex, originPosition, bounceDir); 

    if (!doesPayloadContainHitInfo(p))
    {
        Surface s;
        s.isSky = true;
        return s;
    }

    g_lastBouncePayload = p;

    return hitInfoToSurface_Indirect(
        getHitInfoBounce(p, originPosition, originRoughness, bounceMipBias, out_emission), 
        bounceDir);
}

// Trace the diffuse tail (bounces 2..INDIRECT_DIFFUSE_MAX_BOUNCES) from the
// first-bounce hit surface, accumulating (emission + direct illumination) *
// albedo * 1/pdf through the chain - the same convention the single second
// bounce used. With INDIRECT_DIFFUSE_MAX_BOUNCES == 2 this is mathematically
// identical to the previous single-bounce version, RNG stream included.
// out_secondPayload/out_secondDir capture the state right after the bounce-2
// trace (hit payload + sampled direction) for NRC cache records.
vec3 processDiffuseTailBounces( const uint    seed,
                               const Surface surf,
                               out ShPayload out_secondPayload,
                               out vec3      out_secondDir )
{
    vec3 tail       = vec3( 0.0 );
    vec3 throughput = vec3( 1.0 );
    out_secondDir     = vec3( 0.0 );
    // entry init, silences undefined-variable warnings; overwritten after the bounce-2 trace
    out_secondPayload = g_lastBouncePayload;
    Surface curSurf = surf;

    for( int b = 2; b <= INDIRECT_DIFFUSE_MAX_BOUNCES; b++ )
    {
        float oneOverPdf;
        const vec3 bounceDir = getDiffuseBounce( seed, uint( b ), curSurf.normal, oneOverPdf );

        vec3 emis;
        const Surface hitSurf = traceBounce( curSurf.position + curSurf.normal * 0.01,
                                             curSurf.roughness,
                                             curSurf.instCustomIndex,
                                             bounceDir,
                                             SECOND_BOUNCE_MIP_BIAS,
                                             emis );

        if( b == 2 )
        {
            // snapshot for NRC records: payload of the second-bounce trace
            // (stays the first-bounce payload if that trace hit the sky,
            //  matching the previous single-bounce behavior)
            out_secondPayload = g_lastBouncePayload;
            out_secondDir     = bounceDir;
        }

        emis *= globalUniform.emissionMapBoost;
        throughput *= oneOverPdf;

        if( hitSurf.isSky )
        {
            tail += throughput * getSky( bounceDir );
            break;
        }

        // calculate direct illumination in a hit position; shadow rays for
        // deeper vertices are gated by maxBounceShadowsLights
        const vec3 diffuse = processDirectIllumination( seed, hitSurf, b );

        tail       += throughput * ( emis + diffuse ) * hitSurf.albedo;
        throughput *= hitSurf.albedo;
        curSurf     = hitSurf;

        // deterministic russian roulette - consumes no random numbers, so the
        // RNG stream stays identical to the historical 2-bounce path when
        // INDIRECT_DIFFUSE_MAX_BOUNCES == 2
        if( getLuminance( throughput ) < 0.01 )
        {
            break;
        }
    }

    // last-resort guard: a single NaN in the tail would be stored to the
    // unfiltered indirect image and locked by temporal accumulation forever
    if( any( isnan( tail ) ) || any( isinf( tail ) ) )
    {
        tail = vec3( 0.0 );
    }
    return tail;
}

#ifdef NRC_ENABLED
uint NrcPackDstScreen( const ivec2 pix )
{
    uint x = uint( pix.x ) & 0x7FFF;
    uint y = uint( pix.y ) & 0x7FFF;
    return ( y << 16 ) | ( x << 1 ) | NRC_RECORD_TYPE_SCREEN;
}

uint NrcPackDstTrain( const ivec2 pix )
{
    uint x = uint( pix.x ) & 0x7FFF;
    uint y = uint( pix.y ) & 0x7FFF;
    return ( y << 16 ) | ( x << 1 ) | NRC_RECORD_TYPE_TRAIN;
}
#endif // NRC_ENABLED

SampleIndirect processIndirect( const uint seed, const Surface surf, out float oneOverSourcePdf )
{
    vec3 bounceDir;

    if( useDiffuse( surf.roughness ) )
    {
        bounceDir = getDiffuseBounce( seed, 1, surf.normal, oneOverSourcePdf );
    }
    else
    {
        bounceDir = getSpecularBounce( seed,
                                    1,
                                    surf.normal,
                                    surf.roughness,
                                    surf.specularColor,
                                    surf.toViewerDir,
                                    oneOverSourcePdf );

        // swap to the common domain
        float oneOverDiffusePdf;
        {
            float z           = dot( bounceDir, surf.normal );
            oneOverDiffusePdf = z / M_PI;
        }
        oneOverSourcePdf *= oneOverDiffusePdf;
    }

    vec3 emis;
    const Surface hitSurf = traceBounce(surf.position + surf.normal * 0.01, 
                                        surf.roughness, 
                                        surf.instCustomIndex, 
                                        bounceDir, 
                                        FIRST_BOUNCE_MIP_BIAS,
                                        emis);
    emis *= globalUniform.emissionMapBoost;

    if (hitSurf.isSky)
    {
        SampleIndirect s = createSampleIndirect( //
            surf.position + bounceDir * MAX_RAY_LENGTH,
            -bounceDir,
            getSky( bounceDir ) );
        return s;
    }

    // calculate direct diffuse illumination in a hit position
    vec3 diffuse = processDirectIllumination(seed, hitSurf, 1);

#if INDIRECT_DIFFUSE_MAX_BOUNCES >= 2
#ifdef NRC_ENABLED
    const bool nrcOn        = globalUniform.nrcEnabled != 0;
    // short-circuit: when off, no RNG is consumed for the train decision,
    // keeping the random stream identical to the classic path
    const float trainProbability = uNRCParams.uTrainProbability;
    const bool isTrainSample = nrcOn && ( rnd16( seed, RANDOM_SALT_NRC_TRAIN_DECISION ) < trainProbability );
#endif

    // diffuse bounces after the first hit, gated at compile time by
    // INDIRECT_DIFFUSE_MAX_BOUNCES; the historical runtime toggle
    // (indirSecondBounce) once caused a strong red tint when re-enabled,
    // see git history for the original investigation TODO
    {
#ifdef NRC_ENABLED
        if( nrcOn )
        {
        if( isTrainSample )
        {
            // train sample: trace the full tail, cache its radiance as the target
            ShPayload secondPayload;
            vec3      secondDir;
            const vec3 secondBounce = processDiffuseTailBounces(seed,
                                                                hitSurf,
                                                                secondPayload,
                                                                secondDir);
            diffuse += secondBounce;

            uint slot = atomicAdd( uTotalTrainCount, 1 );
            if( slot < min( uNRCParams.uTrainBatchSize, NRC_TRAIN_RECORD_CAPACITY ) )
            {
                NRCTrainRecordRTGL record;
                record.bias_r             = secondBounce.r;
                record.bias_g             = secondBounce.g;
                record.bias_b             = secondBounce.b;
                record.factor_r           = 0.0;
                record.factor_g           = 0.0;
                record.factor_b           = 0.0;
                record.packed_input.instIdAndIndex      = uint( secondPayload.instIdAndIndex );
                record.packed_input.geomAndPrimIndex    = uint( secondPayload.geomAndPrimIndex );
                record.packed_input.barycentric_2x16U   = packHalf2x16( secondPayload.baryCoords );
                record.packed_input.scattered_dir_2x16U = packHalf2x16( secondDir.xy );
                uTrainRecords[ slot ] = record;
            }
        }
        else
        {
            // inference sample: replace the second bounce with the cached network output
            uint slot = atomicAdd( uEvalCount, 1 );
            if( slot < uint( globalUniform.renderWidth ) * uint( globalUniform.renderHeight ) )
            {
                NRCEvalRecordRTGL record;
                record.dst                             = NrcPackDstScreen( ivec2( gl_LaunchIDEXT.xy ) );
                record.packed_input.instIdAndIndex     = uint( g_lastBouncePayload.instIdAndIndex );
                record.packed_input.geomAndPrimIndex   = uint( g_lastBouncePayload.geomAndPrimIndex );
                record.packed_input.barycentric_2x16U  = packHalf2x16( g_lastBouncePayload.baryCoords );
                record.packed_input.scattered_dir_2x16U = packHalf2x16( -hitSurf.toViewerDir.xy );
                uEvalRecords[ slot ] = record;
            }
        }
        }
        else
        {
            // NRC disabled at runtime: do the full tail, classic path
            ShPayload unusedPayload;
            vec3      unusedDir;
            diffuse += processDiffuseTailBounces( seed,
                                                  hitSurf,
                                                  unusedPayload,
                                                  unusedDir );
        }
#else
        ShPayload unusedPayload;
        vec3      unusedDir;

        diffuse += processDiffuseTailBounces( seed,
                                            hitSurf,
                                            unusedPayload,
                                            unusedDir );
#endif
    }
#endif // INDIRECT_DIFFUSE_MAX_BOUNCES >= 2

    SampleIndirect s = createSampleIndirect( //
        hitSurf.position,
        hitSurf.normal,
        ( emis + diffuse ) * hitSurf.albedo );
    return s;
}
#endif // !RT_FORCE_COMPUTE

vec3 shade(const Surface surf, const SampleIndirect indir, float oneOverPdf)
{
    vec3  l  = safeNormalize2( unpackSampleIndirectPosition( indir ) - surf.position, vec3( 0 ) );
    float nl = dot(surf.normal, l);

    if (nl <= 0)
    {
        return vec3(0);
    }

    const vec3 radiance = decodeE5B9G9R9( indir.radianceE5 );

    if( useDiffuse( surf.roughness ) )
    {
        return oneOverPdf * nl * radiance * evalBRDFLambertian(1.0);
    }
    else
    {
        return oneOverPdf * nl * radiance * evalBRDFSmithGGX(surf.normal, surf.toViewerDir, l, surf.roughness, surf.specularColor);
    }
}

float targetPdfForIndirectSample(const SampleIndirect s)
{
    return getLuminance( decodeE5B9G9R9( s.radianceE5 ) );
}

bool testSurfaceForReuseIndirect(
    const ivec3 curChRenderArea, const ivec2 otherPix,
    float curDepth, float otherDepth,
    const vec3 curNormal, const vec3 otherNormal)
{
    const float DepthThreshold = 0.05;
    const float NormalThreshold = 0.0;

    return 
        testPixInRenderArea(otherPix, curChRenderArea) &&
        (abs(curDepth - otherDepth) / abs(curDepth) < DepthThreshold) &&
        (dot(curNormal, otherNormal) > NormalThreshold);
}



#define TEMPORAL_SAMPLES_INDIR    1
#define TEMPORAL_RADIUS_INDIR_MAX 8.0

#define SPATIAL_SAMPLES_INDIR 2
#define SPATIAL_RADIUS_INDIR  mix( 2.0, 8.0, clamp( globalUniform.renderHeight / 1080.0, 0.0, 1.0 ) ) 

#define DEBUG_TRACE_BIAS_CORRECT_RAY 0



#ifdef RT_RAYGEN_INDIRECT_INIT
void main()
{
    const ivec2 pix = ivec2(gl_LaunchIDEXT.xy);
    const uint seed = getRandomSeed(pix, globalUniform.frameId);
    uint salt = RANDOM_SALT_RESAMPLE_INDIRECT_BASE;

    Surface surf = fetchGbufferSurface(pix);
    surf.position += surf.toViewerDir * RAY_ORIGIN_LEAK_BIAS;

    if (surf.isSky)
    {
        restirIndirect_StoreInitialSample( pix, emptySampleIndirect(), 0.0 );
        return;
    }

    float          oneOverSourcePdf;
    SampleIndirect initial = processIndirect( seed, surf, oneOverSourcePdf );

    restirIndirect_StoreInitialSample( pix, initial, oneOverSourcePdf );
}
#endif // RT_RAYGEN_INDIRECT_INIT



#ifdef RT_RAYGEN_INDIRECT_FINAL
ReservoirIndirect loadInitialSampleAsReservoir( const ivec2 pix )
{
    float          oneOverSourcePdf;
    SampleIndirect s         = restirIndirect_LoadInitialSample( pix, oneOverSourcePdf );
    float          targetPdf = targetPdfForIndirectSample( s );

    ReservoirIndirect r = emptyReservoirIndirect();
    updateReservoirIndirect( r, s, targetPdf, oneOverSourcePdf, 0.5 );
    return r;
}

void main()
{
#ifndef RT_FORCE_COMPUTE
    const ivec2 pix  = ivec2( gl_LaunchIDEXT.xy );
#else
    const ivec2 pix = ivec2( gl_GlobalInvocationID.xy );
#endif
    const uint  seed = getRandomSeed( pix, globalUniform.frameId );
    uint        salt = RANDOM_SALT_RESAMPLE_INDIRECT_BASE;

    Surface surf = fetchGbufferSurface( pix );
#ifndef RT_FORCE_COMPUTE
    surf.position += surf.toViewerDir * RAY_ORIGIN_LEAK_BIAS;
#endif

    if( surf.isSky )
    {
        return;
    }


    ReservoirIndirect combined = loadInitialSampleAsReservoir( pix );


    // assuming that pix is checkerboarded
    const ivec3 chRenderArea = getCheckerboardedRenderArea( pix );
    const float motionZ           = texelFetch( framebufMotion_Sampler, pix, 0 ).z;
    const float depthCur          = texelFetch( framebufDepthWorld_Sampler, pix, 0 ).r;
    const vec2  posPrev           = getPrevScreenPos( framebufMotion_Sampler, pix );

    int spatialSamplesCount = int( SPATIAL_SAMPLES_INDIR * getDiffuseWeight( surf.roughness ) );


    for( int pixIndex = 0; pixIndex < TEMPORAL_SAMPLES_INDIR; pixIndex++ )
    {
        // TODO: need low discrepancy noise
        ivec2 pp;
        {
            vec2 rndOffset = rnd8_4( seed, salt++ ).xy * 2.0 - 1.0;
            rndOffset *= square( getDiffuseWeight( surf.roughness ) );

            pp = ivec2( floor( posPrev +
                               rndOffset * ( pixIndex == 0 ? 0.5 : TEMPORAL_RADIUS_INDIR_MAX ) ) );
        }

        {
            if( isSkyPix( pp ) )
            {
                continue;
            }
        }
        {
            const float depthPrev  = texelFetch( framebufDepthWorld_Prev_Sampler, pp, 0 ).r;
            const vec3  normalPrev = texelFetchNormal_Prev( pp );

            if( !testSurfaceForReuseIndirect(
                    chRenderArea, pp, depthCur, depthPrev - motionZ, surf.normal, normalPrev ) )
            {
                continue;
            }
        }
        {
            const float antilagAlpha_Indir = texelFetch(
                framebufDISGradientHistory_Sampler, pp / COMPUTE_ASVGF_STRATA_SIZE, 0 )[ 1 ];

            // if there's too much difference, don't use a temporal sample
            if( antilagAlpha_Indir > 0.25 )
            {
                continue;
            }
        }

        ReservoirIndirect temporal = restirIndirect_LoadReservoir_Prev( pp );
        // renormalize to prevent precision problems
        normalizeReservoirIndirect( temporal, 20 );

        float rnd = rnd16( seed, salt++ );
        updateCombinedReservoirIndirect( combined, temporal, rnd );

        break;
    }



    {
        uint nobiasM = combined.M; 

        for( int pixIndex = 0; pixIndex < spatialSamplesCount; pixIndex++ )
        {
            // TODO: need low discrepancy noise
            ivec2 pp;
            {
#if OPTIMIZE_MEM
                vec2 rndOffset = rnd16_2( seed, salt++ ) * 2.0 - 1.0;
#else
                vec2 rndOffset = rndBlueNoise16_2( seed, salt++ ) * 2.0 - 1.0;
#endif
                pp = pix + ivec2( rndOffset * SPATIAL_RADIUS_INDIR );
            }

            {
                if( isSkyPix( pp ) )
                {
                    continue;
                }

                const float depthOther  = texelFetch( framebufDepthWorld_Sampler, pp, 0 ).r;
                const vec3  normalOther = texelFetchNormal( pp );

                if( !testSurfaceForReuseIndirect(
                        chRenderArea, pp, depthCur, depthOther, surf.normal, normalOther ) )
                {
                    continue;
                }
            }

            ReservoirIndirect reservoir_q = loadInitialSampleAsReservoir( pp );

            float oneOverJacobian;
            {
                const vec3 x1_r = surf.position;
                const vec3 x1_q = texelFetch( framebufSurfacePosition_Sampler, pp, 0 ).xyz;

                const vec3 x2_q = unpackSampleIndirectPosition( reservoir_q.selected );
                const vec3 n2_q = decodeNormal( reservoir_q.selected.normalPacked );

                const DirectionAndLength phi_r = calcDirectionAndLengthSafe( x2_q, x1_r );
                const DirectionAndLength phi_q = calcDirectionAndLengthSafe( x2_q, x1_q );

                oneOverJacobian =
                    safePositiveRcp( getGeometryFactorClamped( n2_q, phi_r.dir, phi_r.len ) ) *
                    getGeometryFactorClamped( n2_q, phi_q.dir, phi_q.len );

    #if SHIPPING_HACK
                oneOverJacobian = clamp( oneOverJacobian, 0.0, 1.0 );
    #endif
            }

            float targetPdf_curSurf = 0.0;

#if DEBUG_TRACE_BIAS_CORRECT_RAY
            if( !traceShadowRay(
                    surf.instCustomIndex, surf.position, reservoir_q.selected.position, false ) )
#endif
            {
                targetPdf_curSurf =
                    targetPdfForIndirectSample( reservoir_q.selected ) * oneOverJacobian;
            }
#if OPTIMIZE_MEM
            float rnd = rnd16( seed, salt++ );
#else
            float rnd = rndBlueNoise32( seed, salt++ );
#endif
            updateCombinedReservoirIndirect_newSurf(
                combined, reservoir_q, targetPdf_curSurf, rnd );

            if( targetPdf_curSurf > 0.0 )
            {
                nobiasM += reservoir_q.M;
            }
        }

        combined.M = nobiasM;
    }

    restirIndirect_StoreReservoir( pix, combined );



    const vec3 indirRadiance =
        shade( surf, combined.selected, calcSelectedSampleWeightIndirect( combined ) );

    const vec3 surfToHitPoint = unpackSampleIndirectPosition( combined.selected ) - surf.position;

    {
        const vec3 direct = texelFetchUnfilteredSpecular( pix );
        // save indirect hit distance, if brighter than the direct light
        if( getLuminance( direct ) < getLuminance( indirRadiance ) )
        {
            imageStore(
                framebufViewDirection, pix, vec4( -surf.toViewerDir, length( surfToHitPoint ) ) );
        }


        // demodulate for denoising
        imageStoreUnfilteredSpecular( pix,
                                      direct + demodulateSpecular( indirRadiance, surf.specularColor ) );
    }

    {
        imageStoreUnfilteredIndir( pix, indirRadiance );
    }
}
#endif // RT_RAYGEN_INDIRECT_FINAL
