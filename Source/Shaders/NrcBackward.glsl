// NrcBackward.glsl - forward + backward NN kernels for the training pass
// Ported from VkNRC (AdamYuan/VkNRC, MIT), NN_nv.glsl (NN_BACKPROPAGATION part)

fcoopmatNV< 16, gl_ScopeSubgroup, 16, 16 > act_coopmats[ COOPMAT_X ][ SUBGROUP_ACT_COOPMAT_Y ];
fcoopmatNV< 16, gl_ScopeSubgroup, 16, 16 > out_coopmats[ COOPMAT_X ][ SUBGROUP_ACT_COOPMAT_Y ];
fcoopmatNV< 16, gl_ScopeSubgroup, 16, 16 > da3_coopmats[ SUBGROUP_ACT_COOPMAT_Y ];

shared uvec4 SHARED_WEIGHTS[ WEIGHT_64_UV4_COUNT ];

void NNLoadInput( uvec4 inputs[ 8 ] )
{
    for( uint x = 0; x < UV4_X; ++x )
    {
        SHARED_WEIGHTS[ gl_LocalInvocationID.x * UV4_X + x ] = inputs[ x ];
    }
    subgroupBarrier();

    for( uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y )
    {
        uint workgroup_y = gl_SubgroupID * SUBGROUP_ACT_COOPMAT_Y + y;
        for( uint x = 0; x < COOPMAT_X; ++x )
        {
            coopMatLoadNV( act_coopmats[ x ][ y ], SHARED_WEIGHTS, MAT64_COOPMAT_ELEMENT( x, workgroup_y ),
                           MAT64_COOPMAT_STRIDE, true ); // column-major (activations)
        }
    }
    subgroupBarrier();
}

void NNForward64_ReLU( uint layer )
{
    {
        uint perThread = WEIGHT_64_UV4_COUNT / 128;
        for( uint i = 0; i < perThread; ++i )
        {
            SHARED_WEIGHTS[ gl_LocalInvocationID.x * perThread + i ] =
                uWeights[ layer * WEIGHT_64_UV4_COUNT + gl_LocalInvocationID.x * perThread + i ];
        }
        subgroupBarrier();
    }

    fcoopmatNV< 16, gl_ScopeSubgroup, 16, 16 > weight_coopmat;

    for( uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y )
    {
        for( uint x = 0; x < COOPMAT_X; ++x )
        {
            out_coopmats[ x ][ y ] = fcoopmatNV< 16, gl_ScopeSubgroup, 16, 16 >( 0 );
        }
    }

    for( uint w_y = 0; w_y < COOPMAT_X; ++w_y )
    {
        for( uint x = 0; x < COOPMAT_X; ++x )
        {
            coopMatLoadNV( weight_coopmat, SHARED_WEIGHTS, MAT64_COOPMAT_ELEMENT( x, w_y ),
                           MAT64_COOPMAT_STRIDE, false );
            for( uint a_y = 0; a_y < SUBGROUP_ACT_COOPMAT_Y; ++a_y )
            {
                out_coopmats[ w_y ][ a_y ] =
                    coopMatMulAddNV( weight_coopmat, act_coopmats[ x ][ a_y ], out_coopmats[ w_y ][ a_y ] );
            }
        }
    }
    subgroupBarrier();

    // ReLU + save the ReLU mask for backward (NaN for zero activations)
    for( uint x = 0; x < COOPMAT_X; ++x )
    {
        for( uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y )
        {
            for( uint k = 0; k < out_coopmats[ x ][ y ].length(); ++k )
            {
                act_coopmats[ x ][ y ][ k ] =
                    out_coopmats[ x ][ y ][ k ] > float16_t( 0 )
                        ? out_coopmats[ x ][ y ][ k ]
                        : uint16BitsToHalf( uint16_t( 0x7FFF ) ); // NaN
            }
        }
    }
}

void NNForward3( uint layer )
{
    {
        uint base = NRC_HIDDEN_LAYERS * WEIGHT_64_UV4_COUNT;
        SHARED_WEIGHTS[ gl_LocalInvocationID.x ] =
            gl_LocalInvocationID.x < WEIGHT_3_UV4_COUNT
                ? uWeights[ base + gl_LocalInvocationID.x ]
                : uvec4( 0u );
        subgroupBarrier();
    }

    fcoopmatNV< 16, gl_ScopeSubgroup, 16, 16 > weight_coopmat;

    for( uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y )
    {
        da3_coopmats[ y ] = fcoopmatNV< 16, gl_ScopeSubgroup, 16, 16 >( 0 );
    }

    for( uint x = 0; x < COOPMAT_X; ++x )
    {
        coopMatLoadNV( weight_coopmat, SHARED_WEIGHTS, MAT64_COOPMAT_ELEMENT( x, 0 ),
                       MAT64_COOPMAT_STRIDE, false );
        for( uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y )
        {
            da3_coopmats[ y ] = coopMatMulAddNV( weight_coopmat, act_coopmats[ x ][ y ], da3_coopmats[ y ] );
        }
    }
    subgroupBarrier();
}

void NNLoadDA3_L2Loss( vec3 predict, vec3 target, vec3 loss_scale )
{
    vec3 d_l2 = 2.0 * ( predict - target ) * loss_scale;

    SHARED_WEIGHTS[ gl_LocalInvocationID.x * UV4_X ] =
        uvec4( packHalf2x16( d_l2.xy ), packHalf2x16( vec2( d_l2.z, 0 ) ), 0u, 0u );
    for( uint i = 1; i < 16 / FP16_PER_UV4; ++i )
    {
        SHARED_WEIGHTS[ gl_LocalInvocationID.x * UV4_X + i ] = uvec4( 0u );
    }
    subgroupBarrier();

    for( uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y )
    {
        uint workgroup_y = gl_SubgroupID * SUBGROUP_ACT_COOPMAT_Y + y;
        coopMatLoadNV( da3_coopmats[ y ], SHARED_WEIGHTS, MAT64_COOPMAT_ELEMENT( 0, workgroup_y ),
                       MAT64_COOPMAT_STRIDE, false ); // transposed load (row-major)
    }
    subgroupBarrier();
}

// shared helper: store the given coopmats, replace non-positive activations with NaN
void _nn_act_64_relu_mask_t( )
{
    for( uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y )
    {
        uint workgroup_y = gl_SubgroupID * SUBGROUP_ACT_COOPMAT_Y + y;
        for( uint x = 0; x < COOPMAT_X; ++x )
        {
            coopMatStoreNV( act_coopmats[ x ][ y ], SHARED_WEIGHTS, MAT64_COOPMAT_ELEMENT( x, workgroup_y ),
                            MAT64_COOPMAT_STRIDE, true );
        }
    }
    subgroupBarrier();

    float16_t nan_16 = uint16BitsToHalf( uint16_t( 0x7FFF ) );
    for( uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y )
    {
        uint workgroup_y = gl_SubgroupID * SUBGROUP_ACT_COOPMAT_Y + y;
        for( uint x = 0; x < COOPMAT_X; ++x )
        {
            coopMatLoadNV( act_coopmats[ x ][ y ], SHARED_WEIGHTS, MAT64_COOPMAT_ELEMENT( x, workgroup_y ),
                           MAT64_COOPMAT_STRIDE, false ); // transposed reload
            for( uint k = 0; k < act_coopmats[ x ][ y ].length(); ++k )
            {
                act_coopmats[ x ][ y ][ k ] = act_coopmats[ x ][ y ][ k ] > float16_t( 0 ) ? float16_t( 0 ) : nan_16;
            }
        }
    }
    subgroupBarrier();
}

void NNBackwardDA3_ReLU( uint layer )
{
    // inverse ReLU on the (transposed) activation masks
    _nn_act_64_relu_mask_t( );

    // reload the final 64x3 weight block
    {
        uint base = NRC_HIDDEN_LAYERS * WEIGHT_64_UV4_COUNT;
        SHARED_WEIGHTS[ gl_LocalInvocationID.x ] =
            gl_LocalInvocationID.x < WEIGHT_3_UV4_COUNT
                ? uWeights[ base + gl_LocalInvocationID.x ]
                : uvec4( 0u );
        subgroupBarrier();
    }

    // da3^T (128,16) x weights (16,64) = dA^T (128,64)
    fcoopmatNV< 16, gl_ScopeSubgroup, 16, 16 > weight_coopmat;
    for( uint x = 0; x < COOPMAT_X; ++x )
    {
        coopMatLoadNV( weight_coopmat, SHARED_WEIGHTS, MAT64_COOPMAT_ELEMENT( x, 0 ),
                       MAT64_COOPMAT_STRIDE, false );
        for( uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y )
        {
            out_coopmats[ x ][ y ] = coopMatMulAddNV( da3_coopmats[ y ], weight_coopmat, out_coopmats[ x ][ y ] );
            for( uint k = 0; k < out_coopmats[ x ][ y ].length(); ++k )
            {
                out_coopmats[ x ][ y ][ k ] =
                    isnan( out_coopmats[ x ][ y ][ k ] ) ? float16_t( 0 ) : out_coopmats[ x ][ y ][ k ];
            }
        }
    }
    subgroupBarrier();
}

void NNBackwardDA64_ReLU( uint layer )
{
    // inverse ReLU on the (transposed) activation masks
    _nn_act_64_relu_mask_t( );

    // da^T (128,64) x weights (64,64) = dA^T (128,64)
    {
        uint perThread = WEIGHT_64_UV4_COUNT / 128;
        for( uint i = 0; i < perThread; ++i )
        {
            SHARED_WEIGHTS[ gl_LocalInvocationID.x * perThread + i ] =
                uWeights[ layer * WEIGHT_64_UV4_COUNT + gl_LocalInvocationID.x * perThread + i ];
        }
        subgroupBarrier();
    }

    fcoopmatNV< 16, gl_ScopeSubgroup, 16, 16 > weight_coopmat;
    for( uint w_y = 0; w_y < COOPMAT_X; ++w_y )
    {
        for( uint x = 0; x < COOPMAT_X; ++x )
        {
            coopMatLoadNV( weight_coopmat, SHARED_WEIGHTS, MAT64_COOPMAT_ELEMENT( x, w_y ),
                           MAT64_COOPMAT_STRIDE, false );
            for( uint a_y = 0; a_y < SUBGROUP_ACT_COOPMAT_Y; ++a_y )
            {
                out_coopmats[ x ][ a_y ] =
                    coopMatMulAddNV( act_coopmats[ w_y ][ a_y ], weight_coopmat, out_coopmats[ x ][ a_y ] );
                for( uint k = 0; k < out_coopmats[ x ][ a_y ].length(); ++k )
                {
                    out_coopmats[ x ][ a_y ][ k ] =
                        isnan( out_coopmats[ x ][ a_y ][ k ] ) ? float16_t( 0 ) : out_coopmats[ x ][ a_y ][ k ];
                }
            }
        }
    }
    subgroupBarrier();
}

void NNUpdateDW3( uint layer )
{
    // (act (64,128) x da3^T (128,16))^T = dW (16,64)
    fcoopmatNV< 16, gl_ScopeSubgroup, 16, 16 > dw_coopmats[ COOPMAT_X ];
    for( uint w_y = 0; w_y < COOPMAT_X; ++w_y )
    {
        dw_coopmats[ w_y ] = fcoopmatNV< 16, gl_ScopeSubgroup, 16, 16 >( 0 );
        for( uint a_y = 0; a_y < SUBGROUP_ACT_COOPMAT_Y; ++a_y )
        {
            dw_coopmats[ w_y ] = coopMatMulAddNV( act_coopmats[ w_y ][ a_y ], da3_coopmats[ a_y ],
                                                  dw_coopmats[ w_y ] );
        }
    }
    subgroupBarrier();

    for( uint y = 0; y < COOPMAT_X; ++y )
    {
        coopMatStoreNV( dw_coopmats[ y ], SHARED_WEIGHTS, MAT64_COOPMAT_ELEMENT( y, gl_SubgroupID ),
                        MAT64_COOPMAT_STRIDE, false ); // row-major store (dW)
    }
    subgroupBarrier();

    // reduce dW across subgroups, then atomicAdd into uGradients
    vec2 d_w[ FP16_PER_UV4 / 2 ];
    for( uint i = 0; i < FP16_PER_UV4 / 2; ++i )
    {
        d_w[ i ] = vec2( 0 );
    }
    for( uint s = 0; s < NRC_SUBGROUP_COUNT; ++s )
    {
        uvec4 d_w_uv4 = SHARED_WEIGHTS[ MAT64_COOPMAT_ELEMENT( 0, s ) + gl_LocalInvocationID.x ];
        for( uint i = 0; i < FP16_PER_UV4 / 2; ++i )
        {
            d_w[ i ] += unpackHalf2x16( d_w_uv4[ i ] );
        }
    }
    subgroupBarrier();

    if( gl_LocalInvocationID.x < WEIGHT_3_UV4_COUNT )
    {
        uint kWeightFP16Base = NRC_HIDDEN_LAYERS * WEIGHT_64_UV4_COUNT * 8 +
                               layer * WEIGHT_3_UV4_COUNT * 8 + gl_LocalInvocationID.x * FP16_PER_UV4;
        for( uint i = 0; i < FP16_PER_UV4 / 2; ++i )
        {
            uint g0 = min( kWeightFP16Base + ( i << 1u ),     NRC_WEIGHT_COUNT - 1 );
            uint g1 = min( kWeightFP16Base + ( i << 1u | 1u ), NRC_WEIGHT_COUNT - 1 );
            atomicAdd( uGradients[ g0 ], d_w[ i ].x, gl_ScopeQueueFamily, gl_StorageSemanticsBuffer,
                       gl_SemanticsRelaxed );
            atomicAdd( uGradients[ g1 ], d_w[ i ].y, gl_ScopeQueueFamily, gl_StorageSemanticsBuffer,
                       gl_SemanticsRelaxed );
        }
    }
}

void NNUpdateDW64( uint layer )
{
    // (act (64,128) x da^T (128,64))^T = dW (64,64)
    fcoopmatNV< 16, gl_ScopeSubgroup, 16, 16 > dw_coopmats[ COOPMAT_X ][ COOPMAT_X ];
    for( uint w_y = 0; w_y < COOPMAT_X; ++w_y )
    {
        for( uint x = 0; x < COOPMAT_X; ++x )
        {
            dw_coopmats[ w_y ][ x ] = fcoopmatNV< 16, gl_ScopeSubgroup, 16, 16 >( 0 );
            for( uint a_y = 0; a_y < SUBGROUP_ACT_COOPMAT_Y; ++a_y )
            {
                dw_coopmats[ w_y ][ x ] =
                    coopMatMulAddNV( act_coopmats[ w_y ][ a_y ], act_coopmats[ x ][ a_y ], dw_coopmats[ w_y ][ x ] );
            }
        }
    }
    subgroupBarrier();

    for( uint y = 0; y < COOPMAT_X; ++y )
    {
        for( uint x = 0; x < COOPMAT_X; ++x )
        {
            coopMatStoreNV( dw_coopmats[ y ][ x ], SHARED_WEIGHTS,
                            MAT64_COOPMAT_ELEMENT( y, x + COOPMAT_X * gl_SubgroupID ),
                            MAT64_COOPMAT_STRIDE, false );
        }
    }
    subgroupBarrier();

    vec2 d_w[ THREAD_WEIGHT_64_UV4_COUNT ][ FP16_PER_UV4 / 2 ];
    for( uint u = 0; u < THREAD_WEIGHT_64_UV4_COUNT; ++u )
    {
        for( uint i = 0; i < ( FP16_PER_UV4 / 2 ); ++i )
        {
            d_w[ u ][ i ] = vec2( 0 );
        }
    }

    uint kSharedUV4Base = gl_LocalInvocationID.x * THREAD_WEIGHT_64_UV4_COUNT;
    for( uint s = 0; s < NRC_SUBGROUP_COUNT; ++s )
    {
        uint kSubgroupUV4Base = MAT64_COOPMAT_ELEMENT( 0, COOPMAT_X * s ) + kSharedUV4Base;
        for( uint u = 0; u < THREAD_WEIGHT_64_UV4_COUNT; ++u )
        {
            uvec4 d_w_uv4 = SHARED_WEIGHTS[ kSubgroupUV4Base + u ];
            for( uint i = 0; i < ( FP16_PER_UV4 / 2 ); ++i )
            {
                d_w[ u ][ i ] += unpackHalf2x16( d_w_uv4[ i ] );
            }
        }
    }
    subgroupBarrier();

    uint kWeightUV4Base = layer * WEIGHT_64_UV4_COUNT + kSharedUV4Base;
    for( uint u = 0; u < THREAD_WEIGHT_64_UV4_COUNT; ++u )
    {
        uint kWeightFP16Base = ( kWeightUV4Base + u ) * FP16_PER_UV4;
        for( uint i = 0; i < ( FP16_PER_UV4 / 2 ); ++i )
        {
            uint g0 = min( kWeightFP16Base + ( i << 1u ),      NRC_WEIGHT_COUNT - 1 );
            uint g1 = min( kWeightFP16Base + ( i << 1u | 1u ), NRC_WEIGHT_COUNT - 1 );
            atomicAdd( uGradients[ g0 ], d_w[ u ][ i ].x, gl_ScopeQueueFamily, gl_StorageSemanticsBuffer,
                       gl_SemanticsRelaxed );
            atomicAdd( uGradients[ g1 ], d_w[ u ][ i ].y, gl_ScopeQueueFamily, gl_StorageSemanticsBuffer,
                       gl_SemanticsRelaxed );
        }
    }
}