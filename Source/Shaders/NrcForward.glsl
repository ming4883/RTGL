// NrcForward.glsl - forward-only NN kernels for inference
// Ported from VkNRC (AdamYuan/VkNRC, MIT), NN_nv.glsl

fcoopmatNV< 16, gl_ScopeSubgroup, 16, 16 > act_coopmats[ COOPMAT_X ][ SUBGROUP_ACT_COOPMAT_Y ];
fcoopmatNV< 16, gl_ScopeSubgroup, 16, 16 > out_coopmats[ COOPMAT_X ][ SUBGROUP_ACT_COOPMAT_Y ];

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
    // load weights into shared memory
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

    // zero-initialize accumulators
    for( uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y )
    {
        for( uint x = 0; x < COOPMAT_X; ++x )
        {
            out_coopmats[ x ][ y ] = fcoopmatNV< 16, gl_ScopeSubgroup, 16, 16 >( 0 );
        }
    }

    // MMA
    for( uint w_y = 0; w_y < COOPMAT_X; ++w_y )
    {
        for( uint x = 0; x < COOPMAT_X; ++x )
        {
            coopMatLoadNV( weight_coopmat, SHARED_WEIGHTS, MAT64_COOPMAT_ELEMENT( x, w_y ),
                           MAT64_COOPMAT_STRIDE, false ); // row-major (weights)
            for( uint a_y = 0; a_y < SUBGROUP_ACT_COOPMAT_Y; ++a_y )
            {
                out_coopmats[ w_y ][ a_y ] =
                    coopMatMulAddNV( weight_coopmat, act_coopmats[ x ][ a_y ], out_coopmats[ w_y ][ a_y ] );
            }
        }
    }
    subgroupBarrier();

    // ReLU
    for( uint x = 0; x < COOPMAT_X; ++x )
    {
        for( uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y )
        {
            for( uint k = 0; k < out_coopmats[ x ][ y ].length(); ++k )
            {
                act_coopmats[ x ][ y ][ k ] = max( out_coopmats[ x ][ y ][ k ], float16_t( 0 ) );
            }
        }
    }
}

vec3 NNForward3( uint layer )
{
    // load the final 64x3 weight block into shared memory
    {
        uint base = NRC_HIDDEN_LAYERS * WEIGHT_64_UV4_COUNT;
        SHARED_WEIGHTS[ gl_LocalInvocationID.x ] =
            gl_LocalInvocationID.x < WEIGHT_3_UV4_COUNT
                ? uWeights[ base + gl_LocalInvocationID.x ]
                : uvec4( 0u );
        subgroupBarrier();
    }

    fcoopmatNV< 16, gl_ScopeSubgroup, 16, 16 > weight_coopmat;
    fcoopmatNV< 16, gl_ScopeSubgroup, 16, 16 > dst3[ SUBGROUP_ACT_COOPMAT_Y ];

    for( uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y )
    {
        dst3[ y ] = fcoopmatNV< 16, gl_ScopeSubgroup, 16, 16 >( 0 );
    }

    for( uint x = 0; x < COOPMAT_X; ++x )
    {
        coopMatLoadNV( weight_coopmat, SHARED_WEIGHTS, MAT64_COOPMAT_ELEMENT( x, 0 ),
                       MAT64_COOPMAT_STRIDE, false );
        for( uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y )
        {
            dst3[ y ] = coopMatMulAddNV( weight_coopmat, act_coopmats[ x ][ y ], dst3[ y ] );
        }
    }
    subgroupBarrier();

    // store the 16x16 result tiles, then read back the 3 valid columns
    for( uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y )
    {
        uint workgroup_y = gl_SubgroupID * SUBGROUP_ACT_COOPMAT_Y + y;
        coopMatStoreNV( dst3[ y ], SHARED_WEIGHTS, MAT64_COOPMAT_ELEMENT( 0, workgroup_y ),
                        MAT64_COOPMAT_STRIDE, true );
    }
    subgroupBarrier();

    uvec2 uv2 = SHARED_WEIGHTS[ gl_LocalInvocationID.x * UV4_X ].rg;
    return vec3( unpackHalf2x16( uv2.x ), unpackHalf2x16( uv2.y ).x );
}