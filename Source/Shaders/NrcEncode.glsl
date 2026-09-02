// NrcEncode.glsl - input encoding functions (from VkNRC NRCRecord.glsl, MIT)

vec2 NRCSphEncode( vec3 d )
{
    return vec2( d.xy == vec2( 0 ) ? 0.5 : 0.5 + atan( d.y, d.x ) / ( 2.0 * 3.14159265358979 ),
                 acos( clamp( d.z, -1.0, 1.0 ) ) / 3.14159265358979 );
}

vec4 _quartic_cdf( vec4 x, float inv_radius )
{
    vec4 u  = x * inv_radius;
    vec4 u2 = u * u;
    vec4 u4 = u2 * u2;
    return clamp( ( 15.0 / 16.0 ) * u * ( 1 - ( 2.0 / 3.0 ) * u2 + ( 1.0 / 5.0 ) * u4 ) + 0.5, vec4( 0 ), vec4( 1 ) );
}

// x in [0, 1]
vec4 NRCOneBlob4Encode( float x )
{
    vec4 l = vec4( 0, 0.25, 0.5, 0.75 ), r = vec4( 0.25, 0.5, 0.75, 1 );
    return _quartic_cdf( r - x, 4 ) - _quartic_cdf( l - x, 4 );
}

vec4 _nrc_tri( vec4 x )
{
    return 2.0 * abs( mod( x - 0.5, 2.0 ) - 1.0 ) - 1.0;
}

mat3x4 NRCFrequencyEncode( float p )
{
    mat3x4 f = mat3x4( vec4( 1, 2, 4, 8 ), vec4( 16, 32, 64, 128 ), vec4( 256, 512, 1024, 2048 ) ) * p;
    return mat3x4( _nrc_tri( f[ 0 ] ), _nrc_tri( f[ 1 ] ), _nrc_tri( f[ 2 ] ) );
}

uvec4 NRCPackHalf8x16( vec4 a, vec4 b )
{
    return uvec4( packHalf2x16( a.xy ), packHalf2x16( a.zw ), packHalf2x16( b.xy ), packHalf2x16( b.zw ) );
}

// scattered_dir and normal must be pre-encoded through NRCSphEncode
void NRCInputEncode( vec3 position,
                     vec3 scattered_dir_sph,
                     vec3 normal_sph,
                     float roughness,
                     vec3 diffuse,
                     vec3 specular,
                     out uvec4 o[ 8 ] )
{
    mat3x4 pos_freq_0 = NRCFrequencyEncode( position.x );
    mat3x4 pos_freq_1 = NRCFrequencyEncode( position.y );
    mat3x4 pos_freq_2 = NRCFrequencyEncode( position.z );
    vec4   scat_ob_0  = NRCOneBlob4Encode( scattered_dir_sph.x );
    vec4   scat_ob_1  = NRCOneBlob4Encode( scattered_dir_sph.y );
    vec4   norm_ob_0  = NRCOneBlob4Encode( normal_sph.x );
    vec4   norm_ob_1  = NRCOneBlob4Encode( normal_sph.y );
    vec4   r_ob       = NRCOneBlob4Encode( 1 - exp( -roughness ) );

    o[ 0 ] = NRCPackHalf8x16( pos_freq_0[ 0 ], pos_freq_0[ 1 ] );
    o[ 1 ] = NRCPackHalf8x16( pos_freq_0[ 2 ], pos_freq_1[ 0 ] );
    o[ 2 ] = NRCPackHalf8x16( pos_freq_1[ 1 ], pos_freq_1[ 2 ] );
    o[ 3 ] = NRCPackHalf8x16( pos_freq_2[ 0 ], pos_freq_2[ 1 ] );
    o[ 4 ] = NRCPackHalf8x16( pos_freq_2[ 2 ], scat_ob_0 );
    o[ 5 ] = NRCPackHalf8x16( scat_ob_1, norm_ob_0 );
    o[ 6 ] = NRCPackHalf8x16( norm_ob_1, r_ob );
    o[ 7 ] = NRCPackHalf8x16( vec4( diffuse, specular.r ),
                              vec4( specular.gb, 1, 1 ) );
}