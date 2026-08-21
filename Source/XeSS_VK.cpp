// Copyright (c) 2024-2025 V.Shirokii
// Copyright (c) 2021 Sultim Tsyrendashiev
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

#include "XeSS_VK.h"

#if defined( _WIN32 )

    #include "HaltonSequence.h"
    #include "LibraryConfig.h"
    #include "RenderResolutionHelper.h"
    #include "RgException.h"
    #include "Utils.h"

    #include <xess/xess_vk.h>

    #include <cassert>

namespace
{
constexpr RTGL1::FramebufferImageIndex OUTPUT_IMAGE_INDEX = RTGL1::FB_IMAGE_INDEX_UPSCALED_PONG;

#define DECLARE_DLL_FUNC( f ) decltype( &( f ) ) f = nullptr

struct XessSdk
{
    DECLARE_DLL_FUNC( xessVKGetRequiredInstanceExtensions );
    DECLARE_DLL_FUNC( xessVKGetRequiredDeviceExtensions );
    DECLARE_DLL_FUNC( xessVKGetRequiredDeviceFeatures );
    DECLARE_DLL_FUNC( xessVKCreateContext );
    DECLARE_DLL_FUNC( xessVKBuildPipelines );
    DECLARE_DLL_FUNC( xessVKInit );
    DECLARE_DLL_FUNC( xessVKGetInitParams );
    DECLARE_DLL_FUNC( xessVKExecute );
    DECLARE_DLL_FUNC( xessGetInputResolution );
    DECLARE_DLL_FUNC( xessSetVelocityScale );
    DECLARE_DLL_FUNC( xessSetLoggingCallback );
    DECLARE_DLL_FUNC( xessDestroyContext );
};

#undef DECLARE_DLL_FUNC

XessSdk pfn{};
HMODULE g_xessModule = nullptr;

HMODULE LoadLibrary_Path( const std::filesystem::path& p )
{
    HMODULE dll = LoadLibraryW( p.c_str() );
    if( !dll )
    {
        RTGL1::debug::Error( "XeSS: Failed to load DLL '{}'", p.string() );
    }
    return dll;
}

#define RETURN_FAIL         \
    pfn = {};               \
    if( g_xessModule )      \
    {                       \
        FreeLibrary( g_xessModule ); \
        g_xessModule = nullptr;      \
    }                       \
    return false

#define GET_FUNC( f )                                                           \
    do                                                                          \
    {                                                                           \
        pfn.f = reinterpret_cast< decltype( pfn.f ) >( GetProcAddress( g_xessModule, #f ) ); \
        if( !pfn.f )                                                            \
        {                                                                       \
            RTGL1::debug::Error( "XeSS: Failed to load DLL function: '" #f "'" ); \
            RETURN_FAIL;                                                        \
        }                                                                       \
    } while( 0 )

bool LoadDllFunctions( const std::filesystem::path& folder )
{
    if( g_xessModule )
    {
        return true;
    }

    g_xessModule = LoadLibrary_Path( folder / "libxess.dll" );
    if( !g_xessModule )
    {
        RETURN_FAIL;
    }

    GET_FUNC( xessVKGetRequiredInstanceExtensions );
    GET_FUNC( xessVKGetRequiredDeviceExtensions );
    GET_FUNC( xessVKGetRequiredDeviceFeatures );
    GET_FUNC( xessVKCreateContext );
    GET_FUNC( xessVKBuildPipelines );
    GET_FUNC( xessVKInit );
    GET_FUNC( xessVKGetInitParams );
    GET_FUNC( xessVKExecute );
    GET_FUNC( xessGetInputResolution );
    GET_FUNC( xessSetVelocityScale );
    GET_FUNC( xessSetLoggingCallback );
    GET_FUNC( xessDestroyContext );

    return true;
}

#undef RETURN_FAIL
#undef GET_FUNC

void FreeDll()
{
    if( g_xessModule )
    {
        FreeLibrary( g_xessModule );
        g_xessModule = nullptr;
    }
    pfn = {};
}

void CheckError( xess_result_t r )
{
    if( r != XESS_RESULT_SUCCESS )
    {
        RTGL1::debug::Error( "XeSS: Fail, xess_result_t={}", static_cast< int >( r ) );
        throw RTGL1::RgException( RG_RESULT_GRAPHICS_API_ERROR, "Can't initialize XeSS" );
    }
}

xess_quality_settings_t ToXessQuality( RgRenderResolutionMode mode )
{
    switch( mode )
    {
        case RG_RENDER_RESOLUTION_MODE_ULTRA_PERFORMANCE:
            return XESS_QUALITY_SETTING_ULTRA_PERFORMANCE;
        case RG_RENDER_RESOLUTION_MODE_PERFORMANCE: return XESS_QUALITY_SETTING_PERFORMANCE;
        case RG_RENDER_RESOLUTION_MODE_BALANCED: return XESS_QUALITY_SETTING_BALANCED;
        case RG_RENDER_RESOLUTION_MODE_QUALITY: return XESS_QUALITY_SETTING_QUALITY;
        case RG_RENDER_RESOLUTION_MODE_NATIVE_AA: return XESS_QUALITY_SETTING_AA;
        default:
            assert( 0 );
            return XESS_QUALITY_SETTING_BALANCED;
    }
}

void PrintXeSSLog( const char* message, xess_logging_level_t level )
{
    using namespace RTGL1;

    if( !message )
    {
        return;
    }

    switch( level )
    {
        case XESS_LOGGING_LEVEL_DEBUG: debug::Verbose( "XeSS: {}", message ); break;
        case XESS_LOGGING_LEVEL_INFO: debug::Info( "XeSS: {}", message ); break;
        case XESS_LOGGING_LEVEL_WARNING: debug::Warning( "XeSS: {}", message ); break;
        case XESS_LOGGING_LEVEL_ERROR: debug::Error( "XeSS: {}", message ); break;
        default: assert( 0 );
    }
}

xess_vk_image_view_info ToXeSSImageInfo( RTGL1::FramebufferImageIndex  fbImage,
                                         uint32_t                      frameIndex,
                                         const RTGL1::Framebuffers&    framebuffers,
                                         const RTGL1::ResolutionState& resolutionState )
{
    auto [ image, view, format, sz ] =
        framebuffers.GetImageHandles( fbImage, frameIndex, resolutionState );

    return xess_vk_image_view_info{
        .imageView        = view,
        .image            = image,
        .subresourceRange = { .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                              .baseMipLevel   = 0,
                              .levelCount     = 1,
                              .baseArrayLayer = 0,
                              .layerCount     = 1 },
        .format           = format,
        .width            = sz.width,
        .height           = sz.height,
    };
}

template< size_t N >
void InsertBarriers( VkCommandBuffer            cmd,
                     uint32_t                   frameIndex,
                     const RTGL1::Framebuffers& framebuffers,
                     const RTGL1::FramebufferImageIndex ( &inputsAndOutput )[ N ],
                     bool isBackwards )
{
    assert( std::ranges::contains( inputsAndOutput, OUTPUT_IMAGE_INDEX ) );

    VkImageMemoryBarrier2 barriers[ N ];

    for( size_t i = 0; i < N; i++ )
    {
        barriers[ i ] = {
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask        = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask       = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
            .dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask       = inputsAndOutput[ i ] == OUTPUT_IMAGE_INDEX
                                       ? VK_ACCESS_2_SHADER_WRITE_BIT
                                       : VK_ACCESS_2_SHADER_READ_BIT,
            .oldLayout           = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout           = inputsAndOutput[ i ] == OUTPUT_IMAGE_INDEX
                                       ? VK_IMAGE_LAYOUT_GENERAL
                                       : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = framebuffers.GetImage( inputsAndOutput[ i ], frameIndex ),
            .subresourceRange    = { .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                                     .baseMipLevel   = 0,
                                     .levelCount     = 1,
                                     .baseArrayLayer = 0,
                                     .layerCount     = 1 },
        };

        if( isBackwards )
        {
            auto& b = barriers[ i ];

            std::swap( b.srcStageMask, b.dstStageMask );
            std::swap( b.srcAccessMask, b.dstAccessMask );
            std::swap( b.oldLayout, b.newLayout );
            std::swap( b.srcQueueFamilyIndex, b.dstQueueFamilyIndex );
        }
    }

    VkDependencyInfoKHR dependencyInfo = {
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
        .imageMemoryBarrierCount = uint32_t( std::size( barriers ) ),
        .pImageMemoryBarriers    = barriers,
    };

    RTGL1::svkCmdPipelineBarrier2KHR( cmd, &dependencyInfo );
}
}

RTGL1::XeSS_VK::XeSS_VK( VkInstance _instance, VkDevice _device, VkPhysicalDevice _physDevice )
    : instance{ _instance }
    , device{ _device }
    , physDevice{ _physDevice }
{
    if( !LoadDllFunctions( Utils::FindBinFolder() ) )
    {
        debug::Error( "XeSS: Failed to initialize DLL-s. XeSS will not be available." );
        m_context = nullptr;
        return;
    }

    xess_result_t r = pfn.xessVKCreateContext( instance, physDevice, device, &m_context );
    if( r != XESS_RESULT_SUCCESS )
    {
        debug::Error( "XeSS: xessVKCreateContext failed, xess_result_t={}", static_cast< int >( r ) );
        m_context = nullptr;
    }
}

RTGL1::XeSS_VK::~XeSS_VK()
{
    Destroy();
}

void RTGL1::XeSS_VK::Destroy()
{
    if( m_context )
    {
        vkDeviceWaitIdle( device );

        if( pfn.xessDestroyContext )
        {
            xess_result_t r = pfn.xessDestroyContext( m_context );
            assert( r == XESS_RESULT_SUCCESS );
        }
        m_context = nullptr;
    }
}

bool RTGL1::XeSS_VK::Valid() const
{
    return m_context != nullptr;
}

auto RTGL1::XeSS_VK::MakeInstance( VkInstance       instance,
                                   VkDevice         device,
                                   VkPhysicalDevice physDevice )
    -> std::shared_ptr< XeSS_VK >
{
    auto inst = std::make_shared< XeSS_VK >( instance, device, physDevice );
    if( !inst || !inst->Valid() )
    {
        return {};
    }
    return inst;
}

void RTGL1::XeSS_VK::OnFramebuffersSizeChange( const ResolutionState& resolutionState )
{
    if( !m_context )
    {
        return;
    }

    m_prevResolution = resolutionState;

    xess_vk_init_params_t initParams = {
        .outputResolution = { resolutionState.upscaledWidth, resolutionState.upscaledHeight },
        .qualitySetting   = ToXessQuality( m_currentMode ),
        .initFlags        = XESS_INIT_FLAG_NONE,
        .creationNodeMask = 1,
        .visibleNodeMask  = 1,
        .tempBufferHeap   = VK_NULL_HANDLE,
        .bufferHeapOffset = 0,
        .tempTextureHeap  = VK_NULL_HANDLE,
        .textureHeapOffset = 0,
        .pipelineCache    = VK_NULL_HANDLE,
    };

    // XeSS expects LDR color with auto exposure, motion vectors in render resolution.
    initParams.initFlags = XESS_INIT_FLAG_LDR_INPUT_COLOR |
                           XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE |
                           XESS_INIT_FLAG_JITTERED_MV;

    xess_result_t r = pfn.xessVKBuildPipelines( m_context, VK_NULL_HANDLE, true, initParams.initFlags );
    CheckError( r );

    r = pfn.xessVKInit( m_context, &initParams );
    CheckError( r );

    // Default velocity scale matches RTGL motion vector format.
    r = pfn.xessSetVelocityScale( m_context,
                                  float( resolutionState.renderWidth ),
                                  float( resolutionState.renderHeight ) );
    if( r != XESS_RESULT_SUCCESS )
    {
        debug::Warning( "XeSS: xessSetVelocityScale failed, xess_result_t={}", static_cast< int >( r ) );
    }

    r = pfn.xessSetLoggingCallback( m_context, XESS_LOGGING_LEVEL_DEBUG, PrintXeSSLog );
    if( r != XESS_RESULT_SUCCESS )
    {
        debug::Warning( "XeSS: xessSetLoggingCallback failed, xess_result_t={}", static_cast< int >( r ) );
    }
}

auto RTGL1::XeSS_VK::GetOptimalSettings( uint32_t               userWidth,
                                         uint32_t               userHeight,
                                         RgRenderResolutionMode mode ) const
    -> std::pair< uint32_t, uint32_t >
{
    if( !m_context )
    {
        assert( 0 );
        return { userWidth, userHeight };
    }

    xess_2d_t outputResolution = { userWidth, userHeight };
    xess_2d_t inputResolution  = {};

    m_currentMode = mode;

    xess_result_t r =
        pfn.xessGetInputResolution( m_context, &outputResolution, ToXessQuality( mode ), &inputResolution );
    if( r != XESS_RESULT_SUCCESS )
    {
        debug::Warning( "XeSS: xessGetInputResolution failed, xess_result_t={}", static_cast< int >( r ) );
        assert( 0 );
        return { userWidth, userHeight };
    }

    return { inputResolution.x, inputResolution.y };
}

RTGL1::FramebufferImageIndex RTGL1::XeSS_VK::Apply( VkCommandBuffer               cmd,
                                                    uint32_t                      frameIndex,
                                                    const Framebuffers&           framebuffers,
                                                    const RenderResolutionHelper& renderResolution,
                                                    RgFloat2D                     jitterOffset,
                                                    double                        timeDelta,
                                                    bool                          resetAccumulation )
{
    using FI = FramebufferImageIndex;

    if( !m_context )
    {
        debug::Error( "XeSS: context is null" );
        return FB_IMAGE_INDEX_FINAL;
    }

    const auto& resolutionState = renderResolution.GetResolutionState();

    if( m_prevResolution.renderWidth != resolutionState.renderWidth ||
        m_prevResolution.renderHeight != resolutionState.renderHeight ||
        m_prevResolution.upscaledWidth != resolutionState.upscaledWidth ||
        m_prevResolution.upscaledHeight != resolutionState.upscaledHeight )
    {
        OnFramebuffersSizeChange( resolutionState );
    }

    FI rs[] = { FI::FB_IMAGE_INDEX_FINAL,
                FI::FB_IMAGE_INDEX_DEPTH_NDC,
                FI::FB_IMAGE_INDEX_MOTION_DLSS,
                FI::FB_IMAGE_INDEX_REACTIVITY,
                OUTPUT_IMAGE_INDEX };
    InsertBarriers( cmd, frameIndex, framebuffers, rs, false );

    xess_vk_execute_params_t execParams = {
        .colorTexture     = ToXeSSImageInfo( FI::FB_IMAGE_INDEX_FINAL, frameIndex, framebuffers, resolutionState ),
        .velocityTexture  = ToXeSSImageInfo( FI::FB_IMAGE_INDEX_MOTION_DLSS, frameIndex, framebuffers, resolutionState ),
        .depthTexture     = ToXeSSImageInfo( FI::FB_IMAGE_INDEX_DEPTH_NDC, frameIndex, framebuffers, resolutionState ),
        .exposureScaleTexture     = {},
        .responsivePixelMaskTexture = ToXeSSImageInfo( FI::FB_IMAGE_INDEX_REACTIVITY, frameIndex, framebuffers, resolutionState ),
        .outputTexture    = ToXeSSImageInfo( OUTPUT_IMAGE_INDEX, frameIndex, framebuffers, resolutionState ),
        .jitterOffsetX    = jitterOffset.data[ 0 ],
        .jitterOffsetY    = jitterOffset.data[ 1 ],
        .exposureScale    = 1.0f,
        .resetHistory     = resetAccumulation ? 1u : 0u,
        .inputWidth       = resolutionState.renderWidth,
        .inputHeight      = resolutionState.renderHeight,
        .inputColorBase   = { 0, 0 },
        .inputMotionVectorBase = { 0, 0 },
        .inputDepthBase   = { 0, 0 },
        .inputResponsiveMaskBase = { 0, 0 },
        .reserved0        = { 0, 0 },
        .outputColorBase  = { 0, 0 },
    };

    xess_result_t r = pfn.xessVKExecute( m_context, cmd, &execParams );
    if( r != XESS_RESULT_SUCCESS )
    {
        debug::Error( "XeSS: xessVKExecute failed, xess_result_t={}", static_cast< int >( r ) );
        InsertBarriers( cmd, frameIndex, framebuffers, rs, true );
        return FB_IMAGE_INDEX_FINAL;
    }

    InsertBarriers( cmd, frameIndex, framebuffers, rs, true );

    return OUTPUT_IMAGE_INDEX;
}

RgFloat2D RTGL1::XeSS_VK::GetJitter( const ResolutionState& resolutionState,
                                     uint32_t               frameId ) const
{
    return HaltonSequence::GetJitter_Halton23( frameId );
}

auto RTGL1::XeSS_VK::RequiredVulkanExtensions_Instance()
    -> std::optional< std::vector< const char* > >
{
    if( !LoadDllFunctions( Utils::FindBinFolder() ) )
    {
        return std::nullopt;
    }

    auto supported = std::vector< VkExtensionProperties >{};
    {
        VkResult r{};
        uint32_t count = 0;

        r = vkEnumerateInstanceExtensionProperties( nullptr, &count, nullptr );
        if( r != VK_SUCCESS )
        {
            return std::nullopt;
        }

        supported.resize( count );
        r = vkEnumerateInstanceExtensionProperties( nullptr, &count, supported.data() );
        if( r != VK_SUCCESS )
        {
            return std::nullopt;
        }
    }

    uint32_t           extCount = 0;
    const char* const* ppExts   = nullptr;
    uint32_t           minVkApiVersion = 0;

    xess_result_t xr = pfn.xessVKGetRequiredInstanceExtensions( &extCount, &ppExts, &minVkApiVersion );
    if( xr != XESS_RESULT_SUCCESS )
    {
        debug::Warning( "XeSS: xessVKGetRequiredInstanceExtensions failed, xess_result_t={}", static_cast< int >( xr ) );
        return std::nullopt;
    }

    auto required = std::vector< const char* >{ ppExts, ppExts + extCount };

    for( const char* r : required )
    {
        bool isSupported =
            std::ranges::any_of( supported, [ r ]( const VkExtensionProperties& s ) {
                return strncmp( r, s.extensionName, std::size( s.extensionName ) ) == 0;
            } );

        if( !isSupported )
        {
            debug::Warning(
                "XeSS: Requires Vulkan instance extension {}, but the system doesn't support it",
                r );
            return std::nullopt;
        }
    }

    return required;
}

auto RTGL1::XeSS_VK::RequiredVulkanExtensions_Device( VkPhysicalDevice physDevice )
    -> std::optional< std::vector< const char* > >
{
    if( !LoadDllFunctions( Utils::FindBinFolder() ) )
    {
        return std::nullopt;
    }

    auto supported = std::vector< VkExtensionProperties >{};
    {
        VkResult r{};
        uint32_t count = 0;

        r = vkEnumerateDeviceExtensionProperties( physDevice, nullptr, &count, nullptr );
        if( r != VK_SUCCESS )
        {
            return std::nullopt;
        }

        supported.resize( count );
        r = vkEnumerateDeviceExtensionProperties( physDevice, nullptr, &count, supported.data() );
        if( r != VK_SUCCESS )
        {
            return std::nullopt;
        }
    }

    uint32_t           extCount = 0;
    const char* const* ppExts   = nullptr;

    xess_result_t xr = pfn.xessVKGetRequiredDeviceExtensions(
        VK_NULL_HANDLE, physDevice, &extCount, &ppExts );
    if( xr != XESS_RESULT_SUCCESS )
    {
        debug::Warning( "XeSS: xessVKGetRequiredDeviceExtensions failed, xess_result_t={}", static_cast< int >( xr ) );
        return std::nullopt;
    }

    auto required = std::vector< const char* >{ ppExts, ppExts + extCount };

    for( const char* r : required )
    {
        bool isSupported =
            std::ranges::any_of( supported, [ r ]( const VkExtensionProperties& s ) {
                return strncmp( r, s.extensionName, std::size( s.extensionName ) ) == 0;
            } );

        if( !isSupported )
        {
            debug::Warning(
                "XeSS: Requires Vulkan device extension {}, but the system doesn't support it", r );
            return std::nullopt;
        }
    }

    return required;
}
void* RTGL1::XeSS_VK::GetRequiredVulkanDeviceFeaturesChain( VkInstance       instance,
                                                            VkPhysicalDevice physDevice )
{
    if( !LoadDllFunctions( Utils::FindBinFolder() ) )
    {
        return nullptr;
    }

    void*         xessFeatureChain = nullptr;
    xess_result_t xr               = pfn.xessVKGetRequiredDeviceFeatures( instance, physDevice, &xessFeatureChain );
    if( xr != XESS_RESULT_SUCCESS )
    {
        debug::Warning( "XeSS: xessVKGetRequiredDeviceFeatures failed, xess_result_t={}", static_cast< int >( xr ) );
        return nullptr;
    }

    return xessFeatureChain;
}
#else

RTGL1::XeSS_VK::XeSS_VK( VkInstance, VkDevice _device, VkPhysicalDevice )
    : instance{}
    , device{ _device }
    , physDevice{}
{
}

RTGL1::XeSS_VK::~XeSS_VK() = default;

void RTGL1::XeSS_VK::OnFramebuffersSizeChange( const ResolutionState& ) {}

RTGL1::FramebufferImageIndex RTGL1::XeSS_VK::Apply( VkCommandBuffer,
                                                    uint32_t,
                                                    const Framebuffers&,
                                                    const RenderResolutionHelper&,
                                                    RgFloat2D,
                                                    double,
                                                    bool )
{
    return FB_IMAGE_INDEX_FINAL;
}

RgFloat2D RTGL1::XeSS_VK::GetJitter( const ResolutionState&, uint32_t ) const
{
    return {};
}

auto RTGL1::XeSS_VK::GetOptimalSettings( uint32_t userWidth,
                                         uint32_t,
                                         RgRenderResolutionMode ) const
    -> std::pair< uint32_t, uint32_t >
{
    return { userWidth, 0 };
}

auto RTGL1::XeSS_VK::RequiredVulkanExtensions_Instance()
    -> std::optional< std::vector< const char* > >
{
    return std::nullopt;
}


auto RTGL1::XeSS_VK::RequiredVulkanExtensions_Device( VkPhysicalDevice )
    -> std::optional< std::vector< const char* > >
{
    return std::nullopt;
}

bool RTGL1::XeSS_VK::Valid() const
{
    return false;
}

void RTGL1::XeSS_VK::Destroy() {}

#endif