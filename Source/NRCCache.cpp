// NRCCache.cpp - Neural Radiance Cache buffers, descriptors and MLP init.
// Adapted from VkNRC (AdamYuan/VkNRC, MIT license) for RTGL.

#include "NRCCache.h"

#include "CommandBufferManager.h"
#include "Generated/ShaderCommonC.h"
#include "RgException.h"
#include "Utils.h"

#include <cstring>
#include <random>
#include <vector>
#include <immintrin.h>

using namespace RTGL1;

namespace
{

constexpr float NRC_TRAIN_PROBABILITY_F = 0.25f;

VkBuffer MakeBuffer( VkDevice                 device,
                     MemoryAllocator&         alloc,
                     VkDeviceSize             size,
                     VkBufferUsageFlags       usage,
                     VkMemoryPropertyFlags    memFlags,
                     VkDeviceMemory&          outMemory,
                     const char*              name )
{
    auto info = VkBufferCreateInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    VkBuffer buffer = VK_NULL_HANDLE;
    VkResult r = vkCreateBuffer( device, &info, nullptr, &buffer );
    VK_CHECKERROR( r );
    SET_DEBUG_NAME( device, buffer, VK_OBJECT_TYPE_BUFFER, name );

    VkMemoryRequirements memReq{};
    vkGetBufferMemoryRequirements( device, buffer, &memReq );

    auto memoryTypeIndex = alloc.GetMemoryTypeIndex( memReq.memoryTypeBits, memFlags );
    if( !memoryTypeIndex )
    {
        throw RgException( RG_RESULT_INTERNAL_ERROR, "Can't find memory type for NRC buffer" );
    }

    outMemory = alloc.AllocDedicated( memReq, memFlags, MemoryAllocator::AllocType::DEFAULT, name );

    // AllocDedicated allocates exactly memReq.size bytes, so the buffer must
    // be bound at offset 0; binding at memReq.alignment would run past the end
    // of the allocation.
    r = vkBindBufferMemory( device, buffer, outMemory, 0 );
    VK_CHECKERROR( r );

    return buffer;
}

void DestroyBuffer( VkDevice device, MemoryAllocator& alloc, VkBuffer& buffer, VkDeviceMemory& memory )
{
    if( buffer != VK_NULL_HANDLE )
    {
        vkDestroyBuffer( device, buffer, nullptr );
        buffer = VK_NULL_HANDLE;
    }
    if( memory != VK_NULL_HANDLE )
    {
        MemoryAllocator::FreeDedicated( device, memory );
        memory = VK_NULL_HANDLE;
    }
}

VkImage MakeImage( VkDevice device, MemoryAllocator& alloc, uint32_t width, uint32_t height, VkFormat format,
                   VkImage& outImage, VkDeviceMemory& outMemory, VkImageView& outView, const char* name )
{
    auto info = VkImageCreateInfo{
        .sType     = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format    = format,
        .extent    = { width, height, 1 },
        .mipLevels   = 1,
        .arrayLayers = 1,
        .samples     = VK_SAMPLE_COUNT_1_BIT,
        .tiling      = VK_IMAGE_TILING_OPTIMAL,
        .usage       = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VkResult r = vkCreateImage( device, &info, nullptr, &outImage );
    VK_CHECKERROR( r );
    SET_DEBUG_NAME( device, outImage, VK_OBJECT_TYPE_IMAGE, name );

    VkMemoryRequirements memReq{};
    vkGetImageMemoryRequirements( device, outImage, &memReq );

    auto memoryTypeIndex = alloc.GetMemoryTypeIndex( memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );
    if( !memoryTypeIndex )
    {
        throw RgException( RG_RESULT_INTERNAL_ERROR, "Can't find memory type for NRC image" );
    }

    outMemory = alloc.AllocDedicated( memReq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                      MemoryAllocator::AllocType::DEFAULT, name );

    // see MakeBuffer: the dedicated allocation is exactly memReq.size, so bind at 0
    r = vkBindImageMemory( device, outImage, outMemory, 0 );
    VK_CHECKERROR( r );

    auto viewInfo = VkImageViewCreateInfo{
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image    = outImage,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format   = format,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
    };

    r = vkCreateImageView( device, &viewInfo, nullptr, &outView );
    VK_CHECKERROR( r );
    SET_DEBUG_NAME( device, outView, VK_OBJECT_TYPE_IMAGE_VIEW, name );

    return outImage;
}

}

NRCCache::NRCCache( VkDevice                           aDevice,
                    std::shared_ptr< MemoryAllocator > aAllocator,
                    std::shared_ptr< Framebuffers >    aFramebuffers,
                    std::shared_ptr< CommandBufferManager > aCmdManager,
                    GlobalUniform&                     aUniform,
                    bool                               rIsCoopmatSupported )
    : device( aDevice )
    , allocator( aAllocator )
    , framebuffers( aFramebuffers )
    , cmdManager( aCmdManager )
    , paramsAutoBuffer( aAllocator )
{
    active = rIsCoopmatSupported;

    // Set 12 is always part of the ray tracing pipeline layout, so the layout
    // must exist even when NRC is disabled on this device.
    CreateDescLayout();

    if( !active )
    {
        debug::Warning( "NRC is disabled on this device" );
        return;
    }

    CreateBuffers();

    paramsAutoBuffer.Create( sizeof( NRCFrameParamsGpu ),
                             VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                             "NRC frame params" );

    // one-time upload of the initial MLP weights
    InitMLP( *aCmdManager );
}

NRCCache::~NRCCache()
{
    // all handles are VK_NULL_HANDLE when NRC was disabled, so this is a no-op
    DestroySizeDependent();

    auto alloc = allocator.lock();

    if( alloc )
    {
        DestroyBuffer( device, *alloc, weightsBuffer, weightsMemory );
        DestroyBuffer( device, *alloc, optimizerEntriesBuffer, optimizerEntriesMemory );
        DestroyBuffer( device, *alloc, optimizerStateBuffer, optimizerStateMemory );
        DestroyBuffer( device, *alloc, gradientsBuffer, gradientsMemory );
    }

    // paramsAutoBuffer destroys itself

    if( descLayout != VK_NULL_HANDLE )
    {
        vkDestroyDescriptorSetLayout( device, descLayout, nullptr );
    }
    if( descPool != VK_NULL_HANDLE )
    {
        vkDestroyDescriptorPool( device, descPool, nullptr );
    }
}

void NRCCache::CreateBuffers()
{
    auto alloc = allocator.lock();

    weightsBuffer = MakeBuffer( device, *alloc, NRC_WEIGHT_COUNT * sizeof( uint16_t ),
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, weightsMemory, "NRC Weights" );

    optimizerEntriesBuffer = MakeBuffer( device, *alloc, NRC_WEIGHT_COUNT * 16,
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, optimizerEntriesMemory,
                                         "NRC Optimizer Entries" );

    optimizerStateBuffer = MakeBuffer( device, *alloc, sizeof( uint32_t ) + sizeof( float ) * 4,
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, optimizerStateMemory,
                                       "NRC Optimizer State" );

    gradientsBuffer = MakeBuffer( device, *alloc, NRC_WEIGHT_COUNT * sizeof( float ),
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, gradientsMemory, "NRC Gradients" );
}

void NRCCache::CreateImages()
{
    auto alloc = allocator.lock();

    biasFactorRImage = MakeImage( device, *alloc, width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
                                  biasFactorRImage, biasFactorRMemory, biasFactorRView, "NRC BiasFactorR" );
    factorGBImage = MakeImage( device, *alloc, width, height, VK_FORMAT_R32G32_SFLOAT,
                               factorGBImage, factorGBMemory, factorGBView, "NRC FactorGB" );
}

void NRCCache::CreateDescLayout()
{
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  8 },
    };

    auto poolInfo = VkDescriptorPoolCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = MAX_FRAMES_IN_FLIGHT,
        .poolSizeCount = std::size( poolSizes ),
        .pPoolSizes = poolSizes,
    };

    VkResult r = vkCreateDescriptorPool( device, &poolInfo, nullptr, &descPool );
    VK_CHECKERROR( r );

    VkDescriptorSetLayoutBinding bindings[ 12 ] = {};

    auto make = [ & ]( uint32_t binding, VkDescriptorType type ) {
        bindings[ binding ] = VkDescriptorSetLayoutBinding{
            .binding         = binding,
            .descriptorType  = type,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        };
    };

    make( 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER );  // eval count
    make( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER );  // eval records
    make( 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER );  // batch train counts
    make( 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER );  // batch train records
    make( 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER );  // weights
    make( 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER );  // gradients
    make( 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER );  // optimizer entries
    make( 7, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER );  // optimizer state
    make( 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER );  // dispatch cmd
    make( 9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );   // bias/factorR image
    make( 10, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );  // factorGB image
    make( 11, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ); // NRC frame params

    auto layoutInfo = VkDescriptorSetLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 12,
        .pBindings = bindings,
    };

    VkResult r2 = vkCreateDescriptorSetLayout( device, &layoutInfo, nullptr, &descLayout );
    VK_CHECKERROR( r2 );
}

void NRCCache::UpdateDescriptors()
{
    // This function runs on every framebuffer resize. Reset the pool first so
    // previously allocated sets are freed; without the reset,
    // vkAllocateDescriptorSets fails with OUT_OF_POOL_MEMORY (pool maxSets is
    // MAX_FRAMES_IN_FLIGHT). Safe because resize is always preceded by
    // vkDeviceWaitIdle in Framebuffers::PrepareForSize.
    if( descPool != VK_NULL_HANDLE )
    {
        vkResetDescriptorPool( device, descPool, 0 );
    }

    // Allocate one set at a time. descriptorSetCount must match the number of
    // elements pointed to by pSetLayouts: allocating MAX_FRAMES_IN_FLIGHT sets
    // with a pointer to a single layout made the driver read the member right
    // after descLayout as a second layout handle, corrupting the allocated
    // sets and crashing later inside vkCmdBindDescriptorSets.
    for( uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++ )
    {
        auto allocSets = VkDescriptorSetAllocateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &descLayout,
        };

        VkResult r = vkAllocateDescriptorSets( device, &allocSets, &descSets[ i ] );
        if( r != VK_SUCCESS )
        {
            throw RgException( RG_RESULT_INTERNAL_ERROR,
                               "Can't allocate NRC descriptor sets" );
        }
    }

    for( uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++ )
    {
        VkDescriptorBufferInfo infos[ 8 ] = {};

        infos[ 0 ] = { evalCountBuffer, 0, VK_WHOLE_SIZE };
        infos[ 1 ] = { evalRecordsBuffer, 0, VK_WHOLE_SIZE };
        infos[ 2 ] = { batchTrainCountsBuffer, 0, VK_WHOLE_SIZE };
        infos[ 3 ] = { batchTrainRecordsBuffer, 0, VK_WHOLE_SIZE };
        infos[ 4 ] = { weightsBuffer, 0, VK_WHOLE_SIZE };
        infos[ 5 ] = { gradientsBuffer, 0, VK_WHOLE_SIZE };
        infos[ 6 ] = { optimizerEntriesBuffer, 0, VK_WHOLE_SIZE };
        infos[ 7 ] = { optimizerStateBuffer, 0, VK_WHOLE_SIZE };
        // infos[8] = dispatchCmdBuffer (see below, appended)

        VkDescriptorBufferInfo dispatchCmdInfo = { dispatchCmdBuffer, 0, VK_WHOLE_SIZE };

        VkDescriptorImageInfo imageInfos[ 2 ] = {
            { VK_NULL_HANDLE, biasFactorRView, VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, factorGBView, VK_IMAGE_LAYOUT_GENERAL },
        };

        VkWriteDescriptorSet writes[ 12 ] = {};

        VkDescriptorBufferInfo paramsInfo = { paramsAutoBuffer.GetDeviceLocal(), 0, VK_WHOLE_SIZE };

        auto makeWrite = [ & ]( uint32_t binding, VkDescriptorType type, const void* p ) {
            writes[ binding ] = VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descSets[ i ],
                .dstBinding = binding,
                .descriptorCount = 1,
                .descriptorType = type,
                .pImageInfo = static_cast< const VkDescriptorImageInfo* >( p ),
                .pBufferInfo = static_cast< const VkDescriptorBufferInfo* >( p ),
            };
        };

        for( uint32_t b = 0; b < 8; b++ )
        {
            VkDescriptorType t = ( b == 7 ) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                            : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            makeWrite( b, t, &infos[ b ] );
        }

        makeWrite( 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &dispatchCmdInfo );

        writes[ 9 ] = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descSets[ i ],
            .dstBinding = 9,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &imageInfos[ 0 ],
        };

        writes[ 10 ] = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descSets[ i ],
            .dstBinding = 10,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &imageInfos[ 1 ],
        };

        writes[ 11 ] = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descSets[ i ],
            .dstBinding = 11,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &paramsInfo,
        };

        vkUpdateDescriptorSets( device, 12, writes, 0, nullptr );
    }
}

void NRCCache::DestroySizeDependent()
{
    auto alloc = allocator.lock();
    if( !alloc )
    {
        return;
    }

    DestroyBuffer( device, *alloc, evalCountBuffer, evalCountMemory );
    DestroyBuffer( device, *alloc, evalRecordsBuffer, evalRecordsMemory );
    DestroyBuffer( device, *alloc, batchTrainCountsBuffer, batchTrainCountsMemory );
    DestroyBuffer( device, *alloc, batchTrainRecordsBuffer, batchTrainRecordsMemory );
    DestroyBuffer( device, *alloc, dispatchCmdBuffer, dispatchCmdMemory );

    auto destroyImage = [ & ]( VkImage& image, VkDeviceMemory& memory, VkImageView& view ) {
        if( view != VK_NULL_HANDLE )
        {
            vkDestroyImageView( device, view, nullptr );
            view = VK_NULL_HANDLE;
        }
        if( image != VK_NULL_HANDLE )
        {
            vkDestroyImage( device, image, nullptr );
            image = VK_NULL_HANDLE;
        }
        if( memory != VK_NULL_HANDLE )
        {
            MemoryAllocator::FreeDedicated( device, memory );
            memory = VK_NULL_HANDLE;
        }
    };

    destroyImage( biasFactorRImage, biasFactorRMemory, biasFactorRView );
    destroyImage( factorGBImage, factorGBMemory, factorGBView );
}

void NRCCache::OnFramebuffersSizeChange( const ResolutionState& resolutionState )
{
    if( !active )
    {
        return;
    }

    const uint32_t w = resolutionState.renderWidth;
    const uint32_t h = resolutionState.renderHeight;

    DestroySizeDependent();

    width  = w;
    height = h;

    auto alloc = allocator.lock();

    evalCountBuffer = MakeBuffer( device, *alloc, sizeof( uint32_t ),
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, evalCountMemory, "NRC Eval Count" );

    evalRecordsBuffer = MakeBuffer( device, *alloc, w * h * 20,
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, evalRecordsMemory, "NRC Eval Records" );

    batchTrainCountsBuffer = MakeBuffer( device, *alloc, NRC_TRAIN_BATCH_COUNT * sizeof( uint32_t ),
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, batchTrainCountsMemory,
                                         "NRC Train Counts" );

    batchTrainRecordsBuffer = MakeBuffer( device, *alloc, NRC_TRAIN_BATCH_COUNT * NRC_TRAIN_BATCH_SIZE * 40,
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, batchTrainRecordsMemory,
                                          "NRC Train Records" );

    dispatchCmdBuffer = MakeBuffer( device, *alloc, 3 * sizeof( uint32_t ),
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dispatchCmdMemory,
                                    "NRC Dispatch Cmd" );

    CreateImages();

    // The storage images are created in UNDEFINED layout - transition both to
    // GENERAL before the first storage access (same pattern as
    // Framebuffers::CreateImages). Safe to submit and wait: resize always
    // happens after vkDeviceWaitIdle in Framebuffers::PrepareForSize.
    if( cmdManager )
    {
        VkCommandBuffer cmd = cmdManager->StartGraphicsCmd();

        Utils::BarrierImage( cmd,
                             biasFactorRImage,
                             0,
                             VK_ACCESS_SHADER_WRITE_BIT,
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL );
        Utils::BarrierImage( cmd,
                             factorGBImage,
                             0,
                             VK_ACCESS_SHADER_WRITE_BIT,
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL );

        cmdManager->Submit( cmd );
        cmdManager->WaitGraphicsIdle();
    }

    UpdateDescriptors();
}

void NRCCache::UpdateFrameParams( VkCommandBuffer       cmd,
                                  uint32_t              frameIndex,
                                  const float*          cameraPosition,
                                  uint32_t              frameId,
                                  uint32_t              width,
                                  uint32_t              height )
{
    if( !active )
    {
        return;
    }

    // normalization scale: map ~100m of world space into [-1, 1]
    NRCFrameParamsGpu params = {
        .uPosNormalizeScale  = { 0.01f, 0.01f, 0.01f, 0.0f },
        .uPosNormalizeCenter = { cameraPosition[ 0 ], cameraPosition[ 1 ], cameraPosition[ 2 ], 0.0f },
        .uFrameId            = frameId,
        .uWidth              = width,
        .uHeight             = height,
        .uTrainProbability   = NRC_TRAIN_PROBABILITY_F,
    };

    // per-frame upload through persistent per-frame staging buffers, recorded
    // into the frame's command buffer (same pattern as GlobalUniform::Upload)
    void* mapped = paramsAutoBuffer.GetMapped( frameIndex );
    memcpy( mapped, &params, sizeof( params ) );
    paramsAutoBuffer.CopyFromStaging( cmd, frameIndex, sizeof( params ) );
}

void NRCCache::InitMLP( CommandBufferManager& cmdManager )
{
    if( !active )
    {
        return;
    }

    // Kaiming init: N(0, sqrt(2/64)), stored as fp16; optimizer entries {0,0,w,w}; state {0,1,1,1,1}
    std::mt19937                          gen{ 42 };
    std::normal_distribution< float >     norm{ 0.0f, std::sqrt( 2.0f / 64.0f ) };

    std::vector< uint16_t >      weights( NRC_WEIGHT_COUNT );
    std::vector< uint8_t >       optEntries( NRC_WEIGHT_COUNT * 16 );
    std::vector< uint8_t >       optState( 20 );
    std::vector< uint8_t >       zerosG( NRC_WEIGHT_COUNT * 4, 0 );

    for( uint32_t i = 0; i < NRC_WEIGHT_COUNT; i++ )
    {
        float w = norm( gen );

        // float -> fp16 bits using hardware conversion
        __m128 v = _mm_set_ss( w );
        __m128i h = _mm_cvtps_ph( v, 0 );
        weights[ i ] = static_cast< uint16_t >( _mm_cvtsi128_si32( h ) & 0xFFFF );

        float* entry = reinterpret_cast< float* >( optEntries.data() + i * 16 );
        entry[ 0 ] = 0.0f; // m
        entry[ 1 ] = 0.0f; // v
        entry[ 2 ] = w;    // weight
        entry[ 3 ] = w;    // ema_weight
    }

    {
        uint32_t* state = reinterpret_cast< uint32_t* >( optState.data() );
        state[ 0 ] = 0;
        float* f = reinterpret_cast< float* >( optState.data() + 4 );
        f[ 0 ] = 1.0f;
        f[ 1 ] = 1.0f;
        f[ 2 ] = 1.0f;
        f[ 3 ] = 1.0f;
    }

    // upload via staging; MakeBuffer selects the memory type and binds at offset 0
    VkDeviceSize totalSize = weights.size() * 2 + optEntries.size() + optState.size() + zerosG.size();

    auto alloc = allocator.lock();

    VkBuffer      staging        = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    staging = MakeBuffer( device, *alloc, totalSize,
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          stagingMemory, "NRC MLP Staging" );

    VkResult r = VK_SUCCESS;

    {
        void* p = nullptr;
        r = vkMapMemory( device, stagingMemory, 0, totalSize, 0, &p );
        VK_CHECKERROR( r );

        VkDeviceSize offset = 0;
        memcpy( static_cast< uint8_t* >( p ) + offset, weights.data(), weights.size() * 2 );
        offset += weights.size() * 2;
        memcpy( static_cast< uint8_t* >( p ) + offset, optEntries.data(), optEntries.size() );
        offset += optEntries.size();
        memcpy( static_cast< uint8_t* >( p ) + offset, optState.data(), optState.size() );
        offset += optState.size();
        memcpy( static_cast< uint8_t* >( p ) + offset, zerosG.data(), zerosG.size() );

        vkUnmapMemory( device, stagingMemory );
    }

    VkCommandBuffer cmd = cmdManager.StartTransferCmd();

    VkBufferCopy copies[ 4 ] = {};

    VkDeviceSize off = 0;
    copies[ 0 ] = { off, 0, weights.size() * 2 };
    off += weights.size() * 2;
    copies[ 1 ] = { off, 0, optEntries.size() };
    off += optEntries.size();
    copies[ 2 ] = { off, 0, optState.size() };
    off += optState.size();
    copies[ 3 ] = { off, 0, zerosG.size() };

    vkCmdCopyBuffer( cmd, staging, weightsBuffer, 1, &copies[ 0 ] );
    vkCmdCopyBuffer( cmd, staging, optimizerEntriesBuffer, 1, &copies[ 1 ] );
    vkCmdCopyBuffer( cmd, staging, optimizerStateBuffer, 1, &copies[ 2 ] );
    vkCmdCopyBuffer( cmd, staging, gradientsBuffer, 1, &copies[ 3 ] );

    cmdManager.Submit( cmd );

    // wait for the copies to finish before freeing the staging resources
    cmdManager.WaitTransferIdle();

    DestroyBuffer( device, *alloc, staging, stagingMemory );
}
