#pragma once

// Neural Radiance Cache: all GPU buffers, descriptor set 12, and MLP init.
// Created only when VK_NV_cooperative_matrix is available; otherwise inactive.

#include "Common.h"
#include "MemoryAllocator.h"
#include "AutoBuffer.h"
#include "CommandBufferManager.h"
#include "Framebuffers.h"
#include "GlobalUniform.h"
#include "IFramebuffersDependency.h"
#include "Generated/ShaderCommonC.h"

namespace RTGL1
{

inline constexpr uint32_t NRC_HIDDEN_LAYERS     = 5;
inline constexpr uint32_t NRC_WEIGHT_COUNT      = 64 * 64 * NRC_HIDDEN_LAYERS + 64 * 3;
inline constexpr uint32_t NRC_TRAIN_BATCH_COUNT = 4;
inline constexpr uint32_t NRC_TRAIN_BATCH_SIZE  = 16384;

// must match NRCFrameParams in NrcCommon.glsl; 48 bytes
struct NRCFrameParamsGpu
{
    float    uPosNormalizeScale[ 4 ];
    float    uPosNormalizeCenter[ 4 ];
    uint32_t uFrameId;
    uint32_t uWidth;
    uint32_t uHeight;
    float    uTrainProbability;
};

static_assert( sizeof( NRCFrameParamsGpu ) == 48 );

class NRCCache : public IFramebuffersDependency
{
public:
    NRCCache( VkDevice                           device,
              std::shared_ptr< MemoryAllocator > allocator,
              std::shared_ptr< Framebuffers >    framebuffers,
              std::shared_ptr< CommandBufferManager > cmdManager,
              GlobalUniform&                     uniform,
              bool                               rIsCoopmatSupported );

    ~NRCCache();

    NRCCache( const NRCCache& other )     = delete;
    NRCCache( NRCCache&& other ) noexcept = delete;
    NRCCache& operator=( const NRCCache& other ) = delete;
    NRCCache& operator=( NRCCache&& other ) noexcept = delete;

    void OnFramebuffersSizeChange( const ResolutionState& resolutionState ) override;

    VkDescriptorSetLayout GetDescSetLayout() const { return descLayout; }
    VkDescriptorSet       GetDescSet( uint32_t frameIndex ) const
    {
        return descSets[ frameIndex % MAX_FRAMES_IN_FLIGHT ];
    }

    bool IsActive() const { return active; }

    void InitMLP( CommandBufferManager& cmdManager );

    VkBuffer GetWeightsBuffer() const { return weightsBuffer; }
    VkBuffer GetOptimizerEntriesBuffer() const { return optimizerEntriesBuffer; }
    VkBuffer GetOptimizerStateBuffer() const { return optimizerStateBuffer; }
    VkBuffer GetGradientsBuffer() const { return gradientsBuffer; }
    VkBuffer GetEvalCountBuffer() const { return evalCountBuffer; }
    VkBuffer GetEvalRecordsBuffer() const { return evalRecordsBuffer; }
    VkBuffer GetBatchTrainCountsBuffer() const { return batchTrainCountsBuffer; }
    VkBuffer GetBatchTrainRecordsBuffer() const { return batchTrainRecordsBuffer; }
    VkBuffer GetDispatchCmdBuffer() const { return dispatchCmdBuffer; }

    // uploads per-frame normalization params into the uniform buffer (binding 11)
    void UpdateFrameParams( VkCommandBuffer       cmd,
                            uint32_t              frameIndex,
                            const float*          cameraPosition,
                            uint32_t              frameId,
                            uint32_t              width,
                            uint32_t              height );

    VkImage  GetBiasFactorRImage() const { return biasFactorRImage; }
    VkImage  GetFactorGBImage() const { return factorGBImage; }

private:
    void CreateBuffers();
    void CreateImages();
    void CreateDescLayout();
    void UpdateDescriptors();
    void DestroySizeDependent();

    VkDevice                           device;
    std::weak_ptr< MemoryAllocator >   allocator;
    std::weak_ptr< Framebuffers >      framebuffers;
    std::shared_ptr< CommandBufferManager > cmdManager;

    bool active = false;

    VkDescriptorPool      descPool   = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkDescriptorSet       descSets[ MAX_FRAMES_IN_FLIGHT ] = {};

    VkBuffer weightsBuffer          = VK_NULL_HANDLE;
    VkDeviceMemory weightsMemory          = VK_NULL_HANDLE;
    VkBuffer optimizerEntriesBuffer = VK_NULL_HANDLE;
    VkDeviceMemory optimizerEntriesMemory = VK_NULL_HANDLE;
    VkBuffer optimizerStateBuffer   = VK_NULL_HANDLE;
    VkDeviceMemory optimizerStateMemory   = VK_NULL_HANDLE;
    VkBuffer gradientsBuffer        = VK_NULL_HANDLE;
    VkDeviceMemory gradientsMemory        = VK_NULL_HANDLE;

    VkBuffer evalCountBuffer         = VK_NULL_HANDLE;
    VkDeviceMemory evalCountMemory         = VK_NULL_HANDLE;
    VkBuffer evalRecordsBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory evalRecordsMemory       = VK_NULL_HANDLE;
    VkBuffer batchTrainCountsBuffer  = VK_NULL_HANDLE;
    VkDeviceMemory batchTrainCountsMemory  = VK_NULL_HANDLE;
    VkBuffer batchTrainRecordsBuffer = VK_NULL_HANDLE;
    VkDeviceMemory batchTrainRecordsMemory = VK_NULL_HANDLE;
    VkBuffer dispatchCmdBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory dispatchCmdMemory       = VK_NULL_HANDLE;

    AutoBuffer paramsAutoBuffer;

    VkImage     biasFactorRImage = VK_NULL_HANDLE;
    VkDeviceMemory    biasFactorRMemory = VK_NULL_HANDLE;
    VkImageView biasFactorRView  = VK_NULL_HANDLE;
    VkImage     factorGBImage    = VK_NULL_HANDLE;
    VkDeviceMemory    factorGBMemory   = VK_NULL_HANDLE;
    VkImageView factorGBView     = VK_NULL_HANDLE;

    uint32_t width  = 0;
    uint32_t height = 0;
};

}
