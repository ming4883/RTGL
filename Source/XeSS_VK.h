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

#pragma once

#include "Framebuffers.h"
#include "ResolutionState.h"

#include <RTGL1/RTGL1.h>

#include <xess/xess_vk.h>

#include <memory>

namespace RTGL1
{
class RenderResolutionHelper;

class XeSS_VK : public IFramebuffersDependency
{
public:
    static auto MakeInstance( VkInstance       instance,
                              VkDevice         device,
                              VkPhysicalDevice physDevice ) -> std::shared_ptr< XeSS_VK >;

    XeSS_VK( VkInstance instance, VkDevice device, VkPhysicalDevice physDevice );
    ~XeSS_VK() override;

    XeSS_VK( const XeSS_VK& other )                = delete;
    XeSS_VK( XeSS_VK&& other ) noexcept            = delete;
    XeSS_VK& operator=( const XeSS_VK& other )     = delete;
    XeSS_VK& operator=( XeSS_VK&& other ) noexcept = delete;

    void OnFramebuffersSizeChange( const ResolutionState& resolutionState ) override;

    FramebufferImageIndex Apply( VkCommandBuffer               cmd,
                                 uint32_t                      frameIndex,
                                 const Framebuffers&           framebuffers,
                                 const RenderResolutionHelper& renderResolution,
                                 RgFloat2D                     jitterOffset,
                                 double                        timeDelta,
                                 bool                          resetAccumulation );

    RgFloat2D GetJitter( const ResolutionState& resolutionState, uint32_t frameId ) const;

    auto GetOptimalSettings( uint32_t               userWidth,
                             uint32_t               userHeight,
                             RgRenderResolutionMode mode ) const -> std::pair< uint32_t, uint32_t >;

    static auto RequiredVulkanExtensions_Instance()
        -> std::optional< std::vector< const char* > >;
    static auto RequiredVulkanExtensions_Device( VkPhysicalDevice physDevice )
        -> std::optional< std::vector< const char* > >;

    static void* GetRequiredVulkanDeviceFeaturesChain( VkInstance       instance,
                                                       VkPhysicalDevice physDevice );

private:
    bool Valid() const;
    void Destroy();

private:
    VkInstance       instance;
    VkDevice         device;
    VkPhysicalDevice physDevice;

    xess_context_handle_t  m_context{ nullptr };
    ResolutionState        m_prevResolution{};
    mutable RgRenderResolutionMode m_currentMode{ RG_RENDER_RESOLUTION_MODE_CUSTOM };
};
}
