// Copyright (c) 2022 Sultim Tsyrendashiev
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "Common.h"
#include "DebugWindows_DX12.h"

#include <memory>

struct GLFWwindow;

namespace RTGL1
{
class CommandBufferManager;


// ImGui developer window.
//
// Plan B (2026-11-20): the window is now presented through DXGI via
// DebugWindows_DX12 (D3D12 ImGui backend + a dedicated IDXGISwapChain on the
// dev HWND). Previously it used its own native-Vulkan VkSwapchainKHR, which is
// invisible in Remote Desktop sessions (only the DXGI/DWM present chain is
// captured there). The GLFW window is kept solely as the HWND + input source.
//
// The Vulkan handles passed to the constructor are kept for source
// compatibility with VulkanDevice_Init.cpp but are no longer used for
// presentation.
class DebugWindows final
{
public:
    DebugWindows( VkInstance                               instance,
                  VkPhysicalDevice                         physDevice,
                  VkDevice                                 device,
                  uint32_t                                 queueFamiy,
                  VkQueue                                  queue,
                  std::shared_ptr< CommandBufferManager >& cmdManager );
    ~DebugWindows();

    DebugWindows( const DebugWindows& other )                = delete;
    DebugWindows( DebugWindows&& other ) noexcept            = delete;
    DebugWindows& operator=( const DebugWindows& other )     = delete;
    DebugWindows& operator=( DebugWindows&& other ) noexcept = delete;

    // Kept for source compatibility; no-op now (no swapchain subscribers).
    void Init( const std::shared_ptr< DebugWindows >& self );

    bool PrepareForFrame( uint32_t frameIndex, bool vsync );
    // 'cmd' is unused (kept for signature compatibility). All ImGui rendering
    // happens on the D3D12 queue inside DebugWindows_DX12.
    void SubmitForFrame( VkCommandBuffer cmd, uint32_t frameIndex );

    void SetAlwaysOnTop( bool onTop );
    bool IsMinimized() const { return isMinimized; }

private:
    GLFWwindow*                       customWindow{ nullptr };
    std::unique_ptr< DebugWindows_DX12 > dx12;

    bool alwaysOnTop{ false };
    bool isMinimized{ false };
};

}
