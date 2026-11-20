#pragma once

// DXGI presentation backend for the "RTGL1 Dev" ImGui window (2026-11-20, plan B).
//
// The dev window used to be a separate GLFW window with its OWN native-Vulkan
// VkSwapchainKHR, presented via vkQueuePresentKHR. In a Remote Desktop session
// only content that goes through the DXGI/DWM present chain is visible, so the
// dev panel was invisible. This backend renders ImGui with the D3D12 backend
// into a dedicated IDXGISwapChain on the dev window's HWND and calls Present()
// on it, reusing the SAME D3D12 device that dxgi:: (DX12_Swapchain.cpp) already
// created for the main window (a fresh DXGI factory is used, so the main
// window's swapchain, backbuffers, copy path and dxgi::Present are untouched).
//
// The D3D12 instance is created lazily by Swapchain::Create (on the first
// BeginFrame), but DebugWindows is constructed during VulkanDevice_Init, before
// that. So this backend is initialized lazily too: TryInit() is called each
// frame and only builds the dev swapchain once dxgi::HasDX12Instance() is true.

#include "Common.h"
#include <cstdint>

struct ID3D12Device;
struct ID3D12CommandQueue;
struct ID3D12CommandAllocator;
struct ID3D12GraphicsCommandList;
struct ID3D12Resource;
struct ID3D12DescriptorHeap;
struct ID3D12Fence;
struct IDXGISwapChain4;
using HANDLE = void*;

namespace RTGL1
{

class DebugWindows_DX12
{
public:
    DebugWindows_DX12() = default;
    ~DebugWindows_DX12();

    DebugWindows_DX12( const DebugWindows_DX12& )                = delete;
    DebugWindows_DX12( DebugWindows_DX12&& ) noexcept            = delete;
    DebugWindows_DX12& operator=( const DebugWindows_DX12& )     = delete;
    DebugWindows_DX12& operator=( DebugWindows_DX12&& ) noexcept = delete;

    // One-shot lazy init. Returns true once the dev DXGI swapchain + ImGui D3D12
    // renderer backend are ready. Requires dxgi::HasDX12Instance().
    bool TryInit( void* hwnd, uint32_t width, uint32_t height );
    bool IsReady() const { return m_ready; }

    // Called before ImGui::NewFrame() each frame (renderer-backend NewFrame).
    void NewFrame();

    // Records ImGui::GetDrawData() into the dev swapchain backbuffer and
    // presents on the D3D12 queue. ImGui::Render() must have been called by the
    // caller. w/h are the current GLFW framebuffer size (for resize handling).
    void RenderAndPresent( uint32_t width, uint32_t height );

private:
    void Shutdown();
    bool CreateResources( void* hwnd, uint32_t width, uint32_t height );
    bool Resize( uint32_t width, uint32_t height );
    void ReleaseBackBuffers();
    void WaitForLastSubmittedFrame();

private:
    bool m_ready{ false };

#ifdef RG_USE_DX12
    ID3D12Device*              m_device{ nullptr };       // borrowed from dxgi::
    ID3D12CommandQueue*        m_queue{ nullptr };        // owned (dev-only)
    ID3D12CommandAllocator*    m_cmdAllocator{ nullptr }; // owned
    ID3D12GraphicsCommandList* m_cmdList{ nullptr };      // owned
    IDXGISwapChain4*           m_swapchain{ nullptr };    // owned (dev-only)
    ID3D12DescriptorHeap*      m_rtvHeap{ nullptr };      // owned
    ID3D12DescriptorHeap*      m_srvHeap{ nullptr };      // owned (imgui font)
    ID3D12Fence*               m_fence{ nullptr };        // owned
    HANDLE                     m_fenceEvent{ nullptr };   // owned
    ID3D12Resource**           m_backBuffers{ nullptr };  // owned array of swapchain refs
    uint64_t                   m_fenceValue{ 0 };
    uint32_t                   m_rtvDescriptorSize{ 0 };
    uint32_t                   m_width{ 0 };
    uint32_t                   m_height{ 0 };
    static constexpr uint32_t  BackBufferCount = 2;
#endif
};

}
