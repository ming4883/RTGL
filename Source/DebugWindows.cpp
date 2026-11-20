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

#include "DebugWindows.h"

#ifdef RG_USE_IMGUI

#include <imgui.h>
#include <imgui_impl_glfw.h>

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <filesystem>
#include <string>

#include "RgException.h"

namespace
{

void glfwErrorCallback( int error, const char* description )
{
    throw RTGL1::RgException( RG_RESULT_GRAPHICS_API_ERROR,
                              "GLFW error (code " + std::to_string( error ) + "): " + description );
}

GLFWwindow* CreateGLFWWindow()
{
    glfwSetErrorCallback( glfwErrorCallback );

    if( !glfwInit() )
    {
        throw RTGL1::RgException( RG_RESULT_GRAPHICS_API_ERROR,
                                  "Failed to initialize GLFW for debug windows" );
    }

    glfwWindowHint( GLFW_CLIENT_API, GLFW_NO_API );
    glfwWindowHint( GLFW_RESIZABLE, GLFW_TRUE );
    return glfwCreateWindow( 700, 1000, "RTGL1 Dev", nullptr, nullptr );
}

bool IsSizeNull( GLFWwindow* wnd )
{
    int w, h;
    glfwGetWindowSize( wnd, &w, &h );
    return w == 0 || h == 0;
}

constexpr const char* FontsToFind[] = {
    R"(C:\Windows\Fonts\CascadiaMono.ttf)",
    R"(C:\Windows\Fonts\consola.ttf)",
};

} // anonymous

RTGL1::DebugWindows::DebugWindows( VkInstance                               /*_instance*/,
                                   VkPhysicalDevice                         /*_physDevice*/,
                                   VkDevice                                 /*_device*/,
                                   uint32_t                                 /*_queueFamiy*/,
                                   VkQueue                                  /*_queue*/,
                                   std::shared_ptr< CommandBufferManager >& /*_cmdManager*/ )
    : customWindow( CreateGLFWWindow() )
    , dx12( std::make_unique< DebugWindows_DX12 >() )
    , alwaysOnTop( false )
    , isMinimized( false )
{
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // Viewports stay OFF: dx12 backend would create extra viewports/surfaces.

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding    = 4.f;
    style.ChildRounding     = 4.f;
    style.PopupRounding     = 4.f;
    style.FrameRounding     = 4.f;
    style.ScrollbarRounding = 4.f;
    style.GrabRounding      = 4.f;
    style.TabRounding       = 4.f;

    for( const char* f : FontsToFind )
    {
        if( exists( std::filesystem::path{ f } ) )
        {
            io.Fonts->AddFontFromFileTTF( f, 15.f );
        }
    }

    // Platform backend only (GLFW supplies input + HWND). The renderer backend
    // is D3D12, initialized lazily by DebugWindows_DX12 once the shared D3D12
    // device exists. InitForOther must be used (NOT InitForVulkan).
    ImGui_ImplGlfw_InitForOther( customWindow, true );
}

RTGL1::DebugWindows::~DebugWindows()
{
    dx12.reset(); // ImGui_ImplDX12_Shutdown + destroy dev swapchain first

    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow( customWindow );
    glfwTerminate();
}

void RTGL1::DebugWindows::Init( const std::shared_ptr< DebugWindows >& /*self*/ )
{
    // No swapchain subscribers anymore (dev presentation is fully DXGI).
}

bool RTGL1::DebugWindows::PrepareForFrame( uint32_t /*frameIndex*/, bool /*vsync*/ )
{
    if( glfwWindowShouldClose( customWindow ) )
    {
        return false;
    }

    glfwPollEvents();

    isMinimized = IsSizeNull( customWindow );
    if( isMinimized )
    {
        return true;
    }

    // Lazily initialize the DXGI dev swapchain. Needs the shared D3D12 device
    // (created by the main Swapchain on its first AcquireImage), which is not
    // available yet when this DebugWindows is constructed.
    int fbW = 0, fbH = 0;
    glfwGetFramebufferSize( customWindow, &fbW, &fbH );
    if( !dx12->IsReady() )
    {
        HWND hwnd = glfwGetWin32Window( customWindow );
        dx12->TryInit( hwnd, uint32_t( fbW ), uint32_t( fbH ) );
    }

    // Always start an ImGui frame (platform + core). The D3D12 renderer backend
    // NewFrame is only valid once TryInit succeeded. Running the core NewFrame
    // unconditionally keeps Dev_Draw() (which submits widgets every frame) safe
    // even on the first frames before the D3D12 device exists; SubmitForFrame
    // pairs it with ImGui::EndFrame() in that case (see below).
    if( dx12->IsReady() )
    {
        dx12->NewFrame();
    }
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    return true;
}

void RTGL1::DebugWindows::SubmitForFrame( VkCommandBuffer /*cmd*/, uint32_t /*frameIndex*/ )
{
    if( isMinimized )
    {
        return; // PrepareForFrame did not start an ImGui frame when minimized
    }

    if( !dx12->IsReady() )
    {
        // A core ImGui frame was started in PrepareForFrame; close it without
        // rendering so NewFrame/Render stay balanced while D3D12 is not ready.
        ImGui::EndFrame();
        return;
    }

    ImGui::Render();

    int fbW = 0, fbH = 0;
    glfwGetFramebufferSize( customWindow, &fbW, &fbH );
    dx12->RenderAndPresent( uint32_t( fbW ), uint32_t( fbH ) );
}

void RTGL1::DebugWindows::SetAlwaysOnTop( bool onTop )
{
    if( alwaysOnTop != onTop )
    {
        alwaysOnTop = onTop;
        glfwSetWindowAttrib( customWindow, GLFW_FLOATING, alwaysOnTop ? GLFW_TRUE : GLFW_FALSE );
    }
}

#endif // RG_USE_IMGUI
