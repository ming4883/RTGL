#include "DebugWindows_DX12.h"

#ifdef RG_USE_IMGUI

#include "DX12_Interop.h"
#include "DebugPrint.h"

#include <imgui.h>
#include <imgui_impl_dx12.h>

#ifdef RG_USE_DX12

#include <d3d12.h>
#include <dxgi1_5.h>
#include <windows.h>

#endif

namespace
{
#ifdef RG_USE_DX12
    template< class T >
    void SafeRelease( T*& p )
    {
        if( p )
        {
            p->Release();
            p = nullptr;
        }
    }
#endif
}

namespace RTGL1
{

DebugWindows_DX12::~DebugWindows_DX12()
{
    Shutdown();
}

void DebugWindows_DX12::Shutdown()
{
#ifdef RG_USE_DX12
    if( m_ready )
    {
        WaitForLastSubmittedFrame();
        ImGui_ImplDX12_Shutdown();
    }
    ReleaseBackBuffers();
    delete[] m_backBuffers;
    m_backBuffers = nullptr;
    SafeRelease( m_swapchain );
    SafeRelease( m_cmdList );
    SafeRelease( m_cmdAllocator );
    SafeRelease( m_rtvHeap );
    SafeRelease( m_srvHeap );
    SafeRelease( m_fence );
    if( m_fenceEvent )
    {
        CloseHandle( m_fenceEvent );
        m_fenceEvent = nullptr;
    }
    SafeRelease( m_queue );
    // m_device is borrowed from dxgi:: - never released here.
    m_device = nullptr;
#endif
    m_ready = false;
}

bool DebugWindows_DX12::TryInit( void* hwnd, uint32_t width, uint32_t height )
{
    if( m_ready )
    {
        return true;
    }
#ifdef RG_USE_DX12
    // The D3D12 instance is created lazily by the main Swapchain on the first
    // AcquireImage; until then there is no device to reuse.
    if( !dxgi::HasDX12Instance() )
    {
        return false;
    }
    ID3D12Device* device = dxgi::GetD3D12Device();
    if( !device || !hwnd || width == 0 || height == 0 )
    {
        return false;
    }
    m_device = device;

    if( !CreateResources( hwnd, width, height ) )
    {
        Shutdown();
        return false;
    }

    // ImGui D3D12 renderer backend. The SRV heap holds the font atlas texture.
    D3D12_CPU_DESCRIPTOR_HANDLE fontCpu = m_srvHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE fontGpu = m_srvHeap->GetGPUDescriptorHandleForHeapStart();
    if( !ImGui_ImplDX12_Init( m_device,
                              int( BackBufferCount ),
                              DXGI_FORMAT_R8G8B8A8_UNORM,
                              m_srvHeap,
                              fontCpu,
                              fontGpu ) )
    {
        debug::Error( "DebugWindows_DX12: ImGui_ImplDX12_Init failed" );
        Shutdown();
        return false;
    }

    m_ready = true;
    return true;
#else
    ( void )hwnd;
    ( void )width;
    ( void )height;
    return false;
#endif
}

#ifdef RG_USE_DX12
bool DebugWindows_DX12::CreateResources( void* hwnd, uint32_t width, uint32_t height )
{
    HRESULT hr = S_OK;
    m_width   = width;
    m_height  = height;

    // A dedicated DXGI factory for the dev window. The main window's factory /
    // swapchain (in dxgi::) are left completely untouched.
    IDXGIFactory4* factory = nullptr;
    hr = CreateDXGIFactory1( IID_PPV_ARGS( &factory ) );
    if( FAILED( hr ) || !factory )
    {
        debug::Error( "DebugWindows_DX12: CreateDXGIFactory1 failed: {:08x}", uint32_t( hr ) );
        return false;
    }

    // Command queue (DIRECT) on the borrowed device.
    {
        D3D12_COMMAND_QUEUE_DESC q = {};
        q.Type     = D3D12_COMMAND_LIST_TYPE_DIRECT;
        q.Flags    = D3D12_COMMAND_QUEUE_FLAG_NONE;
        q.NodeMask = 0;
        hr = m_device->CreateCommandQueue( &q, IID_PPV_ARGS( &m_queue ) );
        if( FAILED( hr ) )
        {
            debug::Error( "DebugWindows_DX12: CreateCommandQueue failed: {:08x}", uint32_t( hr ) );
            SafeRelease( factory );
            return false;
        }
    }

    // Swapchain on the dev window's HWND.
    {
        DXGI_SWAP_CHAIN_DESC1 sd = {};
        sd.Width       = width;
        sd.Height      = height;
        sd.Format      = DXGI_FORMAT_R8G8B8A8_UNORM;
        sd.SampleDesc  = { 1, 0 };
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount = BackBufferCount;
        sd.SwapEffect  = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        sd.Flags       = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;

        IDXGISwapChain1* sw1 = nullptr;
        hr = factory->CreateSwapChainForHwnd( m_queue,
                                              static_cast< HWND >( hwnd ),
                                              &sd,
                                              nullptr,
                                              nullptr,
                                              &sw1 );
        SafeRelease( factory );
        if( FAILED( hr ) || !sw1 )
        {
            debug::Error( "DebugWindows_DX12: CreateSwapChainForHwnd failed: {:08x}", uint32_t( hr ) );
            return false;
        }
        hr = sw1->QueryInterface( IID_PPV_ARGS( &m_swapchain ) );
        SafeRelease( sw1 );
        if( FAILED( hr ) || !m_swapchain )
        {
            debug::Error( "DebugWindows_DX12: IDXGISwapChain4 QI failed: {:08x}", uint32_t( hr ) );
            return false;
        }
        // ALT+ENTER is handled by GLFW, not DXGI.
        // (MakeWindowAssociation is per-factory and we already released ours;
        //  a fresh factory has no association, which is safe.)
    }

    // RTV heap for the backbuffers.
    {
        D3D12_DESCRIPTOR_HEAP_DESC d = {};
        d.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        d.NumDescriptors = BackBufferCount;
        d.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        d.NodeMask       = 0;
        hr = m_device->CreateDescriptorHeap( &d, IID_PPV_ARGS( &m_rtvHeap ) );
        if( FAILED( hr ) )
        {
            debug::Error( "DebugWindows_DX12: RTV heap failed: {:08x}", uint32_t( hr ) );
            return false;
        }
        m_rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize( D3D12_DESCRIPTOR_HEAP_TYPE_RTV );
    }

    // SRV heap: one shader-visible descriptor for the ImGui font atlas.
    {
        D3D12_DESCRIPTOR_HEAP_DESC d = {};
        d.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        d.NumDescriptors = 1;
        d.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        d.NodeMask       = 0;
        hr = m_device->CreateDescriptorHeap( &d, IID_PPV_ARGS( &m_srvHeap ) );
        if( FAILED( hr ) )
        {
            debug::Error( "DebugWindows_DX12: SRV heap failed: {:08x}", uint32_t( hr ) );
            return false;
        }
    }

    // Command allocator + list + fence.
    hr = m_device->CreateCommandAllocator( D3D12_COMMAND_LIST_TYPE_DIRECT,
                                           IID_PPV_ARGS( &m_cmdAllocator ) );
    if( FAILED( hr ) )
    {
        debug::Error( "DebugWindows_DX12: CreateCommandAllocator failed: {:08x}", uint32_t( hr ) );
        return false;
    }
    hr = m_device->CreateCommandList( 0,
                                      D3D12_COMMAND_LIST_TYPE_DIRECT,
                                      m_cmdAllocator,
                                      nullptr,
                                      IID_PPV_ARGS( &m_cmdList ) );
    if( FAILED( hr ) )
    {
        debug::Error( "DebugWindows_DX12: CreateCommandList failed: {:08x}", uint32_t( hr ) );
        return false;
    }
    m_cmdList->Close();

    hr = m_device->CreateFence( 0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS( &m_fence ) );
    if( FAILED( hr ) )
    {
        debug::Error( "DebugWindows_DX12: CreateFence failed: {:08x}", uint32_t( hr ) );
        return false;
    }
    m_fenceEvent = CreateEvent( nullptr, FALSE, FALSE, nullptr );
    if( !m_fenceEvent )
    {
        debug::Error( "DebugWindows_DX12: CreateEvent failed" );
        return false;
    }
    m_fenceValue = 0;

    // Backbuffer RTVs.
    m_backBuffers = new ID3D12Resource* [ BackBufferCount ]{};
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    for( uint32_t i = 0; i < BackBufferCount; i++ )
    {
        hr = m_swapchain->GetBuffer( i, IID_PPV_ARGS( &m_backBuffers[ i ] ) );
        if( FAILED( hr ) || !m_backBuffers[ i ] )
        {
            debug::Error( "DebugWindows_DX12: GetBuffer failed: {:08x}", uint32_t( hr ) );
            return false;
        }
        m_device->CreateRenderTargetView( m_backBuffers[ i ], nullptr, rtv );
        rtv.ptr += m_rtvDescriptorSize;
    }
    return true;
}

void DebugWindows_DX12::ReleaseBackBuffers()
{
    if( m_backBuffers )
    {
        for( uint32_t i = 0; i < BackBufferCount; i++ )
        {
            SafeRelease( m_backBuffers[ i ] );
        }
    }
}

void DebugWindows_DX12::WaitForLastSubmittedFrame()
{
    if( !m_fence || !m_fenceEvent || !m_queue )
    {
        return;
    }
    if( m_fenceValue == 0 )
    {
        return;
    }
    if( m_fence->GetCompletedValue() < m_fenceValue )
    {
        m_fence->SetEventOnCompletion( m_fenceValue, m_fenceEvent );
        WaitForSingleObject( m_fenceEvent, INFINITE );
    }
}

bool DebugWindows_DX12::Resize( uint32_t width, uint32_t height )
{
    if( width == 0 || height == 0 )
    {
        return false;
    }
    WaitForLastSubmittedFrame();
    ReleaseBackBuffers();

    DXGI_SWAP_CHAIN_DESC desc = {};
    m_swapchain->GetDesc( &desc );
    HRESULT hr = m_swapchain->ResizeBuffers( BackBufferCount,
                                             width,
                                             height,
                                             desc.BufferDesc.Format,
                                             desc.Flags );
    if( FAILED( hr ) )
    {
        debug::Error( "DebugWindows_DX12: ResizeBuffers failed: {:08x}", uint32_t( hr ) );
        return false;
    }
    m_width  = width;
    m_height = height;

    D3D12_CPU_DESCRIPTOR_HANDLE rtv = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    for( uint32_t i = 0; i < BackBufferCount; i++ )
    {
        hr = m_swapchain->GetBuffer( i, IID_PPV_ARGS( &m_backBuffers[ i ] ) );
        if( FAILED( hr ) || !m_backBuffers[ i ] )
        {
            return false;
        }
        m_device->CreateRenderTargetView( m_backBuffers[ i ], nullptr, rtv );
        rtv.ptr += m_rtvDescriptorSize;
    }
    return true;
}

void DebugWindows_DX12::NewFrame()
{
    if( m_ready )
    {
        ImGui_ImplDX12_NewFrame();
    }
}

void DebugWindows_DX12::RenderAndPresent( uint32_t width, uint32_t height )
{
    if( !m_ready )
    {
        return;
    }
    if( width == 0 || height == 0 )
    {
        return;
    }
    if( width != m_width || height != m_height )
    {
        if( !Resize( width, height ) )
        {
            return;
        }
    }

    WaitForLastSubmittedFrame();

    const uint32_t backIdx = m_swapchain->GetCurrentBackBufferIndex();

    m_cmdAllocator->Reset();
    m_cmdList->Reset( m_cmdAllocator, nullptr );

    D3D12_RESOURCE_BARRIER toRT = {};
    toRT.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    toRT.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    toRT.Transition.pResource   = m_backBuffers[ backIdx ];
    toRT.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    toRT.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    toRT.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
    m_cmdList->ResourceBarrier( 1, &toRT );

    D3D12_CPU_DESCRIPTOR_HANDLE rtv = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rtv.ptr += SIZE_T( backIdx ) * m_rtvDescriptorSize;

    const float clear[ 4 ] = { 0.0f, 0.0f, 0.0f, 1.0f };
    m_cmdList->ClearRenderTargetView( rtv, clear, 0, nullptr );
    m_cmdList->OMSetRenderTargets( 1, &rtv, FALSE, nullptr );
    ID3D12DescriptorHeap* heaps[] = { m_srvHeap };
    m_cmdList->SetDescriptorHeaps( 1, heaps );

    ImGui_ImplDX12_RenderDrawData( ImGui::GetDrawData(), m_cmdList );

    D3D12_RESOURCE_BARRIER toPresent = {};
    toPresent.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    toPresent.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    toPresent.Transition.pResource   = m_backBuffers[ backIdx ];
    toPresent.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    toPresent.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    toPresent.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
    m_cmdList->ResourceBarrier( 1, &toPresent );

    m_cmdList->Close();

    ID3D12CommandList* cmds[] = { m_cmdList };
    m_queue->ExecuteCommandLists( 1, cmds );

    m_swapchain->Present( 0, DXGI_PRESENT_ALLOW_TEARING );

    m_fenceValue++;
    m_queue->Signal( m_fence, m_fenceValue );
}

#else  // !RG_USE_DX12

bool DebugWindows_DX12::CreateResources( void*, uint32_t, uint32_t ) { return false; }
bool DebugWindows_DX12::Resize( uint32_t, uint32_t ) { return false; }
void DebugWindows_DX12::ReleaseBackBuffers() {}
void DebugWindows_DX12::WaitForLastSubmittedFrame() {}
void DebugWindows_DX12::NewFrame() {}
void DebugWindows_DX12::RenderAndPresent( uint32_t, uint32_t ) {}

#endif

}

#endif  // RG_USE_IMGUI
