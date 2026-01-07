# Liquid Audio Demo - Cross-Platform Setup

This directory contains setup scripts to fix compatibility issues with `liquid-audio-demo` across different platforms.

## Supported Platforms

- ✅ **macOS (Apple Silicon)** - Uses MPS (Metal Performance Shaders) for acceleration
- ✅ **macOS (Intel)** - Runs on CPU
- ✅ **Windows** - Uses CUDA if available, otherwise CPU
- ✅ **Linux** - Uses CUDA if available, otherwise CPU
- ✅ **Google Colab** - Uses CUDA for GPU acceleration

## Issues Fixed

The setup script automatically fixes these compatibility issues:

1. **Numpy version compatibility** - Installs numpy 2.3.5 (required by numba)
2. **Torch version** - Installs appropriate torch version for your platform
3. **Hardcoded CUDA references** - Fixes 4 files to auto-detect the best device
4. **Safetensors compatibility** - Loads models on CPU first (MPS doesn't support direct loading)

## Usage

### Interactive Setup (Recommended)

Run the Python script and select your platform when prompted:

```bash
python3 setup_liquid_audio.py
```

You'll see this menu:
```
🖥️  Select your platform:

  1. macOS (Apple Silicon)
  2. macOS (Intel)
  3. Windows
  4. Linux
  5. Google Colab
  6. Auto-detect

Enter choice (1-6):
```

The script will then:
- Install the correct versions of numpy and torch for your platform
- Modify the necessary files to use the appropriate device (MPS/CUDA/CPU)
- Apply platform-specific optimizations

### Auto-Detection

Select option **6** to automatically detect your platform. The script will detect:
- macOS vs Windows vs Linux
- Apple Silicon vs Intel on macOS
- Google Colab environment

## What Gets Modified

The script will modify these files in your Python site-packages:

1. `liquid_audio/processor.py` - Auto-detects device, loads safetensors on CPU first
2. `liquid_audio/model/lfm2_audio.py` - Auto-detects device
3. `liquid_audio/demo/model.py` - Uses detected device instead of hardcoded CUDA
4. `liquid_audio/demo/chat.py` - Uses detected device instead of hardcoded CUDA

## Running the Demo

After running the setup script:

```bash
liquid-audio-demo
```

### Platform-Specific Behavior

**macOS (Apple Silicon)**
- Uses MPS (Metal Performance Shaders) for hardware acceleration
- Falls back to CPU if needed

**macOS (Intel)**
- Runs on CPU
- Optimized for x86 architecture

**Windows/Linux**
- Uses CUDA if NVIDIA GPU is available
- Falls back to CPU otherwise

**Google Colab**
- Uses CUDA GPU runtime
- Free tier works well for inference

## Notes

- First run takes several minutes to download the 1.5B parameter model
- Subsequent runs are faster (model is cached)
- Requires ~3-4GB of RAM
- On Colab, make sure GPU runtime is enabled

## Troubleshooting

**"File not found" warnings**
- Make sure `liquid-audio-demo` is installed first
- Run `pip install liquid-audio-demo`

**Out of memory errors**
- Close other applications
- On 8GB RAM machines, you may need to use CPU-only mode

**Slow performance**
- Make sure you're using the correct platform setting
- Check if MPS/CUDA is being utilized (watch for GPU activity)

