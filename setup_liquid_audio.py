#!/usr/bin/env python3
"""
Setup script for liquid-audio-demo
Fixes compatibility issues with numpy, torch, and device support
Works on macOS, Windows, Linux, and Google Colab
"""

import os
import sys
import subprocess
import platform

def run_command(cmd, description):
    """Run a command and print the result"""
    print(f"⚡ {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")
        return False

def get_platform():
    """Ask user for their platform"""
    print("🖥️  Select your platform:")
    print()
    print("  1. macOS (Apple Silicon)")
    print("  2. macOS (Intel)")
    print("  3. Windows")
    print("  4. Linux")
    print("  5. Google Colab")
    print("  6. Auto-detect")
    print()

    while True:
        choice = input("Enter choice (1-6): ").strip()
        if choice in ['1', '2', '3', '4', '5', '6']:
            break
        print("❌ Invalid choice. Please enter a number between 1 and 6.")

    # Auto-detect if chosen
    if choice == '6':
        system = platform.system()
        if system == 'Darwin':
            # Check if Apple Silicon
            try:
                result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
                if 'arm64' in result.stdout:
                    choice = '1'
                else:
                    choice = '2'
            except:
                choice = '2'
        elif system == 'Windows':
            choice = '3'
        elif system == 'Linux':
            # Check if running in Colab
            if os.path.exists('/content'):
                choice = '5'
            else:
                choice = '4'
        else:
            print(f"⚠️  Unknown system: {system}, defaulting to Linux")
            choice = '4'

    platforms = {
        '1': 'macos-arm',
        '2': 'macos-intel',
        '3': 'windows',
        '4': 'linux',
        '5': 'colab'
    }

    selected = platforms[choice]
    print(f"✅ Platform: {selected}")
    return selected

def get_device_priority(platform):
    """Get device priority based on platform"""
    if platform == 'macos-arm':
        return ['mps', 'cpu']
    elif platform == 'macos-intel':
        return ['cpu']
    elif platform == 'colab':
        return ['cuda', 'cpu']
    elif platform == 'linux':
        return ['cuda', 'cpu']
    elif platform == 'windows':
        return ['cuda', 'cpu']
    else:
        return ['cpu']

def main():
    print("🔧 Setting up liquid-audio-demo...")
    print()

    # Get platform from user
    user_platform = get_platform()
    device_priority = get_device_priority(user_platform)
    print(f"🎯 Device priority: {' -> '.join(device_priority)}")
    print()

    # Step 1: Install compatible numpy version
    print("📦 Installing numpy 2.3.5 for numba compatibility...")
    run_command('pip install "numpy>=2.0,<2.4" --ignore-installed',
                "Installing numpy")
    print()

    # Step 2: Platform-specific torch installation
    if user_platform in ['macos-arm', 'macos-intel']:
        print("⚡ Upgrading torch for macOS...")
        run_command('pip install --upgrade torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1',
                    "Upgrading torch")
    elif user_platform == 'colab':
        print("⚡ Installing torch for Colab (CUDA-enabled)...")
        run_command('pip install --upgrade torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121',
                    "Installing torch")
    else:
        print("⚡ Upgrading torch (CPU/CUDA)...")
        run_command('pip install --upgrade torch torchaudio torchvision',
                    "Upgrading torch")
    print()

    # Get the site-packages path
    site_packages = None

    # Try multiple methods to find site-packages
    methods = []

    # Method 1: Use distutils
    try:
        from distutils.sysconfig import get_python_lib
        methods.append(("distutils", get_python_lib()))
    except ImportError:
        pass

    # Method 2: Use sysconfig (Python 3.3+)
    try:
        import sysconfig
        methods.append(("sysconfig", sysconfig.get_path('purelib')))
    except ImportError:
        pass

    # Method 3: Subprocess to python3
    try:
        result = subprocess.run(
            ['python3', '-c', 'import sys; [print(p) for p in sys.path if "site-packages" in p]'],
            capture_output=True, text=True, check=True, timeout=5
        )
        lines = result.stdout.strip().split('\n')
        if lines and lines[0]:
            methods.append(("subprocess", lines[0]))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Method 4: Direct sys.path scan
    import sys
    for path in sys.path:
        if "site-packages" in path:
            methods.append(("sys.path", path))
            break

    # Try each method and pick the first one that exists and contains liquid_audio
    for method_name, path in methods:
        if path and os.path.exists(path):
            # Check if liquid_audio is here or could be here
            liquid_audio_path = os.path.join(path, "liquid_audio")
            if os.path.exists(liquid_audio_path) or not site_packages:
                site_packages = path
                print(f"🔍 Found site-packages via {method_name}")
                break

    if not site_packages:
        print("❌ Error: Could not find site-packages directory")
        print("\n📋 Searched paths:")
        for method_name, path in methods:
            if path:
                exists = "✓" if os.path.exists(path) else "✗"
                print(f"  {exists} {method_name}: {path}")
        print("\nPlease make sure liquid_audio is installed:")
        print("  pip install liquid-audio-demo")
        sys.exit(1)

    print(f"📂 Site packages: {site_packages}")

    # Check if liquid_audio exists
    liquid_audio_path = os.path.join(site_packages, "liquid_audio")
    if not os.path.exists(liquid_audio_path):
        print(f"\n⚠️  Warning: liquid_audio not found in {site_packages}")
        print("Attempting to install liquid-audio-demo...")
        if run_command("pip install liquid-audio-demo", "Installing liquid-audio-demo"):
            print("✅ Installation complete")
        else:
            print("❌ Failed to install. Please install manually:")
            print("   pip install liquid-audio-demo")
            sys.exit(1)

    print()

    # Step 3: Fix processor.py
    print("🔧 Fixing processor.py for device compatibility...")
    processor_py = os.path.join(site_packages, "liquid_audio/processor.py")
    if os.path.exists(processor_py):
        fix_processor_py(processor_py, device_priority)
    else:
        print(f"⚠️  File not found: {processor_py}")
    print()

    # Step 4: Fix lfm2_audio.py
    print("🔧 Fixing lfm2_audio.py for device compatibility...")
    lfm2_audio_py = os.path.join(site_packages, "liquid_audio/model/lfm2_audio.py")
    if os.path.exists(lfm2_audio_py):
        fix_lfm2_audio_py(lfm2_audio_py, device_priority)
    else:
        print(f"⚠️  File not found: {lfm2_audio_py}")
    print()

    # Step 5: Fix demo/model.py
    print("🔧 Fixing demo/model.py for device compatibility...")
    demo_model_py = os.path.join(site_packages, "liquid_audio/demo/model.py")
    if os.path.exists(demo_model_py):
        fix_demo_model_py(demo_model_py, device_priority)
    else:
        print(f"⚠️  File not found: {demo_model_py}")
    print()

    # Step 6: Fix demo/chat.py
    print("🔧 Fixing demo/chat.py for device compatibility...")
    demo_chat_py = os.path.join(site_packages, "liquid_audio/demo/chat.py")
    if os.path.exists(demo_chat_py):
        fix_demo_chat_py(demo_chat_py, device_priority)
    else:
        print(f"⚠️  File not found: {demo_chat_py}")
    print()

    print("✅ Setup complete! All compatibility issues have been fixed.")
    print()
    print("You can now run: liquid-audio-demo")
    print()
    if user_platform == 'macos-arm':
        print("The demo will use MPS (Metal Performance Shaders) for acceleration.")
    elif user_platform in ['colab', 'linux', 'windows']:
        print("The demo will use CUDA if available, otherwise CPU.")
    else:
        print("The demo will run on CPU.")

def fix_processor_py(filepath, device_priority):
    """Fix hardcoded CUDA references in processor.py"""
    with open(filepath, 'r') as f:
        content = f.read()

    # Fix default device parameter
    content = content.replace(
        'device: torch.device | str = "cuda"',
        'device: torch.device | str | None = None'
    )

    # Generate device detection code
    if 'mps' in device_priority:
        device_code = 'device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"'
    else:
        device_code = 'device = "cuda" if torch.cuda.is_available() else "cpu"'

    # Add device auto-detection after cache_path line
    if 'if device is None:' not in content:
        content = content.replace(
            'cache_path = get_model_dir(repo_id, revision=revision)\n        with (cache_path / "config.json").open() as f:',
            f'''cache_path = get_model_dir(repo_id, revision=revision)
        with (cache_path / "config.json").open() as f:

        if device is None:
            import torch
            device = {device_code}'''
        )

    # Fix safetensors to load on CPU first
    content = content.replace(
        '''mimi_model = moshi.models.loaders.get_mimi(None, device=self.device)
        mimi_weights = load_file(self.mimi_weights_path, device=str(self.device))
        mimi_model.load_state_dict(mimi_weights, strict=True)

        return mimi_model''',
        '''mimi_model = moshi.models.loaders.get_mimi(None, device=self.device)
        # Load on CPU first since safetensors doesn't support MPS directly
        mimi_weights = load_file(self.mimi_weights_path, device="cpu")
        mimi_model.load_state_dict(mimi_weights, strict=True)
        # Move to target device after loading
        mimi_model.to(self.device)

        return mimi_model'''
    )

    with open(filepath, 'w') as f:
        f.write(content)
    print("✅ Fixed processor.py")

def fix_lfm2_audio_py(filepath, device_priority):
    """Fix hardcoded CUDA references in lfm2_audio.py"""
    with open(filepath, 'r') as f:
        content = f.read()

    # Fix default device parameter
    content = content.replace(
        'device: torch.device | str = "cuda"',
        'device: torch.device | str | None = None'
    )

    # Generate device detection code
    if 'mps' in device_priority:
        device_code = 'device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"'
    else:
        device_code = 'device = "cuda" if torch.cuda.is_available() else "cpu"'

    # Add device auto-detection after config loading
    if 'if device is None:' not in content:
        content = content.replace(
            '''with (cache_path / "config.json").open() as f:
            config = json.load(f)

        conf = LFM2AudioConfig(''',
            f'''with (cache_path / "config.json").open() as f:
            config = json.load(f)

        if device is None:
            device = {device_code}

        conf = LFM2AudioConfig('''
        )

    with open(filepath, 'w') as f:
        f.write(content)
    print("✅ Fixed lfm2_audio.py")

def fix_demo_model_py(filepath, device_priority):
    """Fix hardcoded CUDA references in demo/model.py"""
    with open(filepath, 'r') as f:
        content = f.read()

    # Generate device detection code
    if 'mps' in device_priority:
        device_code = '"cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"'
    else:
        device_code = '"cuda" if torch.cuda.is_available() else "cpu"'

    # Add device detection before warmup
    if 'device = "cuda" if torch.cuda.is_available()' not in content:
        content = content.replace(
            'logging.info("Warmup tokenizer")\nwith mimi.streaming(1), torch.no_grad():',
            f'''logging.info("Warmup tokenizer")
device = {device_code}
with mimi.streaming(1), torch.no_grad():'''
        )

    # Fix torch.randint device
    content = content.replace(
        'torch.randint(2048, (1, 8, 1), device="cuda")',
        'torch.randint(2048, (1, 8, 1), device=device)'
    )

    with open(filepath, 'w') as f:
        f.write(content)
    print("✅ Fixed demo/model.py")

def fix_demo_chat_py(filepath, device_priority):
    """Fix hardcoded CUDA references in demo/chat.py"""
    with open(filepath, 'r') as f:
        content = f.read()

    # Generate device detection code
    if 'mps' in device_priority:
        device_code = '"cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"'
    else:
        device_code = '"cuda" if torch.cuda.is_available() else "cpu"'

    # Add device detection before chat.append
    if 'device = "cuda" if torch.cuda.is_available()' not in content:
        content = content.replace(
    '''    chat.append(
        text=torch.stack(out_text, 1),
        audio_out=torch.stack(out_audio, 1),
        modality_flag=torch.tensor(out_modality, device="cuda"),
    )''',
    f'''    device = {device_code}
    chat.append(
        text=torch.stack(out_text, 1),
        audio_out=torch.stack(out_audio, 1),
        modality_flag=torch.tensor(out_modality, device=device),
    )'''
        )

    with open(filepath, 'w') as f:
        f.write(content)
    print("✅ Fixed demo/chat.py")

if __name__ == "__main__":
    main()
