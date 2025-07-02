#!/usr/bin/env python3
"""
Automatically launch the MatAnyone Gradio demo with environment checks and dependency installation.
"""

import sys
import os
import subprocess
import argparse
import shutil
import time

# --- Constants ---
VENV_DIR = ".venv"

# --- Bootstrapping Functions ---

def get_venv_python():
    """Returns the absolute path to the Python executable in the venv."""
    # Note: Use abspath to ensure comparisons are reliable.
    if sys.platform == "win32":
        return os.path.abspath(os.path.join(VENV_DIR, "Scripts", "python.exe"))
    else:
        return os.path.abspath(os.path.join(VENV_DIR, "bin", "python"))

def create_venv_if_needed():
    """Creates a virtual environment using Python 3.11 if it doesn't exist."""
    if os.path.isdir(VENV_DIR):
        print(f"✔ Virtual environment '{VENV_DIR}' already exists.")
        return

    print(f"🐍 Creating virtual environment '{VENV_DIR}' with Python 3.11...")
    python_executable = None

    # On Windows, the 'py' launcher is the most reliable way to select a Python version.
    if sys.platform == "win32":
        if shutil.which("py"):
            try:
                check_command = ["py", "-3.11", "--version"]
                print(f"🔍 Checking for Python 3.11 via the 'py' launcher...")
                subprocess.run(check_command, check=True, capture_output=True, text=True, encoding='utf-8')
                python_executable = ["py", "-3.11"]
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("⚠️  'py -3.11' not found or failed.", file=sys.stderr)
    
    # On Linux/macOS, or as a fallback on Windows, search for `python3.11`
    if not python_executable:
        if shutil.which("python3.11"):
            print("🔍 Found 'python3.11' in PATH.")
            python_executable = ["python3.11"]
    
    if not python_executable:
        print("\n❌ Error: Could not find Python 3.11.", file=sys.stderr)
        print("Please install Python 3.11 and ensure it is available in your system's PATH.", file=sys.stderr)
        print("(On Windows, it should be available via `py -3.11`; on Linux/macOS, as `python3.11`).", file=sys.stderr)
        sys.exit(1)

    # Create the venv using the determined python executable
    try:
        create_command = python_executable + ["-m", "venv", VENV_DIR]
        print(f"🚀 Creating venv with: `{' '.join(create_command)}`")
        subprocess.run(create_command, check=True, capture_output=True, text=True, encoding='utf-8')
        print("✅ Virtual environment created successfully.")
    except subprocess.CalledProcessError as e:
        print("❌ Error: Failed to create virtual environment even with a valid Python.", file=sys.stderr)
        print(f"Details: {e.stderr}", file=sys.stderr)
        sys.exit(1)

# --- Dependency and Environment Check Functions (to be run inside the venv) ---

def check_nvidia_gpu_early():
    """
    On Windows/Linux, checks for `nvidia-smi` to give an early warning if no GPU is found.
    This provides a quick hint before any lengthy installations begin.
    """
    if sys.platform not in ["win32", "linux"]:
        return

    if shutil.which("nvidia-smi"):
        print("ℹ️  `nvidia-smi` found. Proceeding with setup. Final CUDA check will occur after PyTorch installation.")
    else:
        print("\n" + "#" * 70, file=sys.stderr)
        print("⚠️  EARLY WARNING: `nvidia-smi` COMMAND NOT FOUND IN SYSTEM PATH ⚠️".center(70), file=sys.stderr)
        print("#" * 70, file=sys.stderr)
        print("\nThis strongly suggests that an NVIDIA GPU is not available or the drivers", file=sys.stderr)
        print("are not installed correctly. The script will continue, but is very likely", file=sys.stderr)
        print("to fall back to extremely slow CPU processing.", file=sys.stderr)
        print("\nPress Ctrl+C now to abort if you do not wish to proceed.", file=sys.stderr)
        print("#" * 70, file=sys.stderr)
        time.sleep(3) # Give user a moment to react

def ensure_python_version():
    """Checks if the current Python version is 3.11."""
    major, minor = sys.version_info[:2]
    if (major, minor) != (3, 11):
        print(f"❌ Error: MatAnyone requires Python 3.11, but the virtual environment is running {sys.version}", file=sys.stderr)
        print(f"Please delete the '{VENV_DIR}' directory and ensure Python 3.11 is available to the script.", file=sys.stderr)
        sys.exit(1)
    print("✔ Python version 3.11 check passed.")

def ensure_torch_with_cuda():
    """
    Checks for a valid PyTorch installation with CUDA support.
    If it's missing or incorrect, attempts to install the proper version.
    """
    try:
        import torch
        # On macOS, just check for torch. MPS is handled by the app.
        if sys.platform == "darwin":
            print("✔ PyTorch is installed.")
            return
        # On other systems, check for CUDA.
        if torch.cuda.is_available():
            print("✔ PyTorch with CUDA support is already installed.")
            return
        else:
            print("⚠️ PyTorch is installed but lacks CUDA support. Attempting to reinstall the correct version.")
    except ImportError:
        print(" PyTorch not found. Installing PyTorch...")

    # Base command
    command = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]

    # Add CUDA-specific index URL for Linux and Windows
    if sys.platform != "darwin":
        command.extend(["--index-url", "https://download.pytorch.org/whl/cu121"])

    print(f"🚀 Running installation: {' '.join(command)}")
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, encoding='utf-8'
        )
        print(result.stdout)
        print("✅ PyTorch with CUDA support installed successfully.")
    except subprocess.CalledProcessError as e:
        print("\n--- Installation Failed ---", file=sys.stderr)
        print(f"Error output:\n{e.stderr}", file=sys.stderr)
        sys.exit(
            "❌ Error: Failed to install PyTorch with CUDA support. "
            "Please visit https://pytorch.org/get-started/locally/ and install it manually."
        )

def install_requirements(demo_dir):
    """Installs dependencies from the requirements.txt file."""
    req_file = os.path.join(demo_dir, "requirements.txt")
    if os.path.isfile(req_file):
        print(f"🐍 Installing Gradio demo dependencies from {req_file}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file], check=True, capture_output=True, text=True, encoding='utf-8')
            print("✅ Other dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"\n--- Dependency Installation Failed ---\n{e.stderr}", file=sys.stderr)
            sys.exit("❌ Error: Failed to install dependencies from requirements.txt.")
    else:
        print(f"⚠️ No requirements.txt found at {req_file}, skipping dependency installation.")

def check_ffmpeg():
    """Checks if ffmpeg is available in the system's PATH."""
    if shutil.which("ffmpeg") is None:
        print("⚠️ Warning: ffmpeg is not found in PATH. The Gradio demo requires ffmpeg. Please install ffmpeg manually.")

def launch_demo(args):
    """Changes to the demo directory and launches the Gradio app."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    demo_dir = os.path.join(script_dir, "hugging_face")
    os.chdir(demo_dir)

    cmd = [sys.executable, "app.py"]
    # Let app.py decide the device, but pass args to it.
    # args.device = "cuda" 
    if args.port:
        cmd += ["--port", str(args.port)]
    # The --device argument is now passed through to app.py if specified.
    # If not, app.py has its own default logic.
    if args.device:
        cmd += ["--device", args.device]
    if args.sam_model_type:
        cmd += ["--sam_model_type", args.sam_model_type]
    if args.mask_save:
        cmd += ["--mask_save"]

    print(f"🎬 Launching Gradio demo with command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"❌ Error: Gradio demo exited with error: {e}")

# --- Main Entrypoint ---
def main():
    """Main function to orchestrate the setup and launch of the demo."""
    # --- BOOTSTRAPPING: Ensure we run inside the venv ---
    venv_python_path = get_venv_python()
    # Use os.path.normcase for case-insensitive path comparison on Windows.
    current_python_path = os.path.abspath(sys.executable)

    if os.path.normcase(current_python_path) != os.path.normcase(venv_python_path):
        print("🌍 Script is running outside the designated virtual environment.")
        create_venv_if_needed()
        print(f"🔄 Relaunching script with: {venv_python_path}")
        try:
            os.execv(venv_python_path, [venv_python_path] + sys.argv)
        except OSError as e:
            print(f"❌ Failed to relaunch script in virtual environment: {e}", file=sys.stderr)
            print("Please try activating the environment manually and running the script again.", file=sys.stderr)
            print(f"Manual activation: `.\\{VENV_DIR}\\Scripts\\activate`", file=sys.stderr)
            sys.exit(1)

    # --- MAIN LOGIC (now guaranteed to be running inside the venv) ---
    print("\n🚀 Starting MatAnyone Gradio demo setup (inside virtual environment)...")

    # Perform an early check for NVIDIA GPU to provide a fast failure warning
    check_nvidia_gpu_early()

    # Update pip to the latest version to avoid notices and ensure compatibility
    print("\n upgrading pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True, capture_output=True, text=True, encoding='utf-8')
        print("✔ pip is up to date.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Could not upgrade pip: {e.stderr}", file=sys.stderr)

    ensure_python_version()

    parser = argparse.ArgumentParser(description="Start MatAnyone Gradio demo")
    parser.add_argument("--port", type=int, help="Port number to launch the Gradio demo")
    parser.add_argument("--device", type=str, help="Device for inference (e.g., 'cuda', 'cpu', 'mps')")
    parser.add_argument("--sam_model_type", type=str, help="SAM model type (vit_h, vit_l, vit_b)")
    parser.add_argument("--mask_save", action="store_true", help="Whether to save intermediate masks")
    args = parser.parse_args()

    ensure_torch_with_cuda()
    try:
        import torch
        # Verification after installation
        if sys.platform != "darwin" and not torch.cuda.is_available():
            print("\n" + "#" * 60, file=sys.stderr)
            print("⚠️ WARNING: NVIDIA CUDA ACCELERATION IS NOT AVAILABLE! ⚠️".center(60), file=sys.stderr)
            print("#" * 60, file=sys.stderr)
            print("\nThe application will run on the CPU, which will be EXTREMELY SLOW.", file=sys.stderr)
            print("A single frame could take minutes to process instead of seconds.", file=sys.stderr)
            print("\nPossible reasons:", file=sys.stderr)
            print("  1. An NVIDIA GPU is not detected on your system.", file=sys.stderr)
            print("  2. The NVIDIA drivers are not installed correctly.", file=sys.stderr)
            print("  3. The installed PyTorch version does not match your CUDA drivers.", file=sys.stderr)
            print("\nIt is STRONGLY recommended to stop (Ctrl+C) and resolve this.", file=sys.stderr)
            print("#" * 60, file=sys.stderr)
            
            try:
                for i in range(5, 0, -1):
                    print(f"\rProceeding with CPU in {i}...", end="", file=sys.stderr)
                    time.sleep(1)
                print("\rProceeding with CPU now.      ")
            except KeyboardInterrupt:
                print("\n\n👋 Aborted by user. Exiting.")
                sys.exit(0)
        
        print("✔ PyTorch installation verified.")
    except ImportError:
        sys.exit("❌ Error: PyTorch could not be imported after installation. Setup failed.")

    demo_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hugging_face")
    install_requirements(demo_dir)
    check_ffmpeg()

    print("\n🎉 All checks passed. Launching the Gradio demo...")
    try:
        stop_key = "Ctrl+C"
        print(f"\n✅ Gradio App is running. To stop it, press {stop_key} in this terminal or close the window.")
        launch_demo(args)
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during demo execution: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 