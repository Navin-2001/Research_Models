import shutil
import subprocess
import sys
import os
from pathlib import Path


def find_ffmpeg():
    """
    Try multiple ways to locate ffmpeg executable
    Returns path string or None
    """
    # Method 1: Standard shutil.which (most reliable)
    path = shutil.which("ffmpeg")
    if path:
        return path

    # Method 2: Common macOS/Linux locations (fallback)
    common_locations = [
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",       # Homebrew on Apple Silicon
        "/usr/bin/ffmpeg",
        "/usr/local/Cellar/ffmpeg/*/bin/ffmpeg",  # Homebrew source path pattern (approximate)
        str(Path.home() / "bin" / "ffmpeg"),
        str(Path.home() / "Downloads" / "ffmpeg"),
    ]

    for loc in common_locations:
        if "*" in loc:
            # Simple glob-like check for Homebrew Cellar (optional enhancement)
            parent = Path(loc.split("*")[0])
            if parent.exists():
                for sub in parent.glob("*"):
                    candidate = sub / "bin" / "ffmpeg"
                    if candidate.is_file() and os.access(candidate, os.X_OK):
                        return str(candidate)
        else:
            p = Path(loc)
            if p.is_file() and os.access(p, os.X_OK):
                return str(p)

    return None


def get_ffmpeg_version(ffmpeg_path):
    """Run ffmpeg -version and extract key information"""
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None, f"ffmpeg returned error code {result.returncode}"

        lines = result.stdout.strip().splitlines()
        if not lines:
            return None, "No output from ffmpeg -version"

        version_line = lines[0].strip()
        # Optional: extract more details
        build_info = next((line for line in lines if "built with" in line), "").strip()
        configuration = next((line for line in lines if "configuration:" in line), "").strip()

        return {
            "full_output": result.stdout.strip(),
            "version_line": version_line,
            "build_info": build_info,
            "configuration_summary": configuration[:120] + "..." if len(configuration) > 120 else configuration,
        }, None

    except subprocess.TimeoutExpired:
        return None, "ffmpeg -version timed out"
    except FileNotFoundError:
        return None, f"File not found: {ffmpeg_path}"
    except PermissionError:
        return None, f"Permission denied: {ffmpeg_path}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def main():
    print("FFmpeg Checker".center(60, "="))
    print()

    # 1. Find ffmpeg
    ffmpeg_path = find_ffmpeg()

    if not ffmpeg_path:
        print("❌ FFmpeg NOT found in PATH or common locations.")
        print("   Try one of these:")
        print("   • brew install ffmpeg                  (recommended)")
        print("   • Add to PATH: export PATH=\"$PATH:/path/to/ffmpeg/folder\"")
        print("   • Check: command -v ffmpeg")
        sys.exit(1)

    print(f"✅ FFmpeg found at: {ffmpeg_path}")
    print(f"   (from command -v / which: {shutil.which('ffmpeg') or 'not in PATH'})")
    print()

    # 2. Get version info
    info, error = get_ffmpeg_version(ffmpeg_path)

    if error:
        print("⚠️ Could not get version information:")
        print(f"   {error}")
    else:
        print("FFmpeg Version Information:")
        print("-" * 60)
        print(info["version_line"])
        if info["build_info"]:
            print(info["build_info"])
        if info["configuration_summary"]:
            print("Configuration (short):")
            print(info["configuration_summary"])
        print()
        print("Full output preview (first few lines):")
        print("\n".join(info["full_output"].splitlines()[:8]))
        print("... (truncated)")

    print()
    print("To use in code: subprocess.run([r'" + ffmpeg_path + "', 'your', 'args'])")
    print("=" * 60)


if __name__ == "__main__":
    main()