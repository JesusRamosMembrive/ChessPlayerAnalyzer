#!/usr/bin/env python3
"""
Build script for MkDocs documentation website
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, capture_output=True, text=True)
        print(f"✅ {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {cmd}")
        print(f"Error: {e.stderr}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    dependencies = [
        "mkdocs",
        "mkdocs-material",
        "mkdocs-mermaid2-plugin"
    ]

    missing = []
    for dep in dependencies:
        try:
            subprocess.run([sys.executable, "-c", f"import {dep.replace('-', '_')}"],
                         check=True, capture_output=True)
        except subprocess.CalledProcessError:
            missing.append(dep)

    if missing:
        print("❌ Missing dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\nInstall with:")
        print(f"pip install {' '.join(missing)}")
        return False

    print("✅ All dependencies installed")
    return True

def create_missing_files():
    """Create any missing documentation files referenced in nav"""

    missing_files = [
        "docs/api/examples.md",
        "docs/modules/database/README.md"
    ]

    for file_path in missing_files:
        path = Path(file_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

            # Create placeholder content
            if "examples" in str(path):
                content = """# API Examples

This page contains practical examples of using the ChessPlayerAnalyzer API.

(Content to be added)

---

## See Also

- [API Endpoints](endpoints.md)
- [API Schemas](schemas.md)
"""
            elif "database" in str(path):
                content = """# Database Module

Documentation for the database layer and ORM models.

(Content to be added)

---

## See Also

- [Database Schema](../../architecture/database-schema.md)
"""
            else:
                content = f"""# {path.stem.replace('-', ' ').title()}

Documentation content to be added.
"""

            path.write_text(content, encoding='utf-8')
            print(f"✅ Created placeholder: {file_path}")

def build_site(serve=False):
    """Build the documentation site"""

    print("🚀 Building ChessPlayerAnalyzer Documentation")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        return False

    # Create missing files
    create_missing_files()

    # Build the site
    if serve:
        print("\n📡 Starting development server...")
        if run_command("mkdocs serve"):
            print("🌐 Documentation available at: http://127.0.0.1:8000")
            return True
    else:
        print("\n🔨 Building static site...")
        if run_command("mkdocs build"):
            print("✅ Site built successfully!")
            print("📁 Output directory: site/")
            return True

    return False

def deploy_github_pages():
    """Deploy to GitHub Pages"""

    print("\n🚀 Deploying to GitHub Pages...")

    if run_command("mkdocs gh-deploy --force"):
        print("✅ Successfully deployed to GitHub Pages!")
        return True

    return False

def main():
    """Main entry point"""

    # Change to repository root
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "serve":
            build_site(serve=True)
        elif command == "build":
            build_site(serve=False)
        elif command == "deploy":
            if build_site(serve=False):
                deploy_github_pages()
        elif command == "deps":
            print("Installing documentation dependencies...")
            deps = ["mkdocs", "mkdocs-material", "mkdocs-mermaid2-plugin"]
            for dep in deps:
                run_command(f"pip install {dep}")
        else:
            print(f"Unknown command: {command}")
            print_usage()
    else:
        print_usage()

def print_usage():
    """Print usage information"""
    print("""
Usage: python scripts/build-docs.py <command>

Commands:
  deps    - Install documentation dependencies
  build   - Build static documentation site
  serve   - Start development server with live reload
  deploy  - Build and deploy to GitHub Pages

Examples:
  python scripts/build-docs.py deps
  python scripts/build-docs.py serve
  python scripts/build-docs.py build
  python scripts/build-docs.py deploy
""")

if __name__ == "__main__":
    main()