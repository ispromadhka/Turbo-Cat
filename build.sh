#!/bin/bash

# =============================================================================
# TurboCat Build Script
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Parse arguments
BUILD_TYPE="Release"
BUILD_PYTHON="ON"
BUILD_TESTS="ON"
BUILD_BENCHMARKS="ON"
INSTALL_DIR=""
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --no-python)
            BUILD_PYTHON="OFF"
            shift
            ;;
        --no-tests)
            BUILD_TESTS="OFF"
            shift
            ;;
        --no-benchmarks)
            BUILD_BENCHMARKS="OFF"
            shift
            ;;
        --install)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --debug          Build in debug mode"
            echo "  --no-python      Don't build Python bindings"
            echo "  --no-tests       Don't build tests"
            echo "  --no-benchmarks  Don't build benchmarks"
            echo "  --install DIR    Install to directory"
            echo "  --clean          Clean build directory first"
            echo "  --help           Show this help"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_header "TurboCat Build"

# Check dependencies
echo "Checking dependencies..."

if ! command -v cmake &> /dev/null; then
    print_error "CMake not found. Please install CMake 3.18+"
    exit 1
fi
print_success "CMake found: $(cmake --version | head -n1)"

if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    print_error "C++ compiler not found. Please install GCC or Clang"
    exit 1
fi

if command -v g++ &> /dev/null; then
    print_success "GCC found: $(g++ --version | head -n1)"
fi

# Check CPU features
if grep -q avx512 /proc/cpuinfo 2>/dev/null; then
    print_success "AVX-512 support detected"
elif grep -q avx2 /proc/cpuinfo 2>/dev/null; then
    print_success "AVX2 support detected"
else
    print_warning "No advanced SIMD support detected"
fi

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo ""
    echo "Cleaning build directory..."
    rm -rf build
    print_success "Build directory cleaned"
fi

# Create build directory
echo ""
echo "Creating build directory..."
mkdir -p build
cd build

# Configure
print_header "Configuring CMake"

CMAKE_ARGS=(
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    "-DTURBOCAT_BUILD_PYTHON=$BUILD_PYTHON"
    "-DTURBOCAT_BUILD_TESTS=$BUILD_TESTS"
    "-DTURBOCAT_BUILD_BENCHMARKS=$BUILD_BENCHMARKS"
    "-DTURBOCAT_USE_OPENMP=ON"
)

if [ -n "$INSTALL_DIR" ]; then
    CMAKE_ARGS+=("-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR")
fi

echo "CMake arguments: ${CMAKE_ARGS[*]}"
echo ""

cmake .. "${CMAKE_ARGS[@]}"

# Build
print_header "Building"

NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "Using $NPROC parallel jobs"
echo ""

cmake --build . --parallel "$NPROC"

print_success "Build completed!"

# Run tests if enabled
if [ "$BUILD_TESTS" = "ON" ] && [ -f "turbocat_tests" ]; then
    print_header "Running Tests"
    ./turbocat_tests --gtest_color=yes || true
fi

# Install if requested
if [ -n "$INSTALL_DIR" ]; then
    print_header "Installing"
    cmake --install .
    print_success "Installed to $INSTALL_DIR"
fi

# Print summary
print_header "Build Summary"

echo "Build type:    $BUILD_TYPE"
echo "Python:        $BUILD_PYTHON"
echo "Tests:         $BUILD_TESTS"
echo "Benchmarks:    $BUILD_BENCHMARKS"

if [ -f "_turbocat"* ] || [ -f "python/_turbocat"* ]; then
    echo ""
    print_success "Python module built!"
    echo ""
    echo "To install Python package:"
    echo "  cd $SCRIPT_DIR/python"
    echo "  pip install -e ."
fi

echo ""
print_success "TurboCat build completed successfully!"
echo ""
echo "Next steps:"
echo "  1. Run benchmarks: ./turbocat_bench"
echo "  2. Install Python: cd ../python && pip install -e ."
echo "  3. Try example: python -c 'import turbocat; turbocat.print_info()'"
