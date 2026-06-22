#!/bin/sh

set -eu

nvcc=${NVCC:-nvcc}
nvidia_smi=${NVIDIA_SMI:-nvidia-smi}

if ! command -v "$nvcc" >/dev/null 2>&1; then
	echo "error: nvcc not found; set NVCC to its path" >&2
	exit 1
fi

if ! command -v "$nvidia_smi" >/dev/null 2>&1; then
	echo "error: nvidia-smi not found; set NVIDIA_SMI to its path" >&2
	exit 1
fi

gpu_caps=$("$nvidia_smi" --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null) || {
	echo "error: could not query GPU compute capabilities with nvidia-smi" >&2
	exit 1
}

min_gpu_arch=
old_ifs=$IFS
IFS='
'
for capability in $gpu_caps; do
	IFS=.
	set -- $capability
	IFS='
'
	major=$(printf '%s' "${1:-}" | tr -d '[:space:]')
	minor=$(printf '%s' "${2:-}" | tr -d '[:space:]')
	case "$major$minor" in
	''|*[!0-9]*)
		echo "error: unexpected compute capability from nvidia-smi: $capability" >&2
		exit 1
		;;
	esac
	gpu_arch=$((major * 10 + minor))
	if [ -z "$min_gpu_arch" ] || [ "$gpu_arch" -lt "$min_gpu_arch" ]; then
		min_gpu_arch=$gpu_arch
	fi
done
IFS=$old_ifs

if [ -z "$min_gpu_arch" ]; then
	echo "error: no NVIDIA GPUs found" >&2
	exit 1
fi

best_arch=
for architecture in $("$nvcc" --list-gpu-arch 2>/dev/null); do
	arch=${architecture#compute_}
	case "$architecture:$arch" in
	compute_*:*[!0-9]*) continue ;;
	compute_*:*) ;;
	*) continue ;;
	esac
	if [ "$arch" -le "$min_gpu_arch" ] && \
		{ [ -z "$best_arch" ] || [ "$arch" -gt "$best_arch" ]; }; then
		best_arch=$arch
	fi
done

if [ -z "$best_arch" ]; then
	echo "error: nvcc cannot target the oldest installed GPU (compute_$min_gpu_arch)" >&2
	exit 1
fi

printf '%s\n' "$best_arch"
