from pathlib import Path
from CUDAM.parser.clang_integration import CUDAClangParser
from CUDAM.translator.host_translator import CUDAHostTranslator
from CUDAM.generator.metal_generator import MetalGenerator

def translate_cuda_to_metal(cuda_file: str):
    # Initialize components
    parser = CUDAClangParser()
    host_translator = CUDAHostTranslator()
    metal_generator = MetalGenerator()

    # Parse CUDA file
    cuda_ast = parser.parse_file(cuda_file)
    if not cuda_ast:
        print("Failed to parse CUDA file")
        return

    # Find kernel functions
    kernels = []
    def find_kernels(node):
        if hasattr(node, 'is_kernel') and node.is_kernel():
            kernels.append(node)
    cuda_ast.traverse(find_kernels)

    # Generate Metal code
    output_dir = Path('metal_output')
    output_dir.mkdir(exist_ok=True)

    # Generate kernel code
    for kernel in kernels:
        metal_code = metal_generator.generate_metal_code(kernel)
        kernel_file = output_dir / f"{kernel.name}.metal"
        kernel_file.write_text(metal_code)

    # Translate host code
    with open(cuda_file) as f:
        cuda_host_code = f.read()
    metal_host_code = host_translator.translate_host_code(cuda_host_code, target_lang='swift')
    host_file = output_dir / "host.swift"
    host_file.write_text(metal_host_code)

if __name__ == "__main__":
    cuda_file = "vector_add.cu"
    translate_cuda_to_metal(cuda_file)