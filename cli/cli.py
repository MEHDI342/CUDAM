import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..parser.cuda_parser import CudaParser
from ..translator.kernel_translator import KernelTranslator
from ..translator.host_adapter import HostAdapter
from ..optimizer.metal_optimizer import MetalOptimizer
from ..utils.error_handler import CudaTranslationError
from ..utils.logger import get_logger
from .config_parser import ConfigParser, MetalConfig

logger = get_logger(__name__)

@dataclass
class TranslationConfig:
    """Translation configuration parameters"""
    input_path: Path
    output_path: Path
    metal_target: str = "2.4"
    optimization_level: int = 2
    generate_tests: bool = True
    preserve_comments: bool = True
    source_map: bool = True
    enable_profiling: bool = False

class CLI:
    """
    Production-grade CLI implementation for CUDA to Metal translation.
    Thread-safe, optimized for performance, with comprehensive error handling.
    """

    def __init__(self):
        """Initialize CLI with required components"""
        self.parser = CudaParser()
        self.kernel_translator = KernelTranslator()
        self.host_adapter = HostAdapter()
        self.optimizer = MetalOptimizer()
        self.config_parser = ConfigParser()

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) * 4))

        # Translation cache for performance
        self._translation_cache: Dict[str, Any] = {}

    def run(self) -> int:
        """
        Main entry point for CLI execution.
        Returns exit code (0 for success, non-zero for error)
        """
        try:
            args = self._parse_arguments()
            config = self._load_configuration(args)

            if args.command == 'translate':
                return self._handle_translation(args, config)
            elif args.command == 'validate':
                return self._handle_validation(args)
            elif args.command == 'analyze':
                return self._handle_analysis(args)

            logger.error(f"Unknown command: {args.command}")
            return 1

        except Exception as e:
            logger.error(f"Error during execution: {str(e)}")
            return 1
        finally:
            self.executor.shutdown(wait=True)

    def _parse_arguments(self) -> argparse.Namespace:
        """Parse and validate command line arguments"""
        parser = argparse.ArgumentParser(
            description='CUDA to Metal Translation Tool',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        parser.add_argument(
            '--verbose', '-v',
            action='count',
            default=0,
            help='Increase output verbosity'
        )

        parser.add_argument(
            '--config',
            type=str,
            help='Path to configuration file'
        )

        subparsers = parser.add_subparsers(dest='command', required=True)

        # Translation command
        translate_parser = subparsers.add_parser('translate')
        translate_parser.add_argument(
            'input',
            type=str,
            help='Input CUDA file or directory'
        )
        translate_parser.add_argument(
            'output',
            type=str,
            help='Output directory for Metal code'
        )
        translate_parser.add_argument(
            '--language',
            choices=['swift', 'objc'],
            default='swift',
            help='Output language for host code'
        )
        translate_parser.add_argument(
            '--optimize',
            type=int,
            choices=[0, 1, 2, 3],
            default=2,
            help='Optimization level'
        )

        # Validation command
        validate_parser = subparsers.add_parser('validate')
        validate_parser.add_argument(
            'input',
            type=str,
            help='Input CUDA file or directory to validate'
        )

        # Analysis command
        analyze_parser = subparsers.add_parser('analyze')
        analyze_parser.add_argument(
            'input',
            type=str,
            help='Input CUDA file or directory to analyze'
        )

        args = parser.parse_args()

        # Set logging level based on verbosity
        if args.verbose == 1:
            logging.getLogger().setLevel(logging.INFO)
        elif args.verbose >= 2:
            logging.getLogger().setLevel(logging.DEBUG)

        return args

    def _load_configuration(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Load and validate configuration from file"""
        if not args.config:
            return {}

        try:
            return self.config_parser.parse(args.config)
        except Exception as e:
            logger.error(f"Failed to parse configuration: {e}")
            raise

    def _handle_translation(self, args: argparse.Namespace, config: Dict[str, Any]) -> int:
        """Handle translation command with full error handling"""
        try:
            input_path = Path(args.input)
            output_path = Path(args.output)

            # Validate paths
            if not input_path.exists():
                raise CudaTranslationError(f"Input path does not exist: {input_path}")

            output_path.mkdir(parents=True, exist_ok=True)

            if input_path.is_file():
                return self._translate_file(input_path, output_path, args, config)
            elif input_path.is_dir():
                return self._translate_directory(input_path, output_path, args, config)

            logger.error(f"Invalid input path: {input_path}")
            return 1

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return 1

    def _translate_file(self, input_file: Path, output_dir: Path,
                        args: argparse.Namespace, config: Dict[str, Any]) -> int:
        """Translate single CUDA file to Metal"""
        try:
            logger.info(f"Translating file: {input_file}")

            # Parse CUDA code
            ast = self.parser.parse_file(str(input_file))

            # Apply optimizations
            if args.optimize > 0:
                ast = self.optimizer.optimize(ast, args.optimize)

            # Generate Metal code
            metal_code = self.kernel_translator.translate_kernel(ast)

            # Generate host code
            if args.language == 'swift':
                host_code = self._generate_swift_host_code(ast)
            else:
                host_code = self._generate_objc_host_code(ast)

            # Write output files
            output_base = output_dir / input_file.stem
            metal_file = output_base.with_suffix('.metal')
            host_file = output_base.with_suffix(
                '.swift' if args.language == 'swift' else '.m'
            )

            metal_file.write_text(metal_code)
            host_file.write_text(host_code)

            logger.info(f"Successfully translated {input_file}")
            return 0

        except Exception as e:
            logger.error(f"Failed to translate {input_file}: {e}")
            return 1

    def _generate_swift_host_code(self, ast: Any) -> str:
        """Generate Swift host code with proper Metal setup"""
        metal_code = []

        # Import statements
        metal_code.append("""
            import Metal
            import MetalKit
            
            // MARK: - Metal Setup
            guard let device = MTLCreateSystemDefaultDevice() else {
                fatalError("Metal is not supported on this device")
            }
            
            guard let commandQueue = device.makeCommandQueue() else {
                fatalError("Failed to create command queue")
            }
            """)

        # Add buffer creation
        for buffer in self._extract_buffers(ast):
            metal_code.append(self._generate_swift_buffer(buffer))

        # Add kernel execution
        for kernel in self._extract_kernels(ast):
            metal_code.append(self._generate_swift_kernel_execution(kernel))

        return "\n".join(metal_code)

    def _generate_objc_host_code(self, ast: Any) -> str:
        """Generate Objective-C host code with proper Metal setup"""
        metal_code = []

        # Import and setup
        metal_code.append("""
            #import <Metal/Metal.h>
            #import <MetalKit/MetalKit.h>
            
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                NSLog(@"Metal is not supported on this device");
                return;
            }
            
            id<MTLCommandQueue> commandQueue = [device newCommandQueue];
            if (!commandQueue) {
                NSLog(@"Failed to create command queue");
                return;
            }
            """)

        # Add buffer creation
        for buffer in self._extract_buffers(ast):
            metal_code.append(self._generate_objc_buffer(buffer))

        # Add kernel execution
        for kernel in self._extract_kernels(ast):
            metal_code.append(self._generate_objc_kernel_execution(kernel))

        return "\n".join(metal_code)

    def _extract_kernels(self, ast: Any) -> List[Any]:
        """Extract kernel nodes from AST"""
        kernels = []
        for node in ast.walk_preorder():
            if hasattr(node, 'is_kernel') and node.is_kernel():
                kernels.append(node)
        return kernels

    def _extract_buffers(self, ast: Any) -> List[Any]:
        """Extract buffer nodes from AST"""
        buffers = []
        for node in ast.walk_preorder():
            if hasattr(node, 'is_buffer') and node.is_buffer():
                buffers.append(node)
        return buffers

    def cleanup(self):
        """Clean up resources"""
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Direct script execution
def main():
    """Main entry point for CLI"""
    cli = CLI()
    try:
        return cli.run()
    finally:
        cli.cleanup()

if __name__ == '__main__':
    import sys
    sys.exit(main())