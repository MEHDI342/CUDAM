# cli/cli.py
from typing import Dict, Any
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from ..parser.cuda_parser import CudaParser
from ..translator.kernel_translator import KernelTranslator
from ..translator.memory_model_translator import MemoryModelTranslator
from ..translator.thread_hierarchy_mapper import ThreadHierarchyMapper
from ..optimizer.code_optimizer import CodeOptimizer
from ..utils.error_handler import CudaTranslationError, CudaParseError
from ..utils.logger import get_logger
from .config_parser import ConfigParser

logger = get_logger(__name__)

class CLI:
    """Command-line interface for CUDA to Metal translation."""

    def __init__(self):
        self.parser = CudaParser()
        self.kernel_translator = KernelTranslator()
        self.memory_translator = MemoryModelTranslator()
        self.thread_mapper = ThreadHierarchyMapper()
        self.optimizer = CodeOptimizer()
        self.config_parser = ConfigParser()

    def run(self) -> int:
        """Run the CLI application."""
        args = self._parse_arguments()

        try:
            if args.command == 'translate':
                return self._handle_translation(args)
            elif args.command == 'validate':
                return self._handle_validation(args)
            elif args.command == 'analyze':
                return self._handle_analysis(args)
            else:
                logger.error(f"Unknown command: {args.command}")
                return 1

        except Exception as e:
            logger.error(f"Error during execution: {str(e)}")
            return 1

    def _parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description='CUDA to Metal Translation Tool'
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
        translate_parser.add_argument(
            '--parallel',
            action='store_true',
            help='Enable parallel processing'
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
        analyze_parser.add_argument(
            '--report',
            type=str,
            help='Output file for analysis report'
        )

        args = parser.parse_args()

        # Set logging level based on verbosity
        if args.verbose == 1:
            logging.getLogger().setLevel(logging.INFO)
        elif args.verbose >= 2:
            logging.getLogger().setLevel(logging.DEBUG)

        return args

    def _handle_translation(self, args: argparse.Namespace) -> int:
        """Handle the translation command."""
        input_path = Path(args.input)
        output_path = Path(args.output)

        # Load configuration if provided
        if args.config:
            try:
                config = self.config_parser.parse(args.config)
            except Exception as e:
                logger.error(f"Failed to parse configuration: {e}")
                return 1
        else:
            config = {}

        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        if input_path.is_file():
            return self._translate_file(input_path, output_path, args, config)
        elif input_path.is_dir():
            return self._translate_directory(input_path, output_path, args, config)
        else:
            logger.error(f"Input path does not exist: {input_path}")
            return 1

    def _translate_file(
            self,
            input_file: Path,
            output_dir: Path,
            args: argparse.Namespace,
            config: Dict
    ) -> int:
        """Translate a single CUDA file to Metal."""
        try:
            logger.info(f"Translating file: {input_file}")

            # Parse CUDA code
            ast = self.parser.parse_file(str(input_file))

            # Optimize if requested
            if args.optimize > 0:
                ast = self.optimizer.optimize(ast)

            # Translate to Metal
            metal_code = self.kernel_translator.translate_kernel(ast)
            host_code = self._generate_host_code(ast, args.language)

            # Write output files
            output_basename = input_file.stem
            metal_file = output_dir / f"{output_basename}.metal"
            host_file = output_dir / f"{output_basename}.{self._get_host_extension(args.language)}"

            metal_file.write_text(metal_code)
            host_file.write_text(host_code)

            logger.info(f"Successfully translated {input_file}")
            return 0

        except (CudaParseError, CudaTranslationError) as e:
            logger.error(f"Translation failed: {str(e)}")
            return 1

    def _translate_directory(
            self,
            input_dir: Path,
            output_dir: Path,
            args: argparse.Namespace,
            config: Dict
    ) -> int:
        """Translate all CUDA files in a directory."""
        cuda_files = list(input_dir.rglob("*.cu"))
        if not cuda_files:
            logger.error(f"No CUDA files found in {input_dir}")
            return 1

        if args.parallel:
            return self._translate_parallel(cuda_files, output_dir, args, config)
        else:
            return self._translate_sequential(cuda_files, output_dir, args, config)

    def _translate_parallel(
            self,
            cuda_files: List[Path],
            output_dir: Path,
            args: argparse.Namespace,
            config: Dict
    ) -> int:
        """Translate files in parallel."""
        with ThreadPoolExecutor() as executor:
            futures = []
            for file in cuda_files:
                future = executor.submit(
                    self._translate_file,
                    file,
                    output_dir,
                    args,
                    config
                )
                futures.append((file, future))

            failed = False
            for file, future in futures:
                try:
                    result = future.result()
                    if result != 0:
                        failed = True
                except Exception as e:
                    logger.error(f"Failed to translate {file}: {e}")
                    failed = True

            return 1 if failed else 0

    def _translate_sequential(
            self,
            cuda_files: List[Path],
            output_dir: Path,
            args: argparse.Namespace,
            config: Dict
    ) -> int:
        """Translate files sequentially."""
        failed = False
        for file in cuda_files:
            if self._translate_file(file, output_dir, args, config) != 0:
                failed = True
        return 1 if failed else 0

    def _handle_validation(self, args: argparse.Namespace) -> int:
        """Handle the validation command."""
        input_path = Path(args.input)

        try:
            if input_path.is_file():
                valid = self.parser.validate_file(str(input_path))
                return 0 if valid else 1
            elif input_path.is_dir():
                return self._validate_directory(input_path)
            else:
                logger.error(f"Input path does not exist: {input_path}")
                return 1

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return 1

    def _handle_analysis(self, args: argparse.Namespace) -> int:
        """Handle the analysis command."""
        input_path = Path(args.input)

        try:
            report = self._analyze_code(input_path)

            if args.report:
                Path(args.report).write_text(report)
            else:
                print(report)

            return 0

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return 1

    def _generate_host_code(self, ast: Any, language: str) -> str:
        """Generate host code in the specified language."""
        if language == 'swift':
            return self._generate_swift_host_code(ast)
        else:
            return self._generate_objc_host_code(ast)

    def _get_host_extension(self, language: str) -> str:
        """Get the file extension for host code."""
        return 'swift' if language == 'swift' else 'm'

    def _validate_directory(self, directory: Path) -> int:
        """Validate all CUDA files in a directory."""
        cuda_files = list(directory.rglob("*.cu"))
        if not cuda_files:
            logger.error(f"No CUDA files found in {directory}")
            return 1

        failed = False
        for file in cuda_files:
            try:
                valid = self.parser.validate_file(str(file))
                if not valid:
                    failed = True
            except Exception as e:
                logger.error(f"Failed to validate {file}: {e}")
                failed = True

        return 1 if failed else 0

    def _analyze_code(self, path: Path) -> str:
        """Analyze CUDA code and generate a report."""
        # Implementation details here
        pass

def main():
    """Main entry point for the CLI."""
    cli = CLI()
    sys.exit(cli.run())

if __name__ == '__main__':
    main()