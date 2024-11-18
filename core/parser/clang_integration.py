from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
import clang.cindex
from clang.cindex import Index, TranslationUnit, Cursor, CursorKind, TypeKind

from .ast_nodes import (
    CUDAType,
    CUDAQualifier,
    CUDANode,
    CUDAKernel,
    CUDAParameter,
    CUDACompoundStmt,
    CUDAThreadIdx,
    CUDABlockIdx,
    CUDAGridDim,
    CUDAAtomicOperation,
    CUDASharedMemory,
    CUDATexture,
    CUDABarrier,
    SourceLocation,
    CUDANodeType
)

class ClangParser:
    """CUDA parser using Clang's Python bindings"""

    def __init__(self, cuda_path: Optional[str] = None):
        self.index = Index.create()
        self.cuda_path = cuda_path or self._find_cuda_path()
        self.cuda_version = self._detect_cuda_version()
        self._init_compilation_args()

    def _find_cuda_path(self) -> str:
        """Find CUDA installation path"""
        common_paths = [
            "/usr/local/cuda",
            "/usr/cuda",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
            "C:/CUDA"
        ]

        for path in common_paths:
            if Path(path).exists():
                return str(Path(path))
        raise RuntimeError("CUDA installation not found")

    def _detect_cuda_version(self) -> str:
        """Detect CUDA version from installation"""
        version_file = Path(self.cuda_path) / "version.txt"
        if version_file.exists():
            content = version_file.read_text()
            import re
            if match := re.search(r'V(\d+\.\d+\.\d+)', content):
                return match.group(1)
        return "unknown"

    def _init_compilation_args(self):
        """Initialize CUDA compilation arguments"""
        self.compilation_args = [
            "-x", "cuda",
            "--cuda-gpu-arch=sm_75",
            "-std=c++14",
            f"-I{Path(self.cuda_path)/'include'}",
            "-D__CUDACC__",
            "-D__CUDA_ARCH__=750",
            "-DNDEBUG",
        ]

    def parse_file(self, cuda_file: Union[str, Path]) -> Optional[CUDANode]:
        """Parse CUDA source file into AST"""
        try:
            tu = self.index.parse(
                str(cuda_file),
                args=self.compilation_args,
                options=(
                        TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD |
                        TranslationUnit.PARSE_INCOMPLETE
                )
            )

            # Check for fatal errors
            if self._has_fatal_errors(tu):
                return None

            # Convert to CUDA AST
            return self._process_translation_unit(tu.cursor)

        except Exception as e:
            logging.error(f"Failed to parse {cuda_file}: {str(e)}")
            return None

    def _has_fatal_errors(self, tu: TranslationUnit) -> bool:
        """Check for fatal parsing errors"""
        has_fatal = False
        for diag in tu.diagnostics:
            if diag.severity >= diag.Error:
                logging.error(
                    f"{diag.location.file}:{diag.location.line} - {diag.spelling}"
                )
                has_fatal = True
        return has_fatal

    def _process_translation_unit(self, cursor: Cursor) -> CUDANode:
        """Process translation unit cursor"""
        root = CUDANode(
            line=cursor.location.line,
            column=cursor.location.column
        )

        for child in cursor.get_children():
            if node := self._process_cursor(child):
                root.add_child(node)

        return root

    def _process_cursor(self, cursor: Cursor) -> Optional[CUDANode]:
        """Process a single Clang cursor"""
        source_location = SourceLocation(
            file=str(cursor.location.file) if cursor.location.file else "",
            line=cursor.location.line,
            column=cursor.location.column,
            offset=cursor.location.offset
        )

        # Handle different cursor kinds
        if cursor.kind == CursorKind.FUNCTION_DECL:
            return self._process_function(cursor, source_location)
        elif cursor.kind == CursorKind.VAR_DECL:
            return self._process_variable(cursor, source_location)
        elif cursor.kind == CursorKind.MEMBER_REF_EXPR:
            return self._process_member_ref(cursor, source_location)
        elif cursor.kind == CursorKind.CALL_EXPR:
            return self._process_call(cursor, source_location)

        return None

# ... rest of the implementation remains the same ...