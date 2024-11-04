# utils/file_utils.py

import os
import shutil
import hashlib
import tempfile
from pathlib import Path
from typing import List, Set, Dict, Optional, Generator
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import logging

from .error_handler import CudaTranslationError
from .logger import get_logger

logger = get_logger(__name__)

class FileCache:
    """Thread-safe file cache manager."""
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "cuda_metal_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._cache_index: Dict[str, Path] = {}
        self._load_cache_index()

    def _load_cache_index(self):
        """Load cache index from disk."""
        with self._lock:
            index_file = self.cache_dir / "index.json"
            if index_file.exists():
                import json
                with open(index_file, 'r') as f:
                    self._cache_index = {k: Path(v) for k, v in json.load(f).items()}

    def _save_cache_index(self):
        """Save cache index to disk."""
        with self._lock:
            index_file = self.cache_dir / "index.json"
            import json
            with open(index_file, 'w') as f:
                json.dump({k: str(v) for k, v in self._cache_index.items()}, f)

    def get_cached_path(self, key: str) -> Optional[Path]:
        """Get cached file path if exists."""
        with self._lock:
            return self._cache_index.get(key)

    def add_to_cache(self, key: str, file_path: Path):
        """Add file to cache."""
        with self._lock:
            cache_path = self.cache_dir / hashlib.sha256(key.encode()).hexdigest()
            shutil.copy2(file_path, cache_path)
            self._cache_index[key] = cache_path
            self._save_cache_index()

class FileTracker:
    """Tracks file dependencies and modifications."""
    def __init__(self):
        self.dependencies: Dict[Path, Set[Path]] = {}
        self._lock = Lock()

    def add_dependency(self, source: Path, dependency: Path):
        """Add a dependency relationship."""
        with self._lock:
            if source not in self.dependencies:
                self.dependencies[source] = set()
            self.dependencies[source].add(dependency)

    def get_dependencies(self, source: Path) -> Set[Path]:
        """Get all dependencies for a file."""
        with self._lock:
            return self.dependencies.get(source, set())

    def is_modified(self, source: Path, dependency: Path) -> bool:
        """Check if dependency is modified after source."""
        try:
            source_mtime = source.stat().st_mtime
            dep_mtime = dependency.stat().st_mtime
            return dep_mtime > source_mtime
        except OSError:
            return True

class FileUtils:
    """Utility class for file operations with Metal-specific optimizations."""

    def __init__(self):
        self.cache = FileCache()
        self.tracker = FileTracker()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="cuda_metal_"))
        self._lock = Lock()

    def read_file(self, path: Path, encoding: str = 'utf-8') -> str:
        """Read file with caching and error handling."""
        try:
            with open(path, 'r', encoding=encoding) as f:
                content = f.read()

            # Cache the content
            cache_key = f"{path}:{path.stat().st_mtime}"
            self.cache.add_to_cache(cache_key, path)

            return content

        except UnicodeDecodeError:
            logger.warning(f"Failed to read {path} with {encoding} encoding, trying alternate encodings")
            for alt_encoding in ['latin1', 'cp1252']:
                try:
                    with open(path, 'r', encoding=alt_encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise CudaTranslationError(f"Unable to read file {path} with any supported encoding")

        except OSError as e:
            raise CudaTranslationError(f"Failed to read file {path}: {str(e)}")

    def write_file(self, path: Path, content: str, encoding: str = 'utf-8', backup: bool = True):
        """Write file with backup and atomic operation."""
        if backup and path.exists():
            self._create_backup(path)

        # Write to temporary file first
        temp_path = self.temp_dir / f"{path.name}.tmp"
        try:
            with open(temp_path, 'w', encoding=encoding) as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())

            # Atomic move
            shutil.move(str(temp_path), str(path))

        except OSError as e:
            raise CudaTranslationError(f"Failed to write file {path}: {str(e)}")
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _create_backup(self, path: Path):
        """Create backup of existing file."""
        backup_path = path.with_suffix(path.suffix + '.bak')
        try:
            shutil.copy2(path, backup_path)
        except OSError as e:
            logger.warning(f"Failed to create backup of {path}: {str(e)}")

    def process_directory(self,
                          directory: Path,
                          pattern: str = "*.cu",
                          recursive: bool = True) -> Generator[Path, None, None]:
        """Process directory with parallel file scanning."""
        try:
            if recursive:
                paths = directory.rglob(pattern)
            else:
                paths = directory.glob(pattern)

            with ThreadPoolExecutor() as executor:
                yield from executor.map(self._process_file, paths)

        except OSError as e:
            raise CudaTranslationError(f"Failed to process directory {directory}: {str(e)}")

    def _process_file(self, path: Path) -> Path:
        """Process individual file with validation."""
        if not path.is_file():
            logger.warning(f"Skipping non-file path: {path}")
            return None

        return path

    def ensure_directory(self, path: Path):
        """Ensure directory exists with proper permissions."""
        try:
            path.mkdir(parents=True, exist_ok=True)

            # Set appropriate permissions
            if os.name == 'posix':
                os.chmod(path, 0o755)

        except OSError as e:
            raise CudaTranslationError(f"Failed to create directory {path}: {str(e)}")

    def copy_with_metadata(self, src: Path, dst: Path):
        """Copy file with all metadata preserved."""
        try:
            shutil.copy2(src, dst)

            # Track dependency
            self.tracker.add_dependency(dst, src)

        except OSError as e:
            raise CudaTranslationError(f"Failed to copy {src} to {dst}: {str(e)}")

    def get_relative_path(self, path: Path, base: Path) -> Path:
        """Get relative path with validation."""
        try:
            return path.relative_to(base)
        except ValueError:
            return path

    def cleanup(self):
        """Clean up temporary files."""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except OSError as e:
            logger.warning(f"Failed to clean up temporary files: {str(e)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

logger.info("FileUtils initialized with Metal-specific optimizations.")