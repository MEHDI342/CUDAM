from typing import Optional, Dict, Any
import traceback

class CudaError(Exception):
    """Base class for CUDA-related errors."""
    def __init__(self, message: str, error_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        error_str = f"[Error {self.error_code}] " if self.error_code else ""
        error_str += self.message
        if self.details:
            error_str += "\nDetails:\n" + "\n".join(f"  {k}: {v}" for k, v in self.details.items())
        return error_str

class CudaParseError(CudaError):
    """Exception raised for errors in parsing CUDA code."""
    def __init__(self, message: str, line: Optional[int] = None, column: Optional[int] = None, filename: Optional[str] = None):
        details = {"line": line, "column": column, "filename": filename}
        super().__init__(message, error_code=1001, details=details)

class CudaTranslationError(CudaError):
    """Exception raised for errors in translating CUDA code to Metal."""
    def __init__(self, message: str, cuda_construct: Optional[str] = None, metal_equivalent: Optional[str] = None):
        details = {"cuda_construct": cuda_construct, "metal_equivalent": metal_equivalent}
        super().__init__(message, error_code=2001, details=details)

class CudaTypeError(CudaError):
    """Exception raised for type-related errors in CUDA code."""
    def __init__(self, message: str, expected_type: Optional[str] = None, actual_type: Optional[str] = None):
        details = {"expected_type": expected_type, "actual_type": actual_type}
        super().__init__(message, error_code=3001, details=details)

class CudaNotSupportedError(CudaError):
    """Exception raised for CUDA features not supported in Metal."""
    def __init__(self, message: str, cuda_feature: str):
        details = {"cuda_feature": cuda_feature}
        super().__init__(message, error_code=4001, details=details)

class CudaWarning:
    """Warning class for non-critical issues in CUDA code parsing or translation."""
    def __init__(self, message: str, warning_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.warning_code = warning_code
        self.details = details or {}

    def __str__(self):
        warning_str = f"[Warning {self.warning_code}] " if self.warning_code else ""
        warning_str += self.message
        if self.details:
            warning_str += "\nDetails:\n" + "\n".join(f"  {k}: {v}" for k, v in self.details.items())
        return warning_str

def handle_exception(e: Exception, logger):
    """
    Handle exceptions, log them, and optionally perform additional actions.
    """
    if isinstance(e, CudaError):
        logger.error(str(e))
    else:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug(f"Stack trace:\n{''.join(traceback.format_tb(e.__traceback__))}")

def raise_cuda_parse_error(message: str, line: Optional[int] = None, column: Optional[int] = None, filename: Optional[str] = None):
    """Convenience function to raise a CudaParseError."""
    raise CudaParseError(message, line, column, filename)

def raise_cuda_translation_error(message: str, cuda_construct: Optional[str] = None, metal_equivalent: Optional[str] = None):
    """Convenience function to raise a CudaTranslationError."""
    raise CudaTranslationError(message, cuda_construct, metal_equivalent)

def raise_cuda_type_error(message: str, expected_type: Optional[str] = None, actual_type: Optional[str] = None):
    """Convenience function to raise a CudaTypeError."""
    raise CudaTypeError(message, expected_type, actual_type)

def raise_cuda_not_supported_error(message: str, cuda_feature: str):
    """Convenience function to raise a CudaNotSupportedError."""
    raise CudaNotSupportedError(message, cuda_feature)

def issue_cuda_warning(message: str, warning_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None, logger=None):
    """Issue a CudaWarning and optionally log it."""
    warning = CudaWarning(message, warning_code, details)
    if logger:
        logger.warning(str(warning))
    return warning