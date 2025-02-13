import logging
from typing import List, Optional, Dict, Any, Union

from ...utils.error_handler import CudaTranslationError
from ..parser.ast_nodes import (
    CUDANode, CUDAKernel, CUDAParameter, CUDASharedMemory, CUDAType,
    CUDAExpressionNode, CUDAStatement, VariableNode, CUDAQualifier,
    CUDANodeType, CUDAThreadIdx, CUDABlockIdx, CUDAKernel as KernelNode,
    FunctionNode
)

logger = logging.getLogger(__name__)

class KernelTranslator:
    """
if you're looking at this == thank you it means that you really took time into looking at this shitshow im trying to build if you want to help feel free, ill really use some help
    """

    def __init__(self):
        # Optionally store translator-level settings or optimization flags
        self.enable_debug_comments = True

    def translate_ast(self, root: CUDANode) -> str:
        """
        Translate the entire AST (which may contain multiple kernels,
        device functions, global vars, etc.) into a single .metal source.
        """
        metal_lines = []

        # Prologue / headers
        metal_lines.append("// -- Auto-Generated Metal Code --")
        metal_lines.append("#include <metal_stdlib>")
        metal_lines.append("#include <metal_atomic>")
        metal_lines.append("using namespace metal;\n")

        # Walk top-level children
        for child in root.children:
            code = self._translate_node(child, top_level=True)
            if code:
                # Separate top-level items with blank lines
                metal_lines.append(code)
                metal_lines.append("")

        return "\n".join(metal_lines)

    def _translate_node(self, node: CUDANode, top_level: bool = False, indent: int = 0) -> str:
        """
        Main dispatch method for translating an arbitrary AST node to MSL.
        Calls specialized sub-methods based on node type.
        """
        # Distinguish node by its class or qualifiers
        if isinstance(node, CUDAKernel):
            return self._translate_kernel(node)
        elif isinstance(node, FunctionNode):
            # Device or host function
            return self._translate_function(node)
        elif isinstance(node, VariableNode):
            # Possibly a global var or local var
            if top_level:
                return self._translate_global_variable(node)
            else:
                return self._translate_local_variable(node, indent)
        elif isinstance(node, CUDASharedMemory):
            # Shared memory, translate to threadgroup array
            return self._translate_shared_memory(node, indent)
        elif isinstance(node, CUDAExpressionNode):
            return self._translate_expression(node, indent)
        elif isinstance(node, CUDAStatement):
            return self._translate_statement(node, indent)
        else:
            # If no specialized method, handle generically
            return self._translate_generic_node(node, indent)

    # --------------------------------------------------------------------------
    # KERNEL, FUNCTION, VARIABLE TRANSLATION
    # --------------------------------------------------------------------------
    def _translate_kernel(self, kernel: CUDAKernel) -> str:
        """
        Translate a __global__ CUDA kernel into a Metal kernel function.
        """
        # Kernel signature
        kernel_name = kernel.name
        # e.g. "kernel void <name>(uint3 tid [[thread_position_in_grid]])"

        # We can add dynamic threadgroup metadata or attributes:
        # For example, if the parser computed optimal threads_per_threadgroup
        # let's say the user sets kernel.optimization_hints['threads_per_threadgroup']
        threads_per_threadgroup = kernel.optimization_hints.get("threads_per_threadgroup", (256, 1, 1))
        # We'll store them for a future attribute usage if you want:
        #   [[threads_per_threadgroup(256,1,1)]]

        lines = []
        lines.append(f"kernel void {kernel_name}(")

        # Translate parameters
        param_lines = []
        for i, param in enumerate(kernel.parameters):
            param_lines.append(self._translate_kernel_parameter(param, index=i))
        lines.append("  " + ",\n  ".join(param_lines))

        # We add built-in thread position variables
        lines.append("  , uint3 gid [[thread_position_in_grid]]")
        lines.append("  , uint3 tid_in_group [[thread_position_in_threadgroup]]")
        lines.append("  , uint3 tg_id [[threadgroup_position_in_grid]]")
        lines.append("  , uint3 tg_size [[threads_per_threadgroup]]")
        lines.append(") {")

        # Optionally generate a debug comment
        if self.enable_debug_comments:
            lines.append(f"  // Kernel: {kernel_name} (line {kernel.line}, col {kernel.column})")

        # Translate kernel body
        # In the AST, kernel.body might be a compound statement or list of statements
        for stmt in kernel.body:
            lines.append(self._translate_node(stmt, indent=1))

        lines.append("}")  # close kernel function

        return "\n".join(lines)

    def _translate_kernel_parameter(self, param: CUDAParameter, index: int) -> str:
        """
        Translate a CUDA kernel parameter into a Metal function parameter,
        including address space and buffer attributes if needed.
        """
        # Example: device float* data [[buffer(0)]]
        # We get the type from param.param_type, and see if it's pointer or not
        metal_type = self._cuda_type_to_metal(param.param_type)
        if param.is_pointer:
            # We'll default to device pointer
            # Or if the parser says it's "const", we might use "const device"
            if CUDAQualifier.CONST in param.qualifiers:
                addr_space = "const device"
            else:
                addr_space = "device"
            return f"{addr_space} {metal_type}* {param.name} [[buffer({index})]]"
        else:
            # For a by-value param, just do something like "float param"
            return f"{metal_type} {param.name}"

    def _translate_function(self, func: FunctionNode) -> str:
        """
        Translate a non-kernel function. Often device or host function.
        Mapped to `inline` or `device` function in MSL.
        """
        lines = []
        func_name = func.name
        return_type = self._cuda_type_to_metal(func.return_type)

        # See if it's device or host function
        if func.is_device_func():
            # We'll call it "inline device" or just "inline"
            # Some prefer "inline" + "static" to limit scope.
            signature = f"inline {return_type} {func_name}("
        else:
            # If it's neither device nor global, treat as a standard function.
            signature = f"inline {return_type} {func_name}("

        # Parameters
        param_strs = []
        for i, p in enumerate(func.parameters):
            param_type = self._cuda_type_to_metal(p.param_type)
            if p.is_pointer:
                addr_space = "device"
                if CUDAQualifier.CONST in p.qualifiers:
                    addr_space = "const device"
                param_strs.append(f"{addr_space} {param_type}* {p.name}")
            else:
                param_strs.append(f"{param_type} {p.name}")

        signature += ", ".join(param_strs)
        signature += ")"

        lines.append(signature + " {")
        if self.enable_debug_comments:
            lines.append(f"  // Function: {func_name}")

        # Function body
        for stmt in func.body:
            lines.append(self._translate_node(stmt, indent=1))

        lines.append("}")
        return "\n".join(lines)

    def _translate_global_variable(self, var_node: VariableNode) -> str:
        """
        Translate a global variable (e.g., file-scope variable) into MSL.
        If it's in shared or constant, we map accordingly. If truly global
        scope in CUDA, that typically means device or constant memory.
        """
        # Memory space logic
        address_space = "device"
        if CUDAQualifier.CONST in var_node.qualifiers:
            address_space = "constant"
        elif CUDAQualifier.SHARED in var_node.qualifiers:
            address_space = "threadgroup"

        metal_type = self._cuda_type_to_metal(var_node.var_type)
        if var_node.is_pointer:
            # E.g. "device float* myGlobalVar;"
            return f"{address_space} {metal_type}* {var_node.name};"
        else:
            # E.g. "device float myGlobalVar;"
            return f"{address_space} {metal_type} {var_node.name};"

    def _translate_local_variable(self, var_node: VariableNode, indent: int) -> str:
        """
        Translate a local variable declaration within a function or kernel.
        """
        pad = "  " * indent
        metal_type = self._cuda_type_to_metal(var_node.var_type)
        if var_node.is_pointer:
            # local pointer
            return f"{pad}{metal_type}* {var_node.name} = nullptr;"  # or f"thread {metal_type}*"
        else:
            return f"{pad}{metal_type} {var_node.name};"

    def _translate_shared_memory(self, shared: CUDASharedMemory, indent: int) -> str:
        """
        Translate a __shared__ memory declaration into Metal threadgroup array.
        """
        pad = "  " * indent
        metal_type = self._cuda_type_to_metal(shared.data_type)
        if shared.size is None:
            # dynamic shared memory
            # In Metal, you typically declare threadgroup arrays with a fixed size, or pass size at runtime
            size_str = "/* dynamic-sized: pass at runtime */"
        else:
            size_str = str(shared.size)

        return f"{pad}threadgroup {metal_type} s_{shared.name}[{size_str}];"

    # --------------------------------------------------------------------------
    # STATEMENTS & EXPRESSIONS
    # --------------------------------------------------------------------------
    def _translate_statement(self, stmt: CUDAStatement, indent: int) -> str:
        """
        Convert a CUDA statement node into MSL lines.
        This covers if/for/while/return etc. (in your advanced AST, these might appear).
        """
        pad = "  " * indent
        kind = stmt.kind.lower()

        if "if" in kind:
            code_lines = []
            code_lines.append(f"{pad}if ({self._translate_expression(stmt.condition, 0)}) {{")
            for c in stmt.then_branch:
                code_lines.append(self._translate_node(c, indent+1))
            code_lines.append(f"{pad}}}")
            if stmt.else_branch:
                code_lines.append(f"{pad}else {{")
                for c in stmt.else_branch:
                    code_lines.append(self._translate_node(c, indent+1))
                code_lines.append(f"{pad}}}")
            return "\n".join(code_lines)

        elif "for" in kind:
            code_lines = []
            init_str = self._translate_expression(stmt.init, 0) if stmt.init else ""
            cond_str = self._translate_expression(stmt.condition, 0) if stmt.condition else ""
            incr_str = self._translate_expression(stmt.increment, 0) if stmt.increment else ""
            code_lines.append(f"{pad}for ({init_str}; {cond_str}; {incr_str}) {{")
            for b in stmt.body:
                code_lines.append(self._translate_node(b, indent+1))
            code_lines.append(f"{pad}}}")
            return "\n".join(code_lines)

        elif "while" in kind:
            code_lines = []
            cond_str = self._translate_expression(stmt.condition, 0) if stmt.condition else "true"
            code_lines.append(f"{pad}while ({cond_str}) {{")
            for b in stmt.body:
                code_lines.append(self._translate_node(b, indent+1))
            code_lines.append(f"{pad}}}")
            return "\n".join(code_lines)

        elif "return" in kind:
            if stmt.expression:
                return f"{pad}return {self._translate_expression(stmt.expression, 0)};"
            else:
                return f"{pad}return;"

        elif "break" in kind:
            return f"{pad}break;"

        elif "continue" in kind:
            return f"{pad}continue;"

        elif "compound" in kind:
            # Just a block
            code_lines = []
            code_lines.append(f"{pad}{{")
            for c in stmt.body:
                code_lines.append(self._translate_node(c, indent+1))
            code_lines.append(f"{pad}}}")
            return "\n".join(code_lines)

        else:
            # Possibly an expression statement or unknown
            if stmt.expression:
                expr_code = self._translate_expression(stmt.expression, 0)
                return f"{pad}{expr_code};"
            else:
                return f"{pad}// Unknown statement type: {stmt.kind}"

    def _translate_expression(self, expr: CUDAExpressionNode, indent: int) -> str:
        """
        Convert a CUDA expression node into MSL code (inline).
        e.g., handle atomic, barrier, math intrinsics, block/thread indices, etc.
        """
        # We won't add indentation for single-line expressions, but we pass `indent` if needed.
        # We'll do a simpler approach: check operator, function, type of expression, etc.
        if expr.operator is not None:
            # e.g., binary operator
            left_code = self._translate_operand(expr.left)
            right_code = self._translate_operand(expr.right)
            return f"({left_code} {expr.operator} {right_code})"

        if expr.function:
            # e.g., a call to __syncthreads or __expf
            return self._translate_function_call(expr)

        # Possibly a single operand or reference
        if expr.operand:
            return self._translate_operand(expr.operand)

        if expr.spelling:
            # e.g. a variable or builtin reference
            return self._translate_builtin_or_var(expr.spelling)

        return "// [expr: unrecognized pattern]"

    def _translate_operand(self, operand: Optional[CUDAExpressionNode]) -> str:
        """Translate a sub-expression operand. If None, return empty."""
        if operand is None:
            return ""
        return self._translate_expression(operand, 0)

    def _translate_function_call(self, expr: CUDAExpressionNode) -> str:
        """
        Map CUDA builtins to Metal intrinsics, handle barrier or atomic calls, etc.
        """
        func_name = expr.function.lower()

        # Barrier example: __syncthreads
        if "syncthreads" in func_name:
            return "threadgroup_barrier(mem_flags::mem_threadgroup)"

        # Atomic example: atomicAdd, atomicSub, etc.
        if "atomic" in func_name:
            return self._translate_atomic_call(expr)

        # Common math intrinsics: __expf, __powf, etc.
        # Could map __expf(x) -> metal::fast::exp(x), etc.
        if func_name.startswith("__"):
            # remove the double underscore
            mapped_name = self._map_cuda_intrinsic_to_metal(func_name)
            # Then, join arguments
            arg_str = ", ".join(self._translate_expression(a, 0) for a in expr.arguments)
            return f"{mapped_name}({arg_str})"

        # Otherwise, treat as a normal function call
        arg_str = ", ".join(self._translate_expression(a, 0) for a in expr.arguments)
        return f"{expr.function}({arg_str})"

    def _translate_builtin_or_var(self, spelling: str) -> str:
        """
        For references like blockIdx.x, threadIdx.y, etc., convert to MSL thread references.
        Otherwise, assume it's a normal variable name.
        """
        # e.g. blockIdx.x -> gid.x or tg_id.x ...
        # threadIdx.x -> tid_in_group.x
        # We'll do some naive mapping:
        if spelling == "blockIdx.x":
            return "tg_id.x"
        elif spelling == "blockIdx.y":
            return "tg_id.y"
        elif spelling == "blockIdx.z":
            return "tg_id.z"
        elif spelling == "threadIdx.x":
            return "tid_in_group.x"
        elif spelling == "threadIdx.y":
            return "tid_in_group.y"
        elif spelling == "threadIdx.z":
            return "tid_in_group.z"
        elif spelling == "blockDim.x":
            return "tg_size.x"
        elif spelling == "blockDim.y":
            return "tg_size.y"
        elif spelling == "blockDim.z":
            return "tg_size.z"
        else:
            # Just a variable or unknown builtin
            return spelling

    def _translate_atomic_call(self, expr: CUDAExpressionNode) -> str:
        """Translate atomicAdd/atomicSub to Metal atomic_fetch_* calls."""
        if expr.function.lower() == "atomicadd":
            # Typically the first argument is pointer, second is value
            if len(expr.arguments) >= 2:
                ptr_code = self._translate_expression(expr.arguments[0], 0)
                val_code = self._translate_expression(expr.arguments[1], 0)
                return f"atomic_fetch_add_explicit({ptr_code}, {val_code}, memory_order_relaxed)"
        elif expr.function.lower() == "atomicsub":
            if len(expr.arguments) >= 2:
                ptr_code = self._translate_expression(expr.arguments[0], 0)
                val_code = self._translate_expression(expr.arguments[1], 0)
                # There's no direct atomic_sub in MSL, we can emulate via atomic_fetch_sub_explicit
                return f"atomic_fetch_sub_explicit({ptr_code}, {val_code}, memory_order_relaxed)"

        # If we don't handle it explicitly
        return f"// Unrecognized atomic call: {expr.function}"

    def _map_cuda_intrinsic_to_metal(self, func_name: str) -> str:
        """
        Map __expf, __sinf, etc. to their Metal equivalents.
        For demonstration, we do some common mappings.
        """
        if "__expf" in func_name:
            return "exp"
        if "__exp" in func_name:
            return "exp"
        if "__sinf" in func_name:
            return "sin"
        if "__sin" in func_name:
            return "sin"
        if "__cosf" in func_name:
            return "cos"
        if "__cos" in func_name:
            return "cos"
        if "__logf" in func_name:
            return "log"
        if "__log" in func_name:
            return "log"
        if "__powf" in func_name:
            return "pow"
        if "__pow" in func_name:
            return "pow"
        # fallback
        return func_name.strip("_")

    # --------------------------------------------------------------------------
    # GENERIC / DEFAULT
    # --------------------------------------------------------------------------
    def _translate_generic_node(self, node: CUDANode, indent: int) -> str:
        """
        Default fallback if no specialized translator is found.
        Could produce a comment or attempt to translate children.
        """
        pad = "  " * indent
        lines = []
        lines.append(f"{pad}// [Generic Node: {type(node).__name__}]")
        for child in node.children:
            lines.append(self._translate_node(child, indent=indent+1))
        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # UTILITY / TYPE MAPPING
    # --------------------------------------------------------------------------
    def _cuda_type_to_metal(self, cuda_type: CUDAType) -> str:
        """
        Map from your CUDAType enum to an MSL type string.
        Extend or refine as needed for vector types, half, etc.
        """
        # Quick map (not exhaustive):
        mapping = {
            CUDAType.VOID: "void",
            CUDAType.INT: "int",
            CUDAType.UINT: "uint",
            CUDAType.FLOAT: "float",
            CUDAType.DOUBLE: "double",
            CUDAType.CHAR: "char",
            CUDAType.USHORT: "ushort",
            # etc.
        }
        return mapping.get(cuda_type, "float")  # fallback

