from typing import List, Dict, Any
from ..parser.ast import CudaASTNode, KernelNode, FunctionNode, VariableNode, ExpressionNode
from ..utils.cuda_to_metal_type_mapping import map_cuda_type_to_metal
from ..utils.metal_equivalents import translate_cuda_call_to_metal, get_metal_equivalent
from ..utils.error_handler import CudaTranslationError
from ..utils.logger import get_logger

logger = get_logger(__name__)

class KernelTranslator:
    def __init__(self):
        self.current_kernel: KernelNode = None
        self.metal_code: List[str] = []
        self.indent_level: int = 0
        self.variable_mappings: Dict[str, str] = {}

    def translate_kernel(self, kernel: KernelNode) -> str:
        self.current_kernel = kernel
        self.metal_code = []
        self.indent_level = 0
        self.variable_mappings = {}

        try:
            self._add_line(self._generate_kernel_signature(kernel))
            self._add_line("{")
            self.indent_level += 1

            self._translate_kernel_body(kernel.body)

            self.indent_level -= 1
            self._add_line("}")

            return "\n".join(self.metal_code)
        except Exception as e:
            logger.error(f"Error translating kernel {kernel.name}: {str(e)}")
            raise CudaTranslationError(f"Failed to translate kernel {kernel.name}", str(e))

    def _generate_kernel_signature(self, kernel: KernelNode) -> str:
        metal_params = []
        for i, param in enumerate(kernel.parameters):
            metal_type = map_cuda_type_to_metal(param.data_type)
            metal_params.append(f"{metal_type} {param.name} [[buffer({i})]]")

        return f"kernel void {kernel.name}({', '.join(metal_params)})"

    def _translate_kernel_body(self, body: List[CudaASTNode]):
        for node in body:
            self._translate_node(node)

    def _translate_node(self, node: CudaASTNode):
        if isinstance(node, VariableNode):
            self._translate_variable_declaration(node)
        elif isinstance(node, FunctionNode):
            self._translate_function_call(node)
        elif isinstance(node, ExpressionNode):
            self._translate_expression(node)
        else:
            self._add_line(f"// TODO: Translate {node.__class__.__name__}")

    def _translate_variable_declaration(self, node: VariableNode):
        metal_type = map_cuda_type_to_metal(node.data_type)
        initializer = f" = {self._translate_expression(node.initializer)}" if node.initializer else ""
        self._add_line(f"{metal_type} {node.name}{initializer};")
        self.variable_mappings[node.name] = metal_type

    def _translate_function_call(self, node: FunctionNode):
        metal_function = translate_cuda_call_to_metal(node.name, [self._translate_expression(arg) for arg in node.arguments])
        self._add_line(f"{metal_function};")

    def _translate_expression(self, node: ExpressionNode) -> str:
        if isinstance(node, VariableNode):
            return node.name
        elif isinstance(node, FunctionNode):
            return translate_cuda_call_to_metal(node.name, [self._translate_expression(arg) for arg in node.arguments])
        elif node.kind == 'BinaryOperator':
            left = self._translate_expression(node.left)
            right = self._translate_expression(node.right)
            return f"({left} {node.operator} {right})"
        elif node.kind == 'Literal':
            return node.value
        else:
            logger.warning(f"Unhandled expression type: {node.__class__.__name__}")
            return f"/* TODO: Translate {node.__class__.__name__} */"

    def _add_line(self, line: str):
        self.metal_code.append("    " * self.indent_level + line)

    def _translate_atomic_operation(self, node: FunctionNode):
        metal_equivalent = get_metal_equivalent(node.name)
        if metal_equivalent.requires_custom_implementation:
            self._add_line(f"// TODO: Implement custom atomic operation for {node.name}")
            self._add_line(f"// {metal_equivalent.metal_function}({', '.join(node.arguments)});")
        else:
            translated_args = [self._translate_expression(arg) for arg in node.arguments]
            metal_function = translate_cuda_call_to_metal(node.name, translated_args)
            self._add_line(f"{metal_function};")

    def _translate_texture_operation(self, node: FunctionNode):
        # This is a placeholder for texture operation translation
        # Actual implementation would depend on the specific texture functions used
        self._add_line(f"// TODO: Implement texture operation translation for {node.name}")
        self._add_line(f"// {node.name}({', '.join(str(arg) for arg in node.arguments)});")

    def _translate_control_flow(self, node: CudaASTNode):
        if node.kind == 'IfStatement':
            condition = self._translate_expression(node.condition)
            self._add_line(f"if ({condition}) {{")
            self.indent_level += 1
            self._translate_kernel_body(node.then_branch)
            self.indent_level -= 1
            self._add_line("}")
            if node.else_branch:
                self._add_line("else {")
                self.indent_level += 1
                self._translate_kernel_body(node.else_branch)
                self.indent_level -= 1
                self._add_line("}")
        elif node.kind == 'ForStatement':
            init = self._translate_expression(node.init)
            condition = self._translate_expression(node.condition)
            increment = self._translate_expression(node.increment)
            self._add_line(f"for ({init}; {condition}; {increment}) {{")
            self.indent_level += 1
            self._translate_kernel_body(node.body)
            self.indent_level -= 1
            self._add_line("}")
        elif node.kind == 'WhileStatement':
            condition = self._translate_expression(node.condition)
            self._add_line(f"while ({condition}) {{")
            self.indent_level += 1
            self._translate_kernel_body(node.body)
            self.indent_level -= 1
            self._add_line("}")
        else:
            logger.warning(f"Unhandled control flow type: {node.kind}")
            self._add_line(f"// TODO: Translate {node.kind}")

    def _translate_memory_operation(self, node: FunctionNode):
        if node.name == 'cudaMalloc':
            self._add_line(f"// TODO: Implement cudaMalloc equivalent")
            self._add_line(f"// device.makeBuffer(length: {node.arguments[1]}, options: []);")
        elif node.name == 'cudaFree':
            self._add_line(f"// Note: Metal handles deallocation automatically")
            self._add_line(f"// No direct equivalent for cudaFree({node.arguments[0]})")
        elif node.name == 'cudaMemcpy':
            self._add_line(f"// TODO: Implement cudaMemcpy equivalent")
            self._add_line(f"// memcpy({node.arguments[0]}, {node.arguments[1]}, {node.arguments[2]});")
        else:
            logger.warning(f"Unhandled memory operation: {node.name}")
            self._add_line(f"// TODO: Translate {node.name}")

