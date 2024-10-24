import math
from typing import List, Dict, Any, Union
from ..utils.error_handler import CudaTranslationError
from ..utils.logger import get_logger
from ..parser.ast import (
    CudaASTNode, KernelNode, ForNode, BinaryOpNode, VarDeclNode,
    ExpressionNode, CompoundStmtNode, UnaryOpNode, IfStmtNode,
    CallExprNode, IntegerLiteralNode, FloatingLiteralNode, DeclRefExprNode
)

logger = get_logger(__name__)

class CodeOptimizer:
    def __init__(self):
        self.metal_simd_width = 32
        self.optimizations = [
            self.optimize_memory_access,
            self.optimize_thread_hierarchy,
            self.optimize_arithmetic_operations,
            self.optimize_control_flow,
            self.optimize_function_calls
        ]

    def optimize(self, ast: CudaASTNode) -> CudaASTNode:
        logger.info("Starting code optimization")
        for optimization in self.optimizations:
            ast = optimization(ast)
        return ast

    def optimize_memory_access(self, node: CudaASTNode) -> CudaASTNode:
        if isinstance(node, KernelNode):
            node = self._coalesce_global_memory_accesses(node)
            node = self._optimize_shared_memory_usage(node)
            node = self._use_texture_memory_for_readonly(node)

        for i, child in enumerate(node.children):
            node.children[i] = self.optimize_memory_access(child)

        return node

    def _coalesce_global_memory_accesses(self, node: KernelNode) -> KernelNode:
        for i, child in enumerate(node.children):
            if isinstance(child, ExpressionNode) and child.kind == 'ArraySubscript':
                index = child.children[1]
                if self._is_thread_index_based(index):
                    new_index = self._transform_to_coalesced_access(index)
                    child.children[1] = new_index
            node.children[i] = self._coalesce_global_memory_accesses(child)
        return node

    def _is_thread_index_based(self, node: ExpressionNode) -> bool:
        return isinstance(node, DeclRefExprNode) and 'threadIdx' in node.name

    def _transform_to_coalesced_access(self, index: ExpressionNode) -> ExpressionNode:
        return BinaryOpNode(
            op='+',
            left=DeclRefExprNode(name='threadIdx.x'),
            right=BinaryOpNode(
                op='*',
                left=DeclRefExprNode(name='blockDim.x'),
                right=DeclRefExprNode(name='blockIdx.x')
            )
        )

    def _optimize_shared_memory_usage(self, node: KernelNode) -> KernelNode:
        for i, child in enumerate(node.children):
            if isinstance(child, VarDeclNode) and 'shared' in child.qualifiers:
                if isinstance(child.initializer, IntegerLiteralNode):
                    optimal_size = self._calculate_optimal_shared_memory_size(int(child.initializer.value))
                    child.initializer.value = str(optimal_size)
            node.children[i] = self._optimize_shared_memory_usage(child)
        return node

    def _calculate_optimal_shared_memory_size(self, size: int) -> int:
        return 2 ** (size - 1).bit_length()

    def _use_texture_memory_for_readonly(self, node: KernelNode) -> KernelNode:
        for i, child in enumerate(node.children):
            if isinstance(child, VarDeclNode) and child.is_readonly() and self._is_suitable_for_texture_memory(child):
                child.data_type = 'texture2d<float>'
                child.qualifiers.append('[[texture(0)]]')
            node.children[i] = self._use_texture_memory_for_readonly(child)
        return node

    def _is_suitable_for_texture_memory(self, node: VarDeclNode) -> bool:
        return node.data_type in ['float', 'int'] and node.dimensions == 2

    def optimize_thread_hierarchy(self, node: CudaASTNode) -> CudaASTNode:
        if isinstance(node, KernelNode):
            node = self._optimize_thread_block_size(node)
            node = self._optimize_grid_size(node)

        for i, child in enumerate(node.children):
            node.children[i] = self.optimize_thread_hierarchy(child)

        return node

    def _optimize_thread_block_size(self, node: KernelNode) -> KernelNode:
        if hasattr(node, 'launch_config') and 'blockDim' in node.launch_config:
            block_dim = node.launch_config['blockDim']
            optimized_block_dim = [((dim + self.metal_simd_width - 1) // self.metal_simd_width) * self.metal_simd_width for dim in block_dim]
            node.launch_config['blockDim'] = optimized_block_dim
        return node

    def _optimize_grid_size(self, node: KernelNode) -> KernelNode:
        if hasattr(node, 'launch_config') and 'gridDim' in node.launch_config and 'blockDim' in node.launch_config:
            grid_dim = node.launch_config['gridDim']
            block_dim = node.launch_config['blockDim']
            optimized_grid_dim = [(grid_dim[i] * block_dim[i] + block_dim[i] - 1) // block_dim[i] for i in range(3)]
            node.launch_config['gridDim'] = optimized_grid_dim
        return node

    def optimize_arithmetic_operations(self, node: CudaASTNode) -> CudaASTNode:
        if isinstance(node, ExpressionNode):
            node = self._use_fast_math(node)
            node = self._optimize_vector_operations(node)

        for i, child in enumerate(node.children):
            node.children[i] = self.optimize_arithmetic_operations(child)

        return node

    def _use_fast_math(self, node: ExpressionNode) -> ExpressionNode:
        fast_math_functions = {
            'sin': 'metal::fast::sin',
            'cos': 'metal::fast::cos',
            'exp': 'metal::fast::exp',
            'log': 'metal::fast::log',
            'pow': 'metal::fast::pow'
        }
        if isinstance(node, CallExprNode) and node.name in fast_math_functions:
            node.name = fast_math_functions[node.name]
        return node

    def _optimize_vector_operations(self, node: ExpressionNode) -> ExpressionNode:
        if isinstance(node, BinaryOpNode) and self._are_vector_operands(node.left, node.right):
            return self._create_vector_operation(node)
        return node

    def _are_vector_operands(self, left: ExpressionNode, right: ExpressionNode) -> bool:
        return hasattr(left, 'type') and hasattr(right, 'type') and left.type.startswith('float4') and right.type.startswith('float4')

    def _create_vector_operation(self, node: BinaryOpNode) -> CallExprNode:
        return CallExprNode(
            name=f'metal::{''.join(node.op)}',
            args=[node.left, node.right],
            type='float4'
        )

    def optimize_control_flow(self, node: CudaASTNode) -> CudaASTNode:
        if isinstance(node, ForNode):
            node = self._unroll_loops(node)
        elif isinstance(node, KernelNode):
            node = self._optimize_branching(node)

        for i, child in enumerate(node.children):
            node.children[i] = self.optimize_control_flow(child)

        return node

    def _unroll_loops(self, node: ForNode) -> Union[ForNode, CompoundStmtNode]:
        if self._is_unrollable(node):
            return self._create_unrolled_loop(node)
        return node

    def _is_unrollable(self, node: ForNode) -> bool:
        return (isinstance(node.init, VarDeclNode) and
                isinstance(node.condition, BinaryOpNode) and
                isinstance(node.increment, UnaryOpNode) and
                self._get_loop_iteration_count(node) <= 8)

    def _get_loop_iteration_count(self, node: ForNode) -> int:
        start = int(node.init.initializer.value)
        end = int(node.condition.right.value)
        return end - start

    def _create_unrolled_loop(self, node: ForNode) -> CompoundStmtNode:
        unrolled_statements = []
        iteration_count = self._get_loop_iteration_count(node)
        for i in range(iteration_count):
            unrolled_statements.extend(self._replace_loop_variable(node.body, node.init.name, i))
        return CompoundStmtNode(children=unrolled_statements)

    def _replace_loop_variable(self, body: List[CudaASTNode], var_name: str, iteration: int) -> List[CudaASTNode]:
        return [self._replace_variable(stmt, var_name, str(iteration)) for stmt in body]

    def _replace_variable(self, node: CudaASTNode, var_name: str, replacement: str) -> CudaASTNode:
        if isinstance(node, DeclRefExprNode) and node.name == var_name:
            return IntegerLiteralNode(value=replacement)
        for i, child in enumerate(node.children):
            node.children[i] = self._replace_variable(child, var_name, replacement)
        return node

    def _optimize_branching(self, node: KernelNode) -> KernelNode:
        node = self._convert_if_to_ternary(node)
        node = self._hoist_invariant_code(node)
        return node

    def _convert_if_to_ternary(self, node: KernelNode) -> KernelNode:
        for i, child in enumerate(node.children):
            if isinstance(child, IfStmtNode) and self._is_simple_if(child):
                node.children[i] = self._create_ternary_operator(child)
            else:
                node.children[i] = self._convert_if_to_ternary(child)
        return node

    def _is_simple_if(self, node: IfStmtNode) -> bool:
        return (len(node.then_branch.children) == 1 and
                isinstance(node.then_branch.children[0], ExpressionNode) and
                (node.else_branch is None or
                 (len(node.else_branch.children) == 1 and
                  isinstance(node.else_branch.children[0], ExpressionNode))))

    def _create_ternary_operator(self, node: IfStmtNode) -> CallExprNode:
        return CallExprNode(
            name='select',
            args=[
                node.condition,
                node.then_branch.children[0],
                node.else_branch.children[0] if node.else_branch else ExpressionNode(kind='NullExpr')
            ]
        )

    def _hoist_invariant_code(self, node: KernelNode) -> KernelNode:
        invariant_code = []
        non_invariant_code = []
        for child in node.children:
            if self._is_loop_invariant(child):
                invariant_code.append(child)
            else:
                non_invariant_code.append(child)
        node.children = non_invariant_code
        return CompoundStmtNode(children=invariant_code + [node])

    def _is_loop_invariant(self, node: CudaASTNode) -> bool:
        return not self._contains_thread_id(node)

    def _contains_thread_id(self, node: CudaASTNode) -> bool:
        if isinstance(node, DeclRefExprNode) and node.name in ['threadIdx', 'blockIdx']:
            return True
        return any(self._contains_thread_id(child) for child in node.children)

    def optimize_function_calls(self, node: CudaASTNode) -> CudaASTNode:
        if isinstance(node, CallExprNode):
            node = self._inline_small_functions(node)
            node = self._optimize_math_functions(node)

        for i, child in enumerate(node.children):
            node.children[i] = self.optimize_function_calls(child)

        return node

    def _inline_small_functions(self, node: CallExprNode) -> CudaASTNode:
        small_functions = {
            'min': lambda x, y: BinaryOpNode(op='<', left=x, right=y),
            'max': lambda x, y: BinaryOpNode(op='>', left=x, right=y),
            'abs': lambda x: CallExprNode(name='metal::abs', args=[x]),
        }

        if node.name in small_functions:
            return small_functions[node.name](*node.args)
        return node

    def _optimize_math_functions(self, node: CallExprNode) -> CallExprNode:
        math_optimizations = {
            'pow': self._optimize_pow,
            'exp': self._optimize_exp,
            'log': self._optimize_log
        }
        if node.name in math_optimizations:
            return math_optimizations[node.name](node)
        return node

    def _optimize_pow(self, node: CallExprNode) -> ExpressionNode:
        if len(node.args) == 2 and isinstance(node.args[1], IntegerLiteralNode):
            exponent = int(node.args[1].value)
            if exponent == 2:
                return BinaryOpNode(
                    op='*',
                    left=node.args[0],
                    right=node.args[0]
                )
            elif exponent == 3:
                return BinaryOpNode(
                    op='*',
                    left=node.args[0],
                    right=BinaryOpNode(
                        op='*',
                        left=node.args[0],
                        right=node.args[0]
                    )
                )
        return CallExprNode(name='metal::pow', args=node.args)

    def _optimize_exp(self, node: CallExprNode) -> CallExprNode:
        return CallExprNode(name='metal::exp', args=node.args)

    def _optimize_log(self, node: CallExprNode) -> CallExprNode:
        return CallExprNode(name='metal::log', args=node.args)

logger.info("CodeOptimizer initialized for Metal-specific code optimizations.")