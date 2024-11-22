import numpy as np
import sys
import argparse
import yaml

class PatternParser:
    """Read Pattern Description file (e.g. pattern.yml) to generate logical plan"""

    def parse(self, filename):
        try:
            with open(filename, 'r') as file:
                pattern = yaml.safe_load(file)['pattern']
                edges = list(map(lambda x: (x[0], x[1]), pattern.get('edges', None)))
                order = list(pattern.get('order', None))
                orientation = list(map(lambda x: (x[0], x[1]), pattern.get('orientation', None)))
                A, res = self.build_adjacency_matrix(edges, order, orientation)
                self.generate_logical_plan(A)
                return A, res
        except Exception as e:
            return f"An error occurred: {e}" 
        
    def build_adjacency_matrix(self, edges, order, res):
        node_index = {node: idx for idx, node in enumerate(order)}
        size = len(order)
        adjacency_matrix = np.zeros((size, size), dtype=int)
    
        for node1, node2 in edges:
            idx1 = node_index[node1]
            idx2 = node_index[node2]
            adjacency_matrix[idx1][idx2] = 1
            adjacency_matrix[idx2][idx1] = 1  # Assuming the graph is undirected
    
        res_idx = {}
        if res:
            for x, y in res:
                # TODO: how to remove this?
                if int(node_index[y]) > int(node_index[x]):
                    res_idx[node_index[y]] = node_index[x]
                else:
                    res_idx[node_index[x]] = node_index[y]
    
        print("> Adjacency matrix of pattern:")
        print(adjacency_matrix)
        print(f"{res_idx=}")
        return adjacency_matrix, res_idx
    
    def generate_logical_plan(self, A):
        pattern_size = len(A)
        indent=""
        print()
        print("> Logical plan:")
        for v in range(pattern_size):
            S = "V"
            for u in range(pattern_size):
                if u >= v:
                    continue
                if A[u][v]:
                    S += f"^N(v{u})"
                else:
                    S += f"-N(v{u})"
            print(f"{indent}for v{v} in {S}:")
            indent += "  "
        print (f"{indent}counter ++")

class ID:
    def __init__(self, value, fmt=None):
        self.format = fmt
        self.value = value

    def __repr__(self):
        if self.format:
            return self.format + str(self.value)
        else:
            return str(self.value)

    def fmt(self):
        return self.format

    def id(self):
        return self.value

class BufferAllocator:
    """ 
    Allocate buffer for intermediate result (IMR), there are two types of IMR:
      - precomputed prefix (P): when prefix is no longer needed, it can be free
      - next candidates (S): can never be freed.
    IMR has two types of format:
      - bitmap (B): require `(max_deg / BITMAP_WIDTH)` vidType, use bitmap_id like B0, B1, ...
      - array (A): require `max_deg` vidType, use slot_id like S0, S1, ...
    """
    def __init__(self):
        # slot_stack[i] stores the usage of slot_i, bitmap_stack does the same thing
        # if max(slot_stack[i]) < current level, we can safely free this slot
        self.slot_stack = [] # make sure pass it by ref
        self.bitmap_stack = [] # make sure pass it by ref
        self.map = {} # pattern -> slot_id/bitmap_id

    def max_slot_num(self):
        return len(self.slot_stack)

    def max_bitmap_num(self):
        return len(self.bitmap_stack)

    def reuse(self, fmt, cur_lvl):
        def scan(stacks, lvl):
            for i in range(len(stacks)):
                stack = stacks[i]
                # mono-stack
                while len(stack) > 0:
                    # this buffer is no-longer needed
                    if(stack[-1] < lvl):
                        stack.pop()
                    else:
                        break
                if len(stack) == 0:
                    # this slot can be reused.
                    return i
        if fmt == "B":
            idx = scan(self.bitmap_stack, cur_lvl)
            return ID(idx, fmt=fmt) if idx else None
        elif fmt == "A":
            idx = scan(self.slot_stack, cur_lvl)
            return ID(idx, fmt=fmt) if idx else None
        return None

    def add_dependency(self, buf_id, lookup_lvl):
        if buf_id.fmt() == "B":
            self.bitmap_stack[buf_id.id()].insert(0, lookup_lvl) # insert to head
        elif buf_id.fmt() == "A":
            self.slot_stack[buf_id.id()].insert(0, lookup_lvl) 

    def get_empty_buffer(self, fmt, cur_lvl, lookup_lvl):
        if fmt == "B":
            bitmap_id = self.reuse(fmt, cur_lvl)
            if bitmap_id == None:
                bitmap_id = ID(self.max_bitmap_num(), fmt=fmt)
                self.bitmap_stack.append([])
            self.add_dependency(bitmap_id, lookup_lvl)
            return bitmap_id
        elif fmt == "A":
            slot_id = self.reuse(fmt, cur_lvl)
            if slot_id == None:
                slot_id = ID(self.max_slot_num(), fmt=fmt)
                self.slot_stack.append([])
            self.add_dependency(slot_id, lookup_lvl)
            return slot_id
        
    def set_mapping(self, out_pat, x_id):
        self.map[out_pat] = x_id

    def del_mapping(self, out_pat):
        self.map.pop(out_pat, None)

    def update_mapping(self, old_pat, new_pat):
        x_id = self.map.get(old_pat, None)
        if x_id:
            self.map[new_pat] = x_id
        self.del_mapping(old_pat)

    def require(self, fmt, out_pat, cur_lvl, lookup_lvl):
        """ out_pat is "0001:B" or "0001:I" or "0001:V"  """
        # Required intermediate result is already in cache
        if (out_pat in self.map.keys()) and self.map[out_pat].id() != -1:
            self.add_dependency(self.map[out_pat], lookup_lvl)
            return False # without materialization
        else:
            if all(c=='0' for c in out_pat.split(":")[0]):
                # if all operation is difference, enable lazy materialization
                self.set_mapping(out_pat, ID(-1))
                return False # without materialization
            else:
                # otherwisze, materialize it immediately
                buf_id = self.get_empty_buffer(fmt, cur_lvl, lookup_lvl)
                self.set_mapping(out_pat, buf_id)
                return True

class Constant:
    def __init__(self, name):
        self.type = "Constant"
        self.name = name

    def __repr__(self):
        return f"{self.name}"

    def code(self):
        return f"{self.name}"

# basic codegen
class SetBuilder:
    """ This is a unary OP that build a set, S = op(operand) """
    def __init__(self, operation, operand, connected=True, ex_parameter=ID(-1)):
        self.type = "SetBuilder"
        self.operation =  operation# str
        assert(operand != None)
        self.operand = operand.id() # str
        self.connected = connected
        self.ex_parameter = ex_parameter.id()
        self.operation_map = {
                "N" : "__get_vlist_from_graph",
                "#" : "__get_vlist_from_heap",
                "L" : "__get_vmap_from_lut",
                "@" : "__get_vmap_from_heap"
                }
        self.common_params = ["g",      # GPUGraph&
                              "meta",   # StorageMeta&
                              ]

    def __repr__(self):
        if self.operation == "N":
            return f"{self.operation}(v{self.operand})"
        elif self.operation == "#":
            operand = self.operand if self.operand >=0 else "ALL_VERTEX"
            return f"{self.operation}{operand}"
        elif self.operation == "L":
            connected = "nbr" if self.connected else "non-nbr"
            return f"{self.operation}(v{self.operand}, {connected})"
        elif self.operation == "@":
            return f"{self.operation}({self.operand}, {self.ex_parameter})"
        else:
            assert("Unrecognized set builder operation.")

    def code(self):
        operation = self.operation_map[self.operation]
        common = ", ".join(self.common_params)
        operand = ""
        if self.operation == "N":
            operand = f"/*vid=*/v{self.operand}"
            return f"{operation}({common}, {operand})"
        elif self.operation == "#":
            operand = f"/*slot_id=*/{self.operand}"
            return f"{operation}({common}, {operand})"
        elif self.operation == "L":
            connected = "true" if self.connected else "false"
            upper_bound = "v"+str(self.ex_parameter) if self.ex_parameter != -1 else -1
            operand = f"/*idx_id=*/v{self.operand}, /*connected=*/{connected}, /*upper_bound=*/{upper_bound}"
            if type(upper_bound)==str and upper_bound[-3:] != "idx":
                return f"{operation}_vid_limit({common}, {operand})"
            else:
                return f"{operation}({common}, {operand})"
        elif self.operation == "@":
            operand = f"/*bitmap_id=*/{self.operand}, /*slot_id=*/{self.ex_parameter}"
            return f"{operation}({common}, {operand})"

class SetOperation:
    def __init__(self, operation, left_operand, right_operand, output, in_pattern=None, out_pattern=None):
        self.type = "SetOperation"
        self.operation = operation
        self.in_pattern = in_pattern
        self.out_pattern = out_pattern
        self.left_operand = left_operand
        self.right_operand = right_operand
        self.upper_bound = -1
        self.output = output
        self.operation_map = {
                "difference": "__difference",
                "intersect" : "__intersect",
                "cnt_intersect" : "__intersect_num",
                "cnt_difference" :  "__difference_num"
                }
        self.common_params = [
                   "meta",      # VertexArrayMeta&
                ]

    def set_upper_bound(self, upper_bound):
        self.upper_bound = upper_bound

    def __repr__(self):
        if self.operation.startswith("cnt_"):
            if self.right_operand != None:
                return f"count += {self.operation}({self.left_operand}, {self.right_operand}) | ({self.in_pattern}->{self.out_pattern})"
            else:
                return f"count += {self.operation}({self.left_operand}) | ({self.in_pattern}->{self.out_pattern})"
        else:
            return f"#{self.output} = {self.operation}({self.left_operand}, {self.right_operand}) | ({self.in_pattern}->{self.out_pattern})"

    def code(self):
        upper_bound = self.upper_bound
        if self.upper_bound != -1:
            upper_bound = f"v{self.upper_bound}"
        if self.operation.startswith("cnt_"):
            if self.right_operand != None:
                return f"count += {self.operation_map[self.operation]}({self.left_operand.code()}, {self.right_operand.code()}, /*upper_bound=*/{upper_bound});"
            else:
                # for counting one row of vmap
                return f"count += {self.operation_map[self.operation]}({self.left_operand.code()}, /*upper_bound=*/{upper_bound});"
        else:
            common = ", ".join(self.common_params)
            return f"{self.operation_map[self.operation]}({common}, {self.left_operand.code()}, {self.right_operand.code()}, /*upper_bound=*/{upper_bound}, /*output_slot=*/{self.output});"

class ForLoop:
    def __init__(self, loop_type, vertex, vertex_set, block, indent, loop_pattern="None"):
        self.type = "ForLoop"
        self.loop_type = loop_type
        self.loop_pattern = loop_pattern
        self.vertex = vertex
        self.vertex_set = vertex_set
        self.block = block
        self.indent = indent

    def __repr__(self):
        if self.loop_type != "inter_warp_edge_parallel_first":
            return f"for v{self.vertex} in {self.vertex_set}: | ({self.loop_pattern})\n{self.block}"
        else:
            return f"for (v0, v1) in #ALL_EDGES: | (:E)\n{self.block}"

    def code(self):
        for_statement = ""
        if self.loop_type == "inter_warp_parallel":
            for_statement = f"auto candidate_v{self.vertex} = {self.vertex_set.code()};\n{self.indent}for(vidType v{self.vertex}_idx = warp_id; v{self.vertex}_idx < candidate_v{self.vertex}.size(); v{self.vertex}_idx += num_warps){{\n{self.indent}  auto v{self.vertex} = candidate_v{self.vertex}[v{self.vertex}_idx];\n"
        elif self.loop_type == "serial":
            for_statement = f"auto candidate_v{self.vertex} = {self.vertex_set.code()};\n{self.indent}for(vidType v{self.vertex}_idx = 0; v{self.vertex}_idx < candidate_v{self.vertex}.size(); v{self.vertex}_idx ++){{\n{self.indent}  auto v{self.vertex} = candidate_v{self.vertex}[v{self.vertex}_idx];\n"
        elif self.loop_type == "intra_warp_parallel":
            for_statement = f"auto candidate_v{self.vertex} = {self.vertex_set.code()};\n{self.indent}for(vidType v{self.vertex}_idx = thread_lane; v{self.vertex}_idx < candidate_v{self.vertex}.size(); v{self.vertex}_idx += WARP_SIZE){{\n{self.indent}  auto v{self.vertex} = candidate_v{self.vertex}[v{self.vertex}_idx];\n"
        elif self.loop_type == "inter_warp_edge_parallel_first":
            for_statement = f"for(vidType e01_idx = warp_id; e01_idx < g.E(); e01_idx += num_warps){{\n{self.indent}  auto v0_idx = g.get_src(e01_idx);\n{self.indent}  auto v0 = v0_idx;\n{self.indent}  auto v1 = g.get_dst(e01_idx);\n{self.indent}  auto v1_idx = g.get_dst_ptr(e01_idx) - g.N(v0);\n"
        else:
            assert("Unknown loop type.")
        return for_statement + self.block.code() + f"\n{self.indent}}}"

class Block:
    """ Nested loop body """
    def __init__(self, indent):
        self.type = "Block"
        self.statements = []
        self.indent = indent

    def remove_nop(self, S):
        return [x for x in S if x.type != "NOP"]

    def __repr__(self):
        return "\n".join(map(lambda x: self.indent + str(x), self.remove_nop(self.statements)))

    def code(self):
        return "\n".join(map(lambda x: self.indent + x.code(), self.remove_nop(self.statements)))

    def size(self):
        return len(self.statements)

    def get_next_loop_idx(self):
        for i, s in enumerate(self.statements):
            if s.type == "ForLoop" or s.type == "FusedForLoop":
                return i
        return None

    def get_next_loop(self):
        for i, s in enumerate(self.statements):
            if s.type == "ForLoop" or s.type == "FusedForLoop":
                return s
        return None

    def get_statement(self, idx):
        return self.statements[idx]

    def insert_statement(self, idx, statement):
        self.statements.insert(idx, statement)

    def replace_statement(self, idx, statement):
        self.statements[idx] = statement

    def add_statement(self, statement):
        self.statements.append(statement)

    def slice(self, start, end):
        new_block = Block(self.indent)
        for i in range(start, end):
            new_block.add_statement(self, self.statements[i])
        return new_block

    def merge(self, blk):
        for s in blk.statements:
            self.add_statement(s)

class Instruction:
    """ 
    Instructions to optimize the plan, they are optional, take cost but bring benefit,
    for example:
      _ build_LUT
      - build_index_from_vmap
      _ build_vlist_from_vmap
      _ build_bitmap_from_vmap
    Optimizer will insert these instructions at proper location.
    """
    def __init__(self, operation, operand, buffer_id=ID(-1)):
        self.type = "Instruction"
        self.operation =  operation # str
        assert(operand != None)
        self.operand = operand # str
        self.buffer_id = buffer_id.id()
        self.operation_map = {
                ".LUT" : "__build_LUT",
                ".I"   : "__build_index_from_vmap",
                ".V"   : "__build_vlist_from_vmap",
                ".B"   : "__build_bitmap_from_vmap",
                "vid"  : "__build_vid_from_vidx"
                }
        self.common_params = ["g",      # GPUGraph&
                              "meta",   # StorageMeta&
                              ]

    def __repr__(self):
        if self.operation == ".LUT":
            return f"{self.operand}{self.operation}()"
        elif self.operation == ".I":
            return f"#{self.buffer_id} = {self.operand}{self.operation}()"
        elif self.operation == ".V":
            return f"#{self.buffer_id} = {self.operand}{self.operation}()"
        elif self.operation == ".B":
            return f"@{self.buffer_id} = {self.operand}{self.operation}()"
        elif self.operation == "vid":
            return f"v{str(self.operand).split("_")[0]} = {self.operation}(v{self.operand})"

    def code(self):
        operation = self.operation_map[self.operation]
        common = ", ".join(self.common_params)
        operand = ""
        if self.operation == ".LUT":
            operand = f"{self.operand.code()}"
            return f"{operation}({common}, {operand});"
        elif self.operation == ".I":
            operand = f"{self.operand.code()}, /*slot_id=*/{self.buffer_id}"
            return f"{operation}({common}, {operand});"
        elif self.operation == ".V":
            operand = f"{self.operand.code()}, /*slot_id=*/{self.buffer_id}"
            return f"{operation}({common}, {operand});"
        elif self.operation == ".B":
            operand = f"{self.operand.code()}, /*bitmap_id=*/{self.buffer_id}"
            return f"{operation}({common}, {operand});"
        elif self.operation == "vid":
            return f"auto v{str(self.operand).split("_")[0]} = {operation}({common}, v{self.operand.code()});"

class NOP:
    def __init__(self):
        self.type = "NOP"

    def __repr__(self):
        return ""

    def code(self):
        return ""

class FusedForLoop:
    def __init__(self, instruction, forloop):
        self.type = "FusedForLoop"
        self.loop_type = forloop.loop_type
        self.loop_pattern = forloop.loop_pattern[:-1]+"I"
        self.vertex = forloop.vertex
        self.instruction = instruction
        self.vertex_set = forloop.vertex_set
        self.block = forloop.block
        self.indent = forloop.indent

    def __repr__(self):
        return f"ffor v{self.vertex}_idx in {self.instruction}: | {self.loop_pattern}\n{self.block}"

    def code(self):
        for_statement = ""
        start = ""
        step = ""
        if self.loop_type == "inter_warp_parallel":
            start = "warp_id"
            step = "num_warps"
        elif self.loop_type == "serial":
            start = "0"
            step = "1"
        elif self.loop_type == "intra_warp_parallel":
            start = "thread_lane"
            step = "WARP_SIZE"
        else:
            assert("Unknown loop type.")
        get_candidate = f"auto candidate_v{self.vertex} = {self.vertex_set.code()};\n"
        build_lut = f"{self.indent}{self.instruction.code()};\n"
        for_statement = f"{self.indent}for (vidType v{self.vertex}_idx = {start}; v{self.vertex}_idx < candidate_v{self.vertex}.size(); v{self.vertex}_idx += {step}) {{\n"
        return get_candidate + build_lut + for_statement + self.block.code() + f"\n{self.indent}}}"

class LoweringPass:
    def InputSet(self, A, i, v, tag):
        return "".join(map(lambda x: str(x), A[i][:v])) + ":" + str(tag)

    def OutputSet(self, A, i, v, tag):
        return "".join(map(lambda x: str(x), A[i][:v+1])) + ":" + str(tag)

    def lower(self, A, O):
        allocator = BufferAllocator()
        pattern_size = len(A)
    
        def Loop(v, tag, indent):
            """
            First Level: inter warp parallel
            Last Level: 
               if vmap + vmap operation: intra wrap parallel
               else: serial
            Others: serial
            """
            loop_pattern = self.InputSet(A, v, v, tag)
            loop_set = allocator.map.get(loop_pattern, None) # get candidate
            loop_type = "inter_warp_parallel" if v == 0 else ("intra_warp_parallel" if v == pattern_size-1 else "serial")
            read_vlist_op = SetBuilder("#", ID(-1)) if v == 0 else (SetBuilder("N", ID(0)) if v == 1 else SetBuilder("#", loop_set))
            loop_block = Block(indent+"  ")
    
            # Analyze data dependency to generate vlist beforehand
            # Cache candidate prefix of v_i.
            for i in range(v+1, pattern_size):
                # if this is candidate, we should always materialize it since we will do nested loop on it. (require_lvl=MAX_INT)
                # if this is a precomputed prefix of a candidate, we can free it after it's usage is done. (require_lvl=i)
                require_lvl = sys.maxsize if i == v+1 else i
                upper_bound_key = i if i == v+1 else -1
                if v == 0: 
                    break
                in_pat = self.InputSet(A, i, v, tag)
                out_pat = self.OutputSet(A, i, v, tag)
                # we will try to materialize (in_pat -> out_pat) here, which will use N(v), add used in iteration i.
                # print(f"DEBUG:{v=}, {i=}, {in_pat=}, {out_pat}, {allocator.map=}")
    
                def add_set_op(operation, lhs, rhs, in_pat, out_pat, do_count=True, current=0, depend_by=sys.maxsize):
                    upper_bound = O.get(current, -1)
                    if do_count:
                        operation = "cnt_" + operation
                        output = Constant("nullptr")
                        set_op = SetOperation(operation, lhs, rhs, output, in_pattern = in_pat, out_pattern = out_pat)
                        set_op.set_upper_bound(upper_bound)
                        loop_block.add_statement(set_op)
                        return ID(-1)
                    else:
                        if allocator.require('A', out_pat, v, depend_by):
                            out_id = allocator.map[out_pat]
                            output = Constant(out_id.id())
                            set_op = SetOperation(operation, lhs, rhs, output, in_pattern = in_pat, out_pattern = out_pat)
                            set_op.set_upper_bound(upper_bound)
                            loop_block.add_statement(set_op)
                            return out_id
                        return None
    
                # Check whether we should cache the result
                in_slot = allocator.map.get(in_pat, None)
                if in_slot == None: 
                    operation = "intersect" if A[i][v] else "difference"
                    lhs = SetBuilder("N", ID(0))
                    rhs = SetBuilder("N", ID(1))
                    if out_pat.split(":")[0] == "01":
                        operation = "difference"
                        lhs, rhs = rhs, lhs
                        in_pat = "-1" +":V"
                        out_pat = "01" + ":V"
                    add_set_op(operation, lhs, rhs, in_pat, out_pat, do_count=(v==pattern_size-2), current=upper_bound_key, depend_by=require_lvl)
                elif in_slot.id() < 0: # means all operation before are difference
                    if A[i][v]: # now we meet an intersection
                        last_slot = None
                        tmp_output_pat = None
                        for j in range(1, v+1): # compute the difference reversely since input_pattern is all zero
                            # tmp input pattern:  ----1:V, ---01:V, --001:V, -0001:V
                            # tmp output pattern: ---01:V, --001:V, -0001:V, 00001:V
                            operation = "difference"
                            lhs = SetBuilder("#", last_slot) if last_slot else SetBuilder("N", ID(v))
                            tmp_input_pat = tmp_output_pat if tmp_output_pat else '-'*v + out_pat[v:]
                            rhs = SetBuilder("N", ID(v-j))
                            tmp_output_pat = '-'*(v-j) + out_pat[(v-j):]
                            # if v-iteration is counting-only, we have to wait until we have backed to N(v0), 
                            last_slot = add_set_op(operation, lhs, rhs, tmp_input_pat, tmp_output_pat, do_count=(v==pattern_size-2 and j==v), current=upper_bound_key, depend_by=require_lvl)
                    else:
                        # still get 0
                        allocator.set_mapping(out_pat, ID(-1))
                else:
                    operation = "intersect" if A[i][v] else "difference"
                    lhs = SetBuilder("#", in_slot)
                    rhs = SetBuilder("N", ID(v))
                    add_set_op(operation, lhs, rhs, in_pat, out_pat, do_count=(v==pattern_size-2), current=upper_bound_key, depend_by=require_lvl)
    
            if v < pattern_size-2:
                inner_loop = Loop(v+1, "V", indent+"  ")
                loop_block.add_statement(inner_loop)
            return ForLoop(loop_type, v, read_vlist_op, loop_block, indent, loop_pattern = loop_pattern)
    
        root = Loop(0, "V", "  ")
        return root, allocator

class OptimizationPass:
    def __init__(self):
        # enable holistic optimization and ensure consistency
        self.global_status = {}
        self.vertex_symbol = {}
        self.lut_pivot = 0
        self.edge_centric = False

    def config_edge_centric(self):
        self.edge_centric = True

    def convert2edge_centric(self, root):
        next_loop = root.block.get_next_loop()
        next_loop_block = next_loop.block
        next_loop_idx = root.block.get_next_loop_idx()
        assert(next_loop_idx != None)
        if next_loop.loop_type == "FusedForLoop":
            #TODO(mengke) build lut here is an overkill
            return root
        else:
            loop_block = root.block.slice(0, next_loop_idx)
            loop_block.merge(next_loop_block)
            root = ForLoop("inter_warp_edge_parallel_first", None, None, loop_block, "  ", loop_pattern = ":E")
            return root

    def set_pivot(self, pivot):
        self.lut_pivot = pivot

    def select(self):
        """ select a vertex, build LUT with its candidate set """
        return self.lut_pivot
    
    def get_prefix_set(self, allocator, pattern, lut_lvl, lut_pattern, need_restrict=False):
        pat, pfmt = pattern.split(":")
        if pfmt == "B":
            if pat[:-1] == lut_pattern: # get one row from lut
                vertex_idx = ID(str(lut_lvl)+"_idx")
                connected = (pat[-1] == '1')
                # If this set is directly the candidate set, we should apply restriction, otherwise we should not.
                upper_bound = self.global_status.get(vertex_idx.id(), -1) if need_restrict else -1
                # N.B. special hack to fix the missing upper bound of get one row from lut
                #assert(self.vertex_symbol.get(upper_bound, False))
                return SetBuilder("L", vertex_idx, connected=connected, ex_parameter=ID(upper_bound))
            bid = allocator.map.get(pat + ":" + pfmt, None)
            if bid:
                return SetBuilder("@", bid, ex_parameter=ID(-1))
        elif pfmt == "V":
            if pat == "1":
                return SetBuilder("N", ID(0))
            elif all(c == "-" for c in pat[:-1]) and pat[-1] == "1":
                # for intput pattern like "----1"
                return SetBuilder("N", ID(len(pat)-1))
            bid = allocator.map.get(pat + ":" + pfmt, None)
            if bid:
                return SetBuilder("#", bid)
        elif pfmt == "I":
            # index always materialized in buffer
            bid = allocator.map.get(pat + ":" + pfmt, None)
            if bid:
                return SetBuilder("@", ID(-1), ex_parameter=bid)
        else:
            return None
    
    def get_new_buffer(self, fmt, allocator, pattern, cur_lvl, depend_by):
        bid = allocator.get_empty_buffer(fmt, cur_lvl, depend_by)
        allocator.set_mapping(pattern, bid)
        return bid
    
    def will_be_loopset(self, A, pat):
        pat = pat.split(":")[0]
        lvl = len(pat)
        t_pattern = "".join(map(lambda x: str(x), A[lvl][:lvl])) 
        if t_pattern == pat:
            return True
        else: 
            return False

    def optimize(self, A, O, root, allocator):
        print(f"{allocator.map=}")
        print("> Before optimized")
        print(root)
 
        c_lvl = 0
        c_loop = root

        t_lvl = self.select()
        if t_lvl == 0 or t_lvl > len(A)-2: 
            # lvl == 0 is ALL_VERTEX, can not build LUT
            # lvl > len(A) - 2, no need to build LUT
            return root, allocator
        t_pattern = "".join(map(lambda x: str(x), A[t_lvl][:t_lvl])) 

        inner_most_loop = None

        self.vertex_symbol[str(0)] = True
        while c_loop != None:
            #for i in range(c_loop.block.size()):
            statement_idx = 0;
            while True:
                if statement_idx >= c_loop.block.size():
                    break
                # Process statements of this Loop
                statement = c_loop.block.get_statement(statement_idx)
                if statement.type == "ForLoop" or statement.type == "FusedForLoop":
                    # Process ForLoop
                    if c_lvl + 1 == t_lvl:
                        # build LUT for target forloop, and change loop to index version
                        vertex_set = statement.vertex_set
                        build_lut = Instruction(".LUT", vertex_set)
                        statement = FusedForLoop(build_lut, statement)
                        c_loop.block.replace_statement(statement_idx, statement)
                        # symbols like v0, v1, v2, v3 will not be generated
                        self.vertex_symbol[str(c_lvl+1)] = False
                    else:
                        # try change loop set to index version
                        loop_pattern = statement.loop_pattern
                        desired_pattern = loop_pattern[:-1] + "I" 
                        vmap_idx = self.get_prefix_set(allocator, desired_pattern, t_lvl, t_pattern)
                        # if we have the index version of loop_set, switch to it
                        if vmap_idx:
                            statement.vertex_set = vmap_idx
                            statement.vertex = str(statement.vertex) + "_idx"
                            statement.loop_pattern = desired_pattern
                            c_loop.block.replace_statement(statement_idx, statement)
                            # symbols like v0, v1, v2, v3 will not be generated
                            self.vertex_symbol[str(c_lvl+1)] = False
                        else:
                            # TODO(mengke): we may not have the vid version of this loop_set
                            # symbols like v0, v1, v2, v3 will be generated
                            self.vertex_symbol[str(c_lvl+1)] = True
                    if statement.block.get_next_loop_idx() == None:
                        # if this is inner_most_loop, we may change it to intra_warp_paralizem
                        inner_most_loop = statement

                elif statement.type == "SetOperation":
                    # Process set opertaion
                    c_in_pattern = statement.in_pattern
                    c_out_pattern = statement.out_pattern

                    # step 1: check the ininitialized bitmap set
                    if c_in_pattern[:-2] == t_pattern:
                        # if the input is same as the LUT, we can rewrite this set operation into "get row form lut"
                        if inner_most_loop:
                            vertex_idx = ID(str(statement.right_operand.operand) + "_idx")
                            lhs = SetBuilder("L", vertex_idx, connected=c_out_pattern[-3]=='1') 
                            statement.left_operand = lhs
                            statement.right_operand = None
                            if statement.upper_bound >= 0:
                                statement.set_upper_bound(str(statement.upper_bound) + "_idx")
                            if inner_most_loop:
                                # change inner-most loop to intra_warp parallel if we are using bitmap counting
                                inner_most_loop.loop_type = "intra_warp_parallel"
                            c_loop.block.replace_statement(statement_idx, statement)
                        else:
                            if statement.upper_bound >= 0:
                                vertex_idx = ID(str(statement.right_operand.operand) + "_idx")
                                self.global_status[vertex_idx.id()] = statement.upper_bound
                                # TODO(mengke): now believe nvcc will ignore no-usage memory access
                                vertex = str(statement.upper_bound)
                                if not self.vertex_symbol.get((vertex), False):
                                   vertex_idx = ID(vertex + "_idx")
                                   ins = Instruction("vid", Constant(vertex_idx.id()))
                                   c_loop.block.insert_statement(statement_idx, ins)
                                   statement_idx += 1
                                   self.vertex_symbol[vertex] = True

                            c_loop.block.replace_statement(statement_idx, NOP())
                        allocator.del_mapping(c_out_pattern)
                        c_out_pattern = c_out_pattern[:-1] + "B"
                    else:
                        # step 2: fix the depdendency of other set operation
                        is_rhs_in_lut = (c_loop.loop_pattern[-1] == "I")
                        if is_rhs_in_lut:
                            # N(vi) is in lut, meaning it always have three formats: B, I, V
                            desired_pattern = "" 
                            lhs = None
                            for pfmt in ["B", "I", "V"]:
                                desired_pattern = c_in_pattern[:-1] + pfmt
                                lhs = self.get_prefix_set(allocator, desired_pattern, t_lvl, t_pattern)
                                if lhs:
                                    break
                            assert(lhs != None)
                            if desired_pattern[-1] == "I":
                                # if bitmap is not existed, recover it from index
                                desired_pattern = c_in_pattern[:-1] + "B"
                                bid = self.get_new_buffer("B", allocator, desired_pattern, c_lvl, sys.maxsize) # TODO(mengek): this can be reused
                                ins = Instruction(".B", lhs, bid) # build index from vmap into buffer with ID=bid
                                c_loop.block.insert_statement(statement_idx, ins)
                                statement_idx += 1
                                lhs = self.get_prefix_set(allocator, desired_pattern, t_lvl, t_pattern)

                            if desired_pattern[-1] == "B":
                                vertex_idx = ID(str(statement.right_operand.operand) + "_idx")
                                rhs = SetBuilder("L", vertex_idx, connected=c_out_pattern[-3]=='1') 
                                statement.left_operand = lhs
                                statement.right_operand = rhs
                                statement.in_pattern = c_in_pattern[:-1] + "B"
                                statement.out_pattern = c_out_pattern[:-1] + "B"
                                c_out_pattern = statement.out_pattern
                                allocator.del_mapping(c_out_pattern)
                                bid = self.get_new_buffer("B", allocator, statement.out_pattern, c_lvl, sys.maxsize) # TODO(mengek): this can be reused
                                statement.output = Constant(bid.id())
                                if statement.upper_bound >= 0:
                                    statement.set_upper_bound(str(statement.upper_bound) + "_idx")
                                if inner_most_loop:
                                    # change inner-most loop to intra_warp parallel if we are using bitmap counting
                                    inner_most_loop.loop_type = "intra_warp_parallel"
                                c_loop.block.replace_statement(statement_idx, statement)
                            elif desired_pattern[-1] == "V":
                                # prefix is not in LUT, thus we have to use original neighbor list
                                c_in_pat = c_in_pattern.split(":")[0]
                                if c_in_pat[0] == "-": # we meet a lazy materialization like ---01
                                    if all(x=="-" for x in c_in_pat[:-1]) and c_in_pat[-1] == '1':
                                        # something like #3 = difference(N(2), N(v1)), both rhs and lhs may not have symbol
                                        if not self.vertex_symbol.get(str(statement.left_operand.operand), False):
                                            vertex_idx = ID(str(statement.left_operand.operand) + "_idx")
                                            ins = Instruction("vid", Constant(vertex_idx.id()))
                                            c_loop.block.insert_statement(statement_idx, ins)
                                            statement_idx += 1
                                            self.vertex_symbol[str(statement.left_operand.operand)] = True
                                        if not self.vertex_symbol.get(str(statement.right_operand.operand), False):
                                            vertex_idx = ID(str(statement.right_operand.operand) + "_idx")
                                            ins = Instruction("vid", Constant(vertex_idx.id()))
                                            c_loop.block.insert_statement(statement_idx, ins)
                                            statement_idx += 1
                                            self.vertex_symbol[str(statement.right_operand.operand)] = True
                                        vertex_l = ID(str(statement.left_operand.operand))
                                        lhs = SetBuilder("N", vertex_l)
                                        vertex_r = ID(str(statement.right_operand.operand))
                                        rhs = SetBuilder("N", vertex_r)
                                        statement.left_operand = lhs
                                        statement.right_operand = rhs
                                        statement.in_pattern = c_in_pattern[:-1] + "V"
                                        statement.out_pattern = c_out_pattern[:-1] + "V"
                                        c_loop.block.replace_statement(statement_idx, statement)
                                    else:
                                        # something like #4 = difference(#3, N(v1)), rhs may not have symbol
                                        if not self.vertex_symbol.get(str(statement.right_operand.operand), False):
                                            vertex_idx = ID(str(statement.right_operand.operand) + "_idx")
                                            ins = Instruction("vid", Constant(vertex_idx.id()))
                                            c_loop.block.insert_statement(statement_idx, ins)
                                            statement_idx += 1
                                            self.vertex_symbol[str(statement.right_operand.operand)] = True
                                        vertex_r = ID(str(statement.right_operand.operand))
                                        rhs = SetBuilder("N", vertex_r)
                                        statement.right_operand = rhs
                                        statement.in_pattern = c_in_pattern[:-1] + "V"
                                        statement.out_pattern = c_out_pattern[:-1] + "V"
                                        c_loop.block.replace_statement(statement_idx, statement)
                                else:
                                    if not self.vertex_symbol.get(str(statement.right_operand.operand), False):
                                        vertex_idx = ID(str(statement.right_operand.operand) + "_idx")
                                        ins = Instruction("vid", Constant(vertex_idx.id()))
                                        c_loop.block.insert_statement(statement_idx, ins)
                                        statement_idx += 1
                                        self.vertex_symbol[str(statement.right_operand.operand)] = True
                                    vertex = ID(str(statement.right_operand.operand))
                                    rhs = SetBuilder("N", vertex)
                                    statement.left_operand = lhs
                                    statement.right_operand = rhs
                                    statement.in_pattern = c_in_pattern[:-1] + "V"
                                    statement.out_pattern = c_out_pattern[:-1] + "V"
                                    c_loop.block.replace_statement(statement_idx, statement)
                        else:
                            # N(vi) is not in lut, meaning it only have one format: V
                            desired_pattern = "" 
                            lhs = None
                            for pfmt in ["V", "I", "B"]:
                                desired_pattern = c_in_pattern[:-1] + pfmt
                                lhs = self.get_prefix_set(allocator, desired_pattern, t_lvl, t_pattern)
                                if lhs:
                                    break
                            #print(f"{allocator.map=}, {desired_pattern=}")
                            #print(root)
                            assert(lhs != None)
                            if desired_pattern[-1] == "B":
                                # if index is not existed, recover it from Bitmap
                                desired_pattern = c_in_pattern[:-1] + "I"
                                bid = self.get_new_buffer("A", allocator, desired_pattern, c_lvl, sys.maxsize) # TODO(mengek): this can be reused
                                ins = Instruction(".I", lhs, bid) # build index from vmap into buffer with ID=bid
                                c_loop.block.insert_statement(statement_idx, ins)
                                statement_idx += 1
                                lhs = self.get_prefix_set(allocator, desired_pattern, t_lvl, t_pattern)

                            if desired_pattern[-1] == "V":
                                pass
                            elif desired_pattern[-1] == "I":
                                statement.left_operand = lhs
                                statement.in_pattern = c_in_pattern[:-1] + "V"
                                statement.out_pattern = c_out_pattern[:-1] + "I"
                                if statement.upper_bound >= 0:
                                    vertex = str(statement.upper_bound)
                                    if not self.vertex_symbol.get((vertex), False):
                                       vertex_idx = ID(vertex + "_idx")
                                       ins = Instruction("vid", Constant(vertex_idx.id()))
                                       c_loop.block.insert_statement(statement_idx, ins)
                                       statement_idx += 1
                                       self.vertex_symbol[vertex] = True
                                allocator.update_mapping(c_out_pattern, c_out_pattern[:-1]+"I")
                                c_out_pattern = statement.out_pattern
                                c_loop.block.replace_statement(statement_idx, statement)

                    # step 3: check whether we should build index
                    if not inner_most_loop and self.will_be_loopset(A, c_out_pattern) and c_out_pattern[-1] == "B":
                        # if we need it's index version
                        desired_pattern = c_out_pattern[:-1] + "I"
                        if allocator.require('A', desired_pattern, c_lvl, sys.maxsize):
                            # if we don't have the index version
                            bitmap_pattern = c_out_pattern[:-1] + "B"
                            # the only situation that we need apply restriction
                            vmap = self.get_prefix_set(allocator, bitmap_pattern, t_lvl, t_pattern, need_restrict=True)
                            if vmap:
                                # if we have the bitmap version of out_pattern, use it to build index
                                bid = allocator.map[desired_pattern]
                                ins = Instruction(".I", vmap, bid) # build index from vmap into buffer with ID=bid
                                c_loop.block.insert_statement(statement_idx+1, ins) 
                                statement_idx += 1
                statement_idx += 1
            c_loop = c_loop.block.get_next_loop()
            c_lvl += 1

        if self.edge_centric and self.select() != 1:
            root = self.convert2edge_centric(root)
        return root, allocator

class CodeGenerator:
    def load_template(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                file_content = file.read()
            return file_content
        except Exception as e:
            return f"An error occurred: {e}"

    def config(self, allocator, name="generated_kernel"):
        code_config = {}
        code_config["name"] = name
        code_config["max_slot_num"] = max(1, allocator.max_slot_num())
        code_config["max_bitmap_num"] = max(1, allocator.max_bitmap_num())
        return code_config

    def codegen(self, root, code_config, template_fn, output_fn):
        template = self.load_template(template_fn)
        d = dict(
                KERNEL_NAME = code_config["name"],
                WARP_LEVEL_DFS_CODE = root.code(),
                SLOT_CAPACITY = code_config["max_slot_num"],
                BITMAP_CAPACITY = code_config["max_bitmap_num"]
            )
        generated_kernel = template.format(**d)

        try:
            with open(output_fn, 'w+') as f:
                f.write(generated_kernel)
        except Exception as e:
            return f"An error occurred: {e}"

def main():
    parser = argparse.ArgumentParser(description='GPM code generator')
    parser.add_argument('-p', '--pattern', required=True, help="[required] a pattern description file.")
    parser.add_argument('-t', '--template', required=True, help="[required] a template file.")
    parser.add_argument('-n', '--name', required=True, help="[required] the name of the generated kernel.")
    parser.add_argument('-c', '--output', required=True, help="[required] wirte genrated kernel to this path.")
    parser.add_argument('-l', '--lut', required=False, help="choose one vertex to generate LUT.")
    parser.add_argument('-e', '--edge-centric', required=False, action='store_true', help="forced use edge-centric.")
    args = parser.parse_args()

    pattern_parser = PatternParser()
    A, O = pattern_parser.parse(args.pattern)
    
    lower_pass =  LoweringPass()
    root, allocator = lower_pass.lower(A, O)

    opt_pass = OptimizationPass()
    if args.edge_centric:
        opt_pass.config_edge_centric()
    opt_pass.set_pivot(int(args.lut))
    root, allocator = opt_pass.optimize(A, O, root, allocator)
    print("> After optimize")
    print(root)

    generator = CodeGenerator()
    code_config = generator.config(allocator, args.name)
    generator.codegen(root, code_config, args.template, args.output)
    #print("> Code generation")
    #print(root.code())

if __name__ == '__main__':
    main()
