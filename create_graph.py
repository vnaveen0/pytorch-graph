import torch
import torchvision
model = torchvision.models.resnet18()
inp   = torch.zeros([64, 3, 7, 7])
trace, grad = torch.jit._get_trace_graph(model, inp)

# Even though we have the trace, it is not a real graph that is traversable. So we need to construct the graph using the nodes in the AST. Here are some useful functions:
#
# trace.nodes(): returns a stream of AST node (iterator)
# trace.param_node(): the parameter passed to the forward pass. For an AST node object, we have
# node.kind(): returns the operator (scope::name)
# node.outputs(): returns a stream of AST nodes that are output variables of the node
# node.inputs(): returns a stream of AST nodes who are passed as input arguments to the node.
# node.unique(): returns a unique identifier of the node

def parse_nodes():
    for node in graph.nodes():
            op = node.kind();
            outputs = [o.unique() for o in node.outputs()]
            inputs  = [i.unique() for i in node.inputs()]
            graph_node = # Create a node for current AST Node
            parsed_graph[graph_node.id] = graph_node
            # Reference:
            # https://github.com/waleedka/hiddenlayer/blob/master/hiddenlayer/pytorch_builder.py
            for to in graph.nodes():
                to_inputs = [i.unique() for i in to.inputs()]
                edges = set(outputs) & set(to_inputs)
                if edges:
                    graph_node.edges.append(Edge(to, edges))


class Node:
    '''
        A Computation Graph Node Representation
    '''

    def __init__(self, name, op, params, shape, inputs, outputs, output_size=None):
        self.id = name
        self.op = op
        self.params = params
        self.shape = shape
        self.outputs = outputs
        self.inputs = inputs
        self.checkpoint = False
        self.output_size = output_size
        self.edges = []

    def adjacent_nodes(self, graph):
        if (self.edges):
            for name, src_name in self.edges:
                yield graph[name], src_name

    def get_output_size(self):
        raise NotImplementedError()

    def to_python(self, ctx: dict, src=False, inline=True):
        raise NotImplementedError()

    def create_name(node):
        return node.kind() \
               + '~>' \
               + hashlib.md5(str(reduce(
            lambda x, y: x + y,
            [str(x) for x in sorted((y.unique() for y in node.outputs()))]
        )).encode()).hexdigest()

    # output.type()  # can give us the type
    # output.type().sizes()  # can give us the shape


    def get_shape(node) -> dict:
        outputs = dict()
        for o in node.outputs():
            typeIs = o.type()
            outputs[o.unique()] = Shape(type=re.match(r'\w+', typeIs.str()).group(), sizes=tuple(typeIs.sizes()))
        return outputs


    def get_value(node) -> dict:
        outputs = dict()
        for o in node.outputs():
            typeIs = o.type().str()
            value = o.toIValue()
            outputs[o.unique()] = Value(type=typeIs, value=value, \
                                        sizes=len(list(node.outputs())) if typeIs.endswith('[]') else 1)
        return outputs

    # graph_node = AtenNode(create_name(node), op, params, get_shape(node), inputs, outputs)

