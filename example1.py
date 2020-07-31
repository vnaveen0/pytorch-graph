import torch
import torchvision
model = torchvision.models.resnet18()
inp   = torch.zeros([64, 3, 7, 7])
trace, grad = torch.jit._get_trace_graph(model, inp)

for node in trace.nodes():
    op = node.kind();
    outputs = [o.unique() for o in node.outputs()]
    inputs = [i.unique() for i in node.inputs()]
    print(inputs)
    # graph_node = # Create a node for current AST Node
    # parsed_graph[graph_node.id] = graph_node
    # # Reference:
    # # https://github.com/waleedka/hiddenlayer/blob/master/hiddenlayer/pytorch_builder.py
    # for to in graph.nodes():
    #     to_inputs = [i.unique() for i in to.inputs()]
    #     edges = set(outputs) & set(to_inputs)
    #     if edges:
    #         graph_node.edges.append(Edge(to, edges))
