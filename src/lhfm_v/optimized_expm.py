"""Order-24 signed affine exponential; saved discrete adjoint and CUDA replay.

Every call owns its tape. Static graph buffers must never be saved directly
for later backward because twenty intervals share them during a rollout.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from .frozen_expm import (TERMS, MAX_SCALED_SHIFT, SHIFT_MULTIPLIER,
                         MAX_INTERNAL_PIECES, exponential_stage, stencil_coefficients)


def coefficients(source, velocity, q):
    return stencil_coefficients(source, velocity, q)


def poisson_weights(mean):
    weight = torch.exp(-mean)
    weights = [weight]
    for order in range(1, TERMS+1):
        weight = weight * mean / order
        weights.append(weight)
    return torch.stack(weights)


def backward_stage(adjoint, coeff_grad, value, coeff, output_grad, weight):
    center, left, right, above, below, left2, right2, above2, below2, _ = coeff.unbind(0)
    coeff_grad = coeff_grad + torch.stack((
        adjoint*value,
        adjoint*F.pad(value[..., :-1], (1, 0, 0, 0)),
        adjoint*F.pad(value[..., 1:], (0, 1, 0, 0)),
        adjoint*F.pad(value[..., :-1, :], (0, 0, 1, 0)),
        adjoint*F.pad(value[..., 1:, :], (0, 0, 0, 1)),
        adjoint*F.pad(value[..., :-2], (2, 0, 0, 0)),
        adjoint*F.pad(value[..., 2:], (0, 2, 0, 0)),
        adjoint*F.pad(value[..., :-2, :], (0, 0, 2, 0)),
        adjoint*F.pad(value[..., 2:, :], (0, 0, 0, 2)), adjoint))
    # P^T requires multiplying spatially varying coefficients BEFORE shifts.
    adjoint = (center*adjoint
        + F.pad((left*adjoint)[..., 1:], (0, 1, 0, 0))
        + F.pad((right*adjoint)[..., :-1], (1, 0, 0, 0))
        + F.pad((above*adjoint)[..., 1:, :], (0, 0, 0, 1))
        + F.pad((below*adjoint)[..., :-1, :], (0, 0, 1, 0))
        + F.pad((left2*adjoint)[..., 2:], (0, 2, 0, 0))
        + F.pad((right2*adjoint)[..., :-2], (2, 0, 0, 0))
        + F.pad((above2*adjoint)[..., 2:, :], (0, 0, 0, 2))
        + F.pad((below2*adjoint)[..., :-2, :], (0, 0, 2, 0))
        + weight*output_grad)
    return adjoint, coeff_grad


def forward_tape(image, coeff, weights, stage=exponential_stage):
    value, result = image, weights[0]*image
    tape = [value]
    for order in range(1, TERMS+1):
        value, result = stage(value, result, *coeff.unbind(0), weights[order])
        tape.append(value)
    return result, torch.stack(tape)


def discrete_backward(tape, coeff, weights, output_grad, stage=backward_stage):
    adjoint, coeff_grad = weights[-1]*output_grad, torch.zeros_like(coeff)
    for order in range(TERMS, 0, -1):
        adjoint, coeff_grad = stage(adjoint, coeff_grad, tape[order-1], coeff,
                                    output_grad, weights[order-1])
    return adjoint, coeff_grad


class _Graph:
    @torch.no_grad()
    def __init__(self, image, coeff, weights, forward_stage, adjoint_stage):
        self.image = torch.empty_like(image, memory_format=torch.contiguous_format)
        self.coeff, self.weights = torch.empty_like(coeff), torch.empty_like(weights)
        self.tape = image.new_empty((TERMS+1, *image.shape))
        self.output_grad = torch.empty_like(self.image)
        self.image.copy_(image)
        self.coeff.copy_(coeff)
        self.weights.copy_(weights)
        self.tape.zero_()
        self.output_grad.fill_(1)
        stream = torch.cuda.Stream(device=image.device)
        stream.wait_stream(torch.cuda.current_stream(image.device))
        with torch.no_grad(), torch.autocast('cuda', enabled=False), torch.cuda.stream(stream):
            # Compile before capture; pools remain separate to avoid aliasing.
            for _ in range(2):
                forward_tape(self.image, self.coeff, self.weights, forward_stage)
                discrete_backward(self.tape, self.coeff, self.weights, self.output_grad, adjoint_stage)
            stream.synchronize()
            self.forward_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.forward_graph, stream=stream):
                self.result, self.forward_saved = forward_tape(
                    self.image, self.coeff, self.weights, forward_stage)
            self.backward_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.backward_graph, stream=stream):
                self.image_grad, self.coeff_grad = discrete_backward(
                    self.tape, self.coeff, self.weights, self.output_grad, adjoint_stage)
        torch.cuda.current_stream(image.device).wait_stream(stream)
        self.forward_replays = self.backward_replays = 0

    def forward(self, image, coeff, weights):
        self.image.copy_(image)
        self.coeff.copy_(coeff)
        self.weights.copy_(weights)
        self.forward_graph.replay()
        self.forward_replays += 1
        return self.result.clone(), self.forward_saved.clone()

    def backward(self, tape, coeff, weights, output_grad):
        self.tape.copy_(tape)
        self.coeff.copy_(coeff)
        self.weights.copy_(weights)
        self.output_grad.copy_(output_grad)
        self.backward_graph.replay()
        self.backward_replays += 1
        return self.image_grad.clone(), self.coeff_grad.clone()


class _Piece(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, coeff, weights, graph):
        result, tape = (forward_tape(image, coeff, weights) if graph is None
                        else graph.forward(image, coeff, weights))
        ctx.save_for_backward(tape, coeff, weights)
        ctx.graph = graph
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, output_grad):
        tape, coeff, weights = ctx.saved_tensors
        image_grad, coeff_grad = (discrete_backward(tape, coeff, weights, output_grad)
            if ctx.graph is None else ctx.graph.backward(tape, coeff, weights, output_grad))
        return image_grad, coeff_grad, None, None


class SavedExponential:
    """One-stream runtime, with a graph cache owned by one Execution instance."""
    def __init__(self, *, cuda_graph=True):
        self.cuda_graph, self.graphs = cuda_graph, {}
        self.coefficients, self.weights = coefficients, poisson_weights
        self.forward_stage, self.backward_stage = exponential_stage, backward_stage
        if cuda_graph:
            # Compile bounded stencil stages, never the entire recurrence.
            def compile_fn(fn):
                return torch.compile(fn, fullgraph=True, dynamic=False,
                                     options={'triton.cudagraphs': False})
            self.coefficients = compile_fn(coefficients)
            self.weights = compile_fn(poisson_weights)
            self.forward_stage = compile_fn(exponential_stage)
            self.backward_stage = compile_fn(backward_stage)

    def __call__(self, image, source, velocity, dt):
        dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
        with torch.autocast(image.device.type, enabled=False):
            image, source, velocity = image.to(dtype), source.to(dtype), velocity.to(dtype)
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError('positive finite frozen interval required')
            rate = float(velocity.detach().abs().sum(1).amax())
            if not math.isfinite(rate):
                raise FloatingPointError('nonfinite frozen transport field')
            rate = SHIFT_MULTIPLIER*max(1., rate)
            pieces = max(1, math.ceil(dt*rate/MAX_SCALED_SHIFT))
            if pieces > MAX_INTERNAL_PIECES:
                raise FloatingPointError('frozen transport rate exceeds numerical work budget; no velocity clipping')
            coeff = self.coefficients(source, velocity, image.new_tensor(rate))
            weights = self.weights(image.new_tensor(dt*rate/pieces))
            graph = None
            if self.cuda_graph:
                if image.device.type != 'cuda':
                    raise ValueError('CUDA Graph execution requires CUDA')
                key = (image.device, image.dtype, tuple(image.shape))
                if key not in self.graphs:
                    self.graphs[key] = _Graph(image, coeff, weights, self.forward_stage, self.backward_stage)
                graph = self.graphs[key]
            for _ in range(pieces):
                image = _Piece.apply(image, coeff, weights, graph)
            return image
