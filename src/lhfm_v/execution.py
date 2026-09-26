"""Keep history activations by default; fuse recall/control and QKV execution.

The optional checkpoint switch is explicit for a separately qualified low-memory
configuration. There is no silent OOM fallback and no truncated rollout gradient.
"""
from contextlib import contextmanager
import torch
import torch._dynamo.config as dynamo_config
import torch._functorch.config as functorch_config
from torch.utils.checkpoint import checkpoint
from .optimized_expm import SavedExponential


@contextmanager
def compiler_context(enabled):
    if not enabled:
        yield
        return
    with functorch_config.patch(backward_pass_autocast='off'), dynamo_config.patch(
            suppress_errors=False, fail_on_recompile_limit_hit=True):
        yield


class Execution:
    def __init__(self, model, *, precision='bf16', compile=False,
                 backend='inductor', activation_checkpoint=False, supervise_plan=False):
        if precision not in ('bf16','fp32') or backend not in ('inductor','aot_eager'):
            raise ValueError('unsupported precision or backend')
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('FP32 master parameters required')
        if type(activation_checkpoint) is not bool:
            raise ValueError('activation_checkpoint must be boolean')
        self.model, self.precision, self.compiled = model, precision, compile
        if type(supervise_plan) is not bool:
            raise ValueError('supervise_plan must be boolean')
        self.supervise_plan = supervise_plan
        self.plan_head = model.predict_plan if supervise_plan else None
        self.activation_checkpoint = activation_checkpoint
        self.prepare, self.fields, self.state = model.prepare, model.fields, model.update_state
        self.frame = model.begin_frame
        self.memory_encode = model.memory.encode
        self.transport = None
        if compile:
            options = dict(backend=backend, fullgraph=True, dynamic=False)
            cuda = next(model.parameters()).device.type == 'cuda'
            if cuda and backend == 'inductor':
                # Let Inductor own both compiled autograd and graph lifetimes.
                # This mode changes launch scheduling, not kernel arithmetic or
                # autotuning options. Keep all ordinary compiler guards.
                options['mode'] = 'reduce-overhead'
            self.prepare = torch.compile(self.prepare, **options)
            if supervise_plan:
                self.plan_head = torch.compile(self.plan_head, **options)
            self.fields = torch.compile(self.fields, **options)
            self.frame = torch.compile(self.frame, **options)
            self.memory_encode = torch.compile(self.memory_encode, **options)
            self.state = torch.compile(self.state, **options)
            if cuda and backend != 'inductor':
                raise ValueError('formal CUDA execution requires Inductor')
            self.transport = SavedExponential(cuda_graph=cuda)

    def __call__(self, context, horizon=10):
        prepare = self.prepare
        if self.activation_checkpoint and torch.is_grad_enabled():
            def prepare(context):
                return checkpoint(self.prepare,context,use_reentrant=False,preserve_rng_state=False)
        # Match per-call FP32 parameter-gradient accumulation between eager and
        # compiled recurrent kernels; cached BF16 weight casts otherwise merge
        # repeated-call gradients in BF16 only on the eager path.
        with compiler_context(self.compiled), torch.autocast(context.device.type,dtype=torch.bfloat16,
                                                            enabled=self.precision=='bf16',cache_enabled=False):
            return self.model.rollout(context,horizon,prepare_fn=prepare,fields_fn=self.fields,
                                      state_fn=self.state,frame_fn=self.frame,transport_fn=self.transport,
                                      memory_encode_fn=self.memory_encode,
                                      **(dict(supervise_plan=True,plan_head_fn=self.plan_head)
                                         if self.supervise_plan else {}))
