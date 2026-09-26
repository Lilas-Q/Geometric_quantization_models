"""Fixed official update budget with explicit nonuniform validation nodes."""
import math


def is_validation_step(step, config):
    return type(step) is int and step in config['validation_steps']


def assess_history(history, config):
    if not isinstance(history, list):
        raise ValueError('validation history must be a list')
    nodes = config['validation_steps']
    if len(history) > len(nodes):
        raise ValueError('validation exceeds fixed budget')
    best = None
    for index, row in enumerate(history):
        score = row.get('mse_raw')
        if (row.get('step') != nodes[index] or type(row.get('step')) is not int
                or type(score) not in (int, float) or not math.isfinite(score) or score < 0):
            raise ValueError('validation must contain every configured node in order')
        if best is None or score < best['mse_raw']:
            best = row
    last = history[-1]['step'] if history else 0
    terminal = last == config['stop_steps']
    return dict(action='stop' if terminal else 'train_fixed',
        reason='fixed_budget_complete' if terminal else 'fixed_official_budget',
        authorized_until=config['stop_steps'], last_validation_step=last,
        best_step=None if best is None else best['step'],
        best_mse_raw=None if best is None else best['mse_raw'])


def validate_saved_control(history, step, config):
    decision = assess_history(history, config)
    if type(step) is not int or not 0 <= step <= config['stop_steps']:
        raise ValueError('checkpoint outside fixed budget')
    required = [node for node in config['validation_steps'] if node < step]
    saved = [row['step'] for row in history]
    if (saved[:len(required)] != required or decision['last_validation_step'] > step):
        raise ValueError('checkpoint skipped required validation')
    return decision
