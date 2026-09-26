"""Unchanged weak spatial TV regularizer for the ungated velocity field."""

def velocity_total_variation(velocity):
    velocity = velocity.float()
    return .5 * ((velocity[..., 1:] - velocity[..., :-1]).abs().mean()
                 + (velocity[..., 1:, :] - velocity[..., :-1, :]).abs().mean())

