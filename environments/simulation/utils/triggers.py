def interval_trigger(step_interval: int):
    return lambda step: step % step_interval == 0


def dummy_episode_trigger(episode_idx: int):
    return lambda episode_idx: True


def episode_trigger(episode_interval: int):
    return lambda episode: episode % episode_interval == 0
