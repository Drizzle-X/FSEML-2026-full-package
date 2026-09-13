from model.replay_Fse import FSEReplayBuffer
from model.replay_simple import SimpleReplayBuffer


class ReplayFactory:
    @staticmethod
    def build(args):
        mode = getattr(args, "replay_mode", "simple").lower()

        builders = {
            "simple": SimpleReplayBuffer,
            "fse": FSEReplayBuffer,
        }
        if mode not in builders:
            raise ValueError(f"Unsupported replay mode: {mode}")
        return builders[mode](args)
