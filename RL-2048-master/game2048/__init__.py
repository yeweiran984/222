from gymnasium.envs.registration import register

register(
    id="Game2048-v0",
    entry_point="game2048.envs:Game2048Env",
)