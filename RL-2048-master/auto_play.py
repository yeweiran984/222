from trpo_game2048 import TRPO
import gymnasium as gym

MODEL_PATH = "best_trpo_game2048_model_38208it_65536batch.pth"
ACTIONS = {0: "Left", 1: "Right", 2: "Up", 3: "Down"}

if __name__ == "__main__":
    env = gym.make("Game2048-v0", render_mode="human")
    agent = TRPO(env)
    agent.load_model(MODEL_PATH)
    state, _ = env.reset()
    done = False
    while not done:
        action, _ = agent.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()